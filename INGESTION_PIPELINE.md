# Ingestion Pipeline

Every audio ingestion path in this codebase, traced from the HTTP request that starts it
to what lands in Postgres. There are three, and they are not three copies of the same
code — only clip upload runs the full 10-stage pipeline; live and F1 radio run a cut-down
ASR+tone path with no diarization or identification.

```
POST /v1/clips              ─┐
POST /v1/races/{id}/clips    ├─> ingest.ingest_clip() ──> queue.enqueue_clip() ─┐
POST /v1/clips/{id}/reprocess┘                                                  │
                                                                                  v
POST /v1/live/{sid}/chunk  ──> queue.enqueue_live_chunk() ──────────────────>  Redis
                                                                                  │  (arq)
POST /v1/f1/ingest        ──> queue.enqueue_f1_radio()  ────────────────────────┤
                                                                                  v
                                                                       worker (arq consumer)
                                                                     worker/worker/main.py
                                                              ┌───────────────┼───────────────┐
                                                              v               v               v
                                                     process_clip_job  transcribe_live   analyze_f1_radio_job
                                                     (full pipeline)   _chunk_job        (ASR + tone, no diar)
```

All three job types share one `ModelPool` and one `asyncio.Semaphore(1)`
(`ctx["gpu_sem"] == ctx["live_sem"]`, set in `worker/worker/main.py:startup`) — jobs never
run concurrently against the models, because pyannote/CTranslate2/silero are not
thread-safe and two jobs sharing them concurrently corrupts the process heap. One worker
process = one model lane; scale by running more worker processes, not more concurrency
inside one.

---

## 1. Clip upload (the full pipeline)

**Entry points**, all funneling through the same function so nothing added there can
diverge between them (`api/app/services/ingest.py`):

| Route | Handler |
|---|---|
| `POST /v1/clips` | `api/app/routers/clips.py:upload_clip` |
| `POST /v1/races/{race_id}/clips` | `api/app/routers/races.py:upload_race_clip` |
| `POST /v1/clips/{clip_id}/reprocess` | `api/app/routers/clips.py:reprocess_clip` — re-enqueues an existing clip, doesn't re-upload |

### `ingest_clip()` (api process)

1. Stream the upload to `DATA_DIR/tmp/<random>.<ext>`, hashing as it goes; reject over
   `max_upload_mb`.
2. Look up the SHA-256. If it already exists: don't re-store, don't re-enqueue — return
   `duplicate: true`. If a `race_id` was given and differs from the existing clip's, treat
   the re-upload as a *move* into that race (an operator dragging an overlapping folder
   expects the files to land in the race, not be silently dropped).
3. Otherwise: insert a `clips` row, move the temp file to its permanent `raw_path`
   (`common/storage.py`), write an audit event (`clip.upload`), and
   `queue.enqueue_clip(clip_id)` — an arq job on the `process_clip_job` function, with a
   fresh `_job_id` per attempt (arq caches by job id, so reprocessing needs a new one).
4. Returns `{clip_id, job_id, status: "QUEUED", ws_url}` — the caller subscribes to
   `/v1/ws/jobs/{job_id}` for stage-by-stage progress.

### `process_clip_job()` → `process_clip()` (worker process)

`worker/worker/pipeline.py`. Acquires `gpu_sem` (one clip at a time on GPU/CPU), reads
the *current* effective settings (not the startup-cached copy, so Settings-page edits
apply without a worker restart), then:

1. **`_reset_derived_data`** — deletes any prior `quality_metrics` / `vad_regions` /
   `speaker_turns` / `transcripts` (cascades to `utterances`/`words`) /
   `clip_speakers` rows and clears the clip's rolled-up sentiment. Runs even on the first
   pass, so `reprocess_clip` never hits a duplicate-key error or leaves stale rows behind.
2. Inserts a `processing_runs` row (pipeline version, config snapshot, model versions,
   device, `corr_id`) — the audit trail one `GET /v1/clips/{id}/result` reads back.
3. Runs the ten stages below in order. Each stage's start/done is emitted over the
   `job:{clip_id}` pub/sub channel with a progress percentage from fixed `STAGE_WEIGHTS`
   (a rough ETA beats no ETA). A stage that raises `RejectError` marks the clip
   `REJECTED` with a machine-readable code and stops — not retried, this is bad input, not
   a transient failure. Any other exception marks the clip `FAILED` and re-raises (arq
   retries up to `job_max_attempts`). On success: `clips.status = 'COMPLETE'`.

| # | Stage | File | What it does | Can reject with |
|---|---|---|---|---|
| 1 | `VALIDATING` | `stages/validate.py` | `ffprobe`s the raw file off the event loop; checks it has an audio stream, is inside `[min_duration_s, max_duration_s]`, and that the container header duration agrees with the stream duration within 10% (else `CORRUPT`). Writes `duration_s`/`sample_rate`/`channels`/`mime` onto `clips`. | `NO_AUDIO_STREAM`, `TOO_LONG`, `TOO_SHORT`, `CORRUPT` |
| 2 | `PREPROCESSING` | `stages/preprocess.py` | Decodes to a 16kHz mono array, DC-removes, high-pass filters, loudness-normalizes (keeping an `audio_norm` copy for embeddings), denoises (`audio_clean`, used for ASR/diarization). Runs VAD (`ctx.pool.vad`) and computes quality metrics (SNR, clipping ratio, bandwidth, silence ratio). Writes the cleaned WAV to `work_path`, inserts `quality_metrics` and `vad_regions`. | `NO_SPEECH_DETECTED` / `INSUFFICIENT_SPEECH` if total VAD speech is under `min_total_speech_s` |
| 3 | `TRANSCRIBING` | `stages/transcribe.py` | faster-whisper over `audio_clean`, word-level timestamps, `condition_on_previous_text=False` (non-negotiable — prevents feedback-loop hallucination), `vad_filter=False` (VAD already ran in stage 2). Drops hallucinated segments (high `no_speech_prob` + low `avg_logprob`, repetitive text via compression ratio, or exact boilerplate phrases like "thank you."). | `NO_TRANSCRIPT` if ASR produces zero words |
| 4 | `DIARIZING` | `stages/diarize.py` | pyannote `SpeakerDiarization` over `audio_clean`, bounded by `diar_min/max_speakers`. Cleans the raw turns: drops sub-`min_turn_s` slivers, drops "onset phantom" turns (a brief conflicting-speaker blip right at another turn's start — a clustering artifact, not real cross-talk), merges adjacent same-speaker turns, flags genuine overlaps, snaps turn edges onto nearby VAD boundaries. Inserts `speaker_turns`. | `NO_SPEAKER_TURNS` if nothing survives cleanup |
| 5 | `RECONCILING` | `stages/reconcile.py` | Assigns each ASR word to the turn it overlaps most; words with no overlap fall back to nearest-turn-by-midpoint with `speaker_conf=0`. Smooths isolated single-word flips surrounded by the same other speaker. Regroups words into `utterances` (new utterance on speaker change, or a mid-speaker pause after sentence-ending punctuation). Inserts `transcripts` + `utterances` + `words`. | — |
| 6 | `EMBEDDING` | `stages/embed.py` | Per diarization label, greedily packs its longest non-overlap turns up to `embed_target_s`, and — if that reaches `embed_min_s` — runs the speaker-embedding model over the concatenated audio. Too little clean speech → no embedding, `reliability_reason="insufficient_speech"`. Reliability is scored from duration + audio quality (`common/speaker.py`). | — |
| 7 | `IDENTIFYING` | `stages/identify.py` | For each speaker with an embedding: `common.speaker.identify()` against enrolled voiceprints (confident / suggested / unknown / abstained). `unknown` speakers above the reliability floor get auto-clustered (`assign_cluster`) so they can be named later even without a profile match. If `auto_enroll` is on and the match is confident enough, the embedding is folded into that profile's centroid. | — |
| 8 | `POSTPROCESSING` | `stages/postprocess.py` | Computes talk share, longest turn, interruption count, overlap seconds per speaker; inserts `clip_speakers`, back-fills `profile_id`/`cluster_id` onto `utterances`. Sets `clips.needs_review = true` if any speaker is `suggested`/`abstained`, has low reliability, or a `HIGH_OVERLAP`/`POOR_AUDIO_QUALITY` warning fired. Writes `.rttm`/`.srt`/`.vtt`/`.txt`/`.json` exports to disk. | — |
| 9 | `SENTIMENT` | `stages/sentiment.py` | Text sentiment (XLM-R) fused with acoustic tone (`worker/worker/audio/tone.py`) per utterance, with protocol-phrase neutralization, short-turn merging, and short-utterance context lead-in (see the module docstring for why — it's tuned against a labeled F1 radio corpus). Rolls up a duration-weighted clip-level sentiment/mood. Skipped (warns, doesn't fail) if the sentiment model isn't installed. | — |
| 10 | `INDEXING` | `stages/index.py` | Sentence-embeds each utterance's text (`ctx.pool.text_embedder`) for semantic search. No-op if there are no utterances. | — |

`ctx.warnings` (non-fatal issues like `POOR_AUDIO_QUALITY`, `HIGH_OVERLAP`,
`HALLUCINATION_FILTERED`, `LOW_LANGUAGE_CONFIDENCE`, `LOW_RELIABILITY_EMBEDDING`,
`SPEAKER_COUNT_AT_CEILING`, `SPEAKER_TOO_SHORT`, `HIGH_SMOOTH_RATE`,
`SENTIMENT_MODEL_MISSING`) accumulate across stages and are stored on `processing_runs`
regardless of outcome.

---

## 2. Live chunk (mic streaming)

`POST /v1/live/{session_id}/chunk?seq=N` (`api/app/routers/live.py`) writes the uploaded
`.webm` chunk straight to `DATA_DIR/tmp/live/{session_id}/{seq}-{rand}.webm` and enqueues
`transcribe_live_chunk_job`. No `clips` row, no dedupe, no size limit check — this is a
few seconds of mic audio, not a file upload.

`transcribe_live_chunk_job()` (`worker/worker/main.py`), under the same `live_sem`:

- Decodes + the *cheap* half of preprocessing only (DC removal, high-pass, loudness
  normalize) — denoise is skipped, it alone costs ~260ms per second of audio in the batch
  run and would eat too much of a live chunk's latency budget.
- No diarization, no speaker ID — a few-second chunk is never enough audio for a stable
  per-chunk speaker read; upload the full recording afterward for that.
- Language: pinned to `asr_language` unless set to `auto`, in which case it locks onto
  whatever the first chunk (or every 6th chunk after, `LIVE_LANG_RECHECK_EVERY`) detects,
  rather than re-guessing from scratch on every few seconds of audio.
- Same hallucination heuristics as the batch transcribe stage, applied inline.
- Tone/mood only runs if ASR actually produced text — acoustic-only features on silence
  were coming back "stressed" on background noise/breathing alone.
- The chunk file is deleted after processing either way (`finally`), and the result is
  emitted over `job:{session_id}` — nothing is persisted to Postgres for live chunks.

---

## 3. F1 team radio

`POST /v1/f1/ingest` (`api/app/routers/f1.py`) is closer to the live path than the batch
one — same ASR-plus-tone shape, no diarization — but with persistence, because a radio
call is a durable, session-scoped record rather than a throwaway mic chunk:

1. Validates `recording_url` starts with the official
   `https://livetiming.formula1.com/` host.
2. Upserts a `radio_calls` row keyed by `recording_url` (idempotent — a session's radio
   feed is fetched by `GET /v1/f1/team_radio` well before anyone asks to analyze a
   specific call).
3. If already analyzed and not `force`: returns the cached `text`/`mood`/`features`
   immediately — no re-download, no re-transcription.
4. Otherwise downloads the MP3 to `DATA_DIR/tmp/f1/{call_id}.mp3` and
   `queue.enqueue_f1_radio(call_id, path, session_key, driver_number)`.

`analyze_f1_radio_job()` (`worker/worker/main.py`): same decode → cheap-preprocess →
ASR → hallucination-drop → tone shape as the live job, except the tone baseline is keyed
by `f"{session_key}:{driver_number}"` instead of a session id — a driver's calls across a
race calibrate against each other. Persists the result onto `radio_calls`
(`text`, `mood`, `features`, `error`, `analyzed_at`) *before* emitting over the websocket,
so a client that reconnects late can still `GET /v1/f1/analyses` and see it. The
downloaded MP3 is deleted afterward — the F1 page replays audio straight from
`livetiming.formula1.com`, so nothing needs a local copy today.

---

## Graph projection

**Graph projection** (`GRAPH_RAG_PLAN.md`) is a derived read model — Neo4j is wiped and
re-projected from Postgres, never written to directly — but it *is* wired into ingestion,
not purely on-demand: `worker/worker/pipeline.py:_sync_graph()` runs
`common.graph_sync.rebuild()` automatically after a clip reaches `COMPLETE` and after a
non-errored F1 radio analysis persists (`analyze_f1_radio_job` in
`worker/worker/main.py`). It's a no-op when `GRAPH_ENABLED=false`, and best-effort when
enabled — a `GraphUnavailable`/connection failure is logged and swallowed rather than
failing the clip, since Neo4j is an optional datastore. Live chunks don't trigger it:
they persist nothing to Postgres, so there's nothing new for the graph to project.

`POST /v1/admin/graph/sync` (`api/app/routers/admin.py`) still exists for a manual
backfill/rebuild — e.g. after editing `f1_aliases`, or recovering from a stretch where
Neo4j was down and the auto-sync kept silently no-op'ing.

## What's not part of ingestion

- **Speaker enrollment corrections** (`POST /v1/clips/{id}/speakers/{label}/assign`) can
  add an enrollment to a profile, but that's a user action after the fact, not part of
  the pipeline itself.
