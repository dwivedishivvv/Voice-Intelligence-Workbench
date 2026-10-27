# Speaker Intelligence Workbench

An on-premises voice-intelligence workbench. Drop in an audio or video clip (≤90s) and it
runs a full offline pipeline — preprocess → transcribe → diarize → reconcile → embed →
identify speakers → index — then lets you review, correct, search, and export the result.
Models live on disk under `MODEL_DIR` and run with `HF_HUB_OFFLINE=1`:
no audio and no text ever leaves the box.

One optional feature is the exception, off by default: the **agent layer** (`LLM_ENABLED`)
sends transcript excerpts to a hosted model. With it disabled — the default — the core
pipeline makes no outbound calls at all.

Two ways in: **batch upload** and **live microphone transcription**. Everything processed is
projected into a **knowledge graph** of people, voices, recordings, topics and the named
things the transcripts mention, which you can explore on the Graph page or question
through the agent.

The interface is built on the **Industry** design system: square hairline frames, Barlow
Condensed headings, monospace for every number, and a four-state vocabulary for
identification — named, suggested, unknown, and *declined to judge*, the last drawn hatched
so a refusal never reads as a result.


---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Running natively on Windows](#running-natively-on-windows)
- [The screens](#the-screens)
- [The design system](#the-design-system)
- [The pipeline, stage by stage](#the-pipeline-stage-by-stage)
- [Speaker identification model](#speaker-identification-model)
- [Search](#search)
- [Live transcription](#live-transcription)
- [Model catalog](#model-catalog)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Data model](#data-model)
- [Operations](#operations)
- [Testing](#testing)
- [Offline verification](#offline-verification)
- [What's scoped out (and where the seam is)](#whats-scoped-out-and-where-the-seam-is)
- [Layout](#layout)

---

## What it does

| Capability | Detail |
|---|---|
| Transcription | faster-whisper `large-v3-turbo` (CTranslate2), word timestamps + per-word logprobs, auto language detect |
| Diarization | pyannote 3.1, VAD-snapped turns, min-turn/merge-gap cleanup, overlap warning |
| Reconciliation | words assigned to turns by overlap, low-confidence smoothing, regrouped into utterances |
| Speaker ID | ECAPA-TDNN embeddings + pgvector, reliability-gated with margin and abstention |
| Enrollment | manual enroll, centroid recomputation, outlier detection, merge/reassign profiles |
| Clustering | unknown voices cluster across clips; promote a cluster to a real profile in one click |
| Calibration | EER sweep over enrolled data to sanity-check the identification threshold |
| Search | Postgres FTS, semantic (MiniLM + pgvector HNSW), or hybrid via reciprocal rank fusion |
| Speakers | create, rename, tag, annotate, merge several at once, delete or fold into another; duplicate-voice suggestions; per-person recordings, lines, connections and samples |
| Library | bulk reprocess / tag / review-flag / delete, rename, notes, and per-line transcript correction that re-indexes search |
| Export | SRT, VTT, RTTM, JSON, TXT |
| Live | mic capture, pause-aligned chunking, a transcript that builds as you speak (no per-chunk speaker ID) |
| Ops | structured logs with correlation IDs, Prometheus metrics, hash-chained audit log, deletion receipts |
| Graph | Neo4j projection of speakers, unnamed voices, clips, utterances, topics, tags, languages and named entities, with an interactive map — a derived read model rebuilt from Postgres |
| Tuning | every pipeline knob editable live from the UI — applies to the next job, no restart |
| Models | download and activate alternate ASR/diarization/embedding models from Settings |
| Agent | optional LLM that answers questions by querying the corpus with read-only tools, and cites the speech ids behind every claim (off by default) |

## Architecture

```
        ┌──────────┐        ┌────────────┐        ┌──────────┐
 audio  │   web    │  HTTP  │    api     │  arq   │  worker  │
 ─────► │ React/TS │ ─────► │  FastAPI   │ ─────► │ pipeline │
        └──────────┘        └────────────┘ (redis)└──────────┘
              ▲                    │                    │
              └── WebSocket ───────┘                    │
                 job progress                           ▼
                                            ┌────────────────────┐
                                            │ Postgres + pgvector│
                                            │ local disk (DATA_DIR)
                                            └────────────────────┘
```

- **api** (FastAPI) — ingest, listing, correction, search, admin, model catalog, WS progress.
  Owns no ML; it enqueues and reads.
- **worker** (arq) — every model lives here, loaded once into a `pool` at startup and shared
  across jobs. One file per pipeline stage under `worker/worker/stages/`.
- **postgres** — pgvector for speaker centroids and utterance embeddings (HNSW indexes),
  plus Postgres full-text search over the same utterances.
- **redis** — arq job queue and the pub/sub channel behind live progress events.
- **web** — React + Vite + TypeScript on the Industry design system: one hand-written
  stylesheet of tokens and component classes (`web/src/style.css`), no CSS framework and no
  component library. Four runtime dependencies in total.

## Quick start

Docker runs the datastores only. The api, worker and web servers run natively — they
reload on edit, and it's the only way the worker can reach a local GPU.

```powershell
cp .env.example .env                  # edit POSTGRES_PASSWORD / API_KEY
echo "<your-hf-token>" > .hf_token    # once — to accept pyannote's gated-model terms
uv pip install -r pyproject.toml --extra api --extra worker
cd web; npm install; cd ..
.\start.ps1                           # postgres + redis in docker, then api / worker / web
.\start.ps1 -Stop -Down               # stop everything
```

Then: web on `:5174`, API on `:8000`, Postgres on `:5432`, Redis on `:6379`.

`start.ps1` prints the detected GPU at launch and warns if the worker's interpreter has a
CPU-only torch — that silent fallback is otherwise invisible and costs you ~20x throughput.

Models are expected on disk under `MODEL_DIR` (default `C:/models`), with the HF cache at
`MODEL_DIR/.cache`. `start.ps1` points `HF_HOME` and `PYANNOTE_CACHE` there, which is what
makes pyannote's nested repo-id references resolve with `HF_HUB_OFFLINE=1`.

The web UI reads its API key from `localStorage`. Once, in the browser console:

```js
localStorage.setItem("api_key", "<API_KEY from .env>")
```

Other targets: `make down` (stop), `make clean` (stop + drop volumes), `make logs`,
`make test`.

### The graph projection (optional)

Off unless `GRAPH_ENABLED=true`. Neo4j holds a *derived* read model — Postgres stays the
source of truth and nothing in the pipeline writes to the graph, so it can be wiped and
rebuilt at any time, and with it disabled every other page behaves exactly as before.

```powershell
docker compose up -d neo4j
python -m common.graph_sync --full  # project Postgres into Neo4j, prints per-label counts
```

The api replays every idempotent schema file in `db/` at startup (`common/schema.py`), so an
existing database picks up new columns without a manual `psql` step.

The sync prints what it *sent* next to what actually *landed*. Those two agreeing is the
check that matters: relationship projections MATCH their endpoints rather than creating
them, so a dropped endpoint shows up as a gap between the two columns rather than as an
error.

Explore it on the **Graph** page, browse it raw at `http://localhost:7474`, or rebuild it
with `POST /v1/admin/graph/sync` (which also writes an `audit_log` row). The worker rebuilds
it on its own after every clip completes and after every identity correction, debounced.

What gets projected (`common/graph_sync.py` is the single source):

| Node | From |
|---|---|
| `Speaker` | enrolled profiles, with status, notes and tags |
| `Voice` | recurring unnamed voices (unpromoted clusters still heard somewhere) |
| `Clip` | processed recordings, with language, length and audio grade |
| `Utterance:Speech` | each diarized turn (switch off with `graph_utterance_nodes` on a large corpus) |
| `Topic` | automatic topic tags |
| `Tag` | your own clip and speaker tags |
| `Language` | detected languages |
| `Entity` | people, companies and places named in the transcripts (`common/entities.py`) |

| Edge | Meaning |
|---|---|
| `APPEARS_IN` / `HEARD_IN` | a speaker / unnamed voice in a clip, with speech seconds and talk share |
| `SPOKEN_BY`, `IN_CLIP`, `NEXT` | who said a line, where, and what came next |
| `TALKED_WITH` | two speakers share recordings (weighted by how many) |
| `ABOUT`, `DISCUSSED` | a clip's topics, and the topics each speaker's recordings cover |
| `MENTIONS`, `MENTIONED` | lines and clips that name an entity; the people who name it |
| `CO_MENTIONED` | two entities named in the same recording |
| `TAGGED`, `IN_LANGUAGE` | tags and language |

Entities come from capitalised spans in the ASR text, not a model, and the extractor is
tuned to abstain: a sentence-initial word only counts if it also appears mid-sentence, and a
name must recur `graph_entity_min_mentions` times (default 2) before it becomes a node.

`POST /v1/graph/context` reads it back for retrieval: the query is ranked in Postgres (the
same hybrid search the search page uses), each hit's neighbourhood is pulled from Neo4j in
one round trip, and the result is rendered as short lines rather than raw JSON:

```
[abc-123] clip "board_call.wav" at 1:15 | Marta
  "Reliance led the round and Dunzo needed their sign-off"
  names: Reliance, Kabir Biswas (person)
  recording topics: quick commerce
  before (Dan): "What happened with the funding?"
```

Every block leads with its speech id, so any claim built on it can be traced back to the
recording. `GET /v1/graph/map` and `GET /v1/graph/node/{id}` feed the Graph page.

See `GRAPH_RAG_PLAN.md` for the original design notes.

Dependencies are managed with [uv](https://docs.astral.sh/uv/) — see `pyproject.toml`'s
`api` and `worker` extras. The worker needs Python 3.11 and a CUDA build of torch; the
repo's `worker/.venv` is the interpreter `start.ps1` launches it with.

## Running natively on Windows

There are two launchers, and they overlap — pick one:

- **`start.ps1`** (repo root) — one command brings up datastores + all three servers in their
  own windows, checks the GPU, and stops cleanly (it kills uvicorn's `--reload` child, which
  otherwise keeps port 8000 and serves stale code). Uses `worker/.venv` for the worker.
- **`scripts/dev.ps1`** — one terminal per component, more explicit, easier to read logs from.
  Expects `.venv` and `.venv-worker` at the repo root.

They want different venv layouts, so whichever you use, create the venvs it expects. Worth
collapsing to one launcher.

`scripts/dev.ps1` runs the Python side on the host and leaves Postgres/Redis in Docker —
faster to iterate on, and the only way the worker can reach a local GPU. One terminal each:

```powershell
.\scripts\dev.ps1 infra     # postgres + redis, waits for healthy
.\scripts\dev.ps1 api       # uvicorn on :8000
.\scripts\dev.ps1 worker    # arq worker
.\scripts\dev.ps1 web       # vite on :5174
```

First run needs two venvs — **Python 3.11 specifically**, since torch 2.4 and numpy 1.26
publish no 3.13 wheels:

```powershell
uv venv --python 3.11 .venv;        $env:VIRTUAL_ENV=".venv";        uv pip install -r pyproject.toml --extra api
uv venv --python 3.11 .venv-worker; $env:VIRTUAL_ENV=".venv-worker"; uv pip install -r pyproject.toml --extra worker
cd web; npm install; cd ..
```

Two Windows-specific notes. Models land in `C:\models`, not `.runtime/models` —
`models/REGISTRY.yaml` hardcodes absolute `/models/...` paths, which resolve to `C:\models`
on Windows, and the script points `MODEL_DIR` there so downloads and `verify_models()` agree.
And creating symlinks needs admin or Developer Mode, so anything that would symlink the
Hugging Face cache copies instead (`WinError 1314` is the symptom when it doesn't).

## The screens

Eight screens behind one sidebar, which carries live counts for each: Library, Review queue,
Ask, Upload, Live, Speakers, Graph, Settings. Light and dark are both first-class — the
palette is a set of CSS variables swapped by a `data-theme` attribute, toggled at the bottom
of the sidebar and remembered per browser.

### Upload

Drop in a folder of audio or video. Validation runs server-side on ingest (container,
duration, size), then each file is queued and its progress streams back over its own
WebSocket — per-stage, in a batch table, with a weighted bar per file and a segmented bar
across the batch. Three files are uploaded at a time; the API streams each to disk, and
firing everything at once buys no throughput.

Duplicates are recognised by audio rather than filename and jump to the existing result.
A rejection states its reason and is never retried silently — `Why?` opens the worker's
own verdict on that file.


### Clip Library

Every processed recording: duration, status, detected language, voice count, upload time,
and a **Why it is here** column stating what the pipeline thought. One segmented control
flips between the whole library and the **Review queue** (`/review`) — clips the pipeline
itself flagged, because identification hesitated or the audio graded poor. Four counters
above the table come from `/v1/admin/stats`, split by identification outcome rather than by
a single accuracy number.

Deleting asks for the filename back, and counts what goes with the clip first — turns,
words, embeddings, and any profile enrollment that came from it.

Tick rows to act on many at once: reprocess, add or remove a tag, flag for or clear from
review, or delete (confirmed in the bar itself). **Edit** renames a clip and changes its
tags, note and review flag.


### Clip detail

Toggle between the original upload and the processed (denoised, loudness-normalized)
audio. The waveform is coloured per speaker and hatched wherever the system declined to
judge; click it to seek.

Each voice gets a card, and which card it gets is the point — the four identification
outcomes say genuinely different things, so they are never flattened into one confidence
number:

| Outcome | What it means | What you can do |
|---|---|---|
| Confident | A name was written to the turn. Shows the gap to the runner-up. | Correct it |
| Suggested | Closest profile, under the bar. Nothing has been written. The match bar marks `id_threshold` where the proposal fell short. | Confirm / Reject |
| Unknown | The audio was good enough to judge, and no profile came close. | Name this voice |
| Abstained | Reliability under the floor — identification was never attempted, so no match score exists. | Label manually |

The threshold marks on those bars are read from live settings, not hard-coded: they are
tunable, and a stale mark would be a lie.

The transcript is speaker-tagged; words below 0.5 confidence carry a dotted amber underline
and their score on hover, and any word seeks on click. **Edit** on a line corrects what the
ASR misheard — full-text and semantic search both pick up the correction, and so does the
graph on its next rebuild. The side rail splits attribution coverage four ways and links to
the recording's neighbourhood in the graph.

A correction is training data, not a UI edit: it re-enrolls and recomputes the centroid,
and says so before you save — including that enrollment will be skipped, with a reason, if
the audio can't support it.

`Reprocess` reruns the pipeline with current settings. `Export SRT` and the other formats
(VTT, RTTM, JSON, TXT) come off `/v1/clips/{id}/export/{fmt}`.


### Live transcription

Speak into the mic. Each chunk cuts on your next natural pause rather than a fixed timer,
so words don't get split mid-sentence. A live session is transcript only — it does not run
identification per chunk, and the screen says so rather than showing a name it cannot stand
behind.


### Ask

Question the corpus. The sidebar hands itself over to thread history while Ask is open;
threads live in the browser, because the API answers one question at a time and keeps no
conversation of its own.

Every claim comes back with the clip it came from: cited ids become numbered chips in the
prose that jump to their evidence card, and each card carries the quote, the recording it
came from, the names it mentions, and a player that starts on that second.
Passages the model read but did not cite are shown separately, hatched, labelled as
carrying no citation authority. A provenance strip closes each answer with the tools that
ran, the model, and the token cost — this is the one component that leaves the box, so what
it did is not hidden behind a spinner.

### Speaker Directory

A searchable, sortable directory of enrolled people. Open one and the detail panel lets you
rename them inline (a name that collides with someone else offers a merge instead), set
their status, edit tags and notes, and browse four tabs: **recordings** (with talk share),
**lines** they spoke, **connections** (who they share recordings with, what they talk about,
what they name), and **samples** — the enrollments behind the voiceprint, each removable,
with its similarity to the centroid. Tick several people to merge them in one go; the page
also lists profile pairs whose voiceprints say they are probably one person. Delete either
unattributes a person's voices or folds them into someone else.

Below the directory are recurring voices nobody has named yet. Each cluster can be **auditioned** before it is named: the montage endpoint stitches
that voice's own speech from every clip it appears in, and one plays at a time — two voices
overlapping is exactly what makes a cluster unrecognisable.

Naming one can create a new person or add the voice to someone already enrolled; a voice
that is noise or a one-off can be dismissed.

Merging states its arithmetic before you commit (samples, recordings, cohesion recomputed
from the combined set) and says plainly that if they are not the same person, cohesion
drops and matches get worse — the number after the merge tells you which happened. Notes and
tags of merged profiles are folded into the kept one, never dropped.

### Graph

An interactive map of the knowledge graph: people, unnamed voices, recordings, topics, your
tags, languages and named entities, each in its own shape and colour from the design
tokens, so both themes follow. Drag to pan, scroll to zoom, drag a node to pin it; filter
by kind, or search for a name. Clicking a node lights its neighbourhood and opens a panel
with its connections (strongest first), a few lines of speech attached to it, and links to
the clip or speaker page — or straight into Ask. Other pages link in with `/graph?focus=<id>`.


### Settings

Every pipeline knob, grouped: Ingest, Pre-processing, Quality grading, Diarization, Speaker
identification, Transcription, Jobs & retention, Graph, Models, System. Changes are stored
as overrides in the database and picked up by the next job — no restart. Each field shows
its type, default and state (`DEFAULT` / `OVERRIDDEN`) and resets individually; the header
counts how many are currently overridden.

**Calibrate from my profiles** runs the EER sweep over your own enrollments and shows what
it would change — `id_threshold` and `verify_threshold`, old value → new, with the measured
error rate and voice separation behind them. Nothing is written until you apply it.


The **Models** tab is the on-demand model catalog: see what's active, download an
alternative, activate it.


**System** shows device, precision, model versions and the read-only environment-level
settings.


## The design system

`web/src/style.css` is the whole of it — tokens, both palettes, and the component classes
(`.btn`, `.input`, `.seg`, `.tag`, `.table`, `.blueprint`, `.hatch`). No CSS framework, no
component library, no build step beyond Vite. Retune the tokens there and every screen
follows.

| Token group | Purpose |
|---|---|
| `--color-bg` / `--color-text` / `--color-divider` | the ground, the ink, the hairline |
| `--color-accent-100…900`, `--color-neutral-100…900` | one shared OKLCH lightness scale, so the same step of any role matches in visual value |
| `--sig-voice-a` / `--sig-voice-b` | named and suggested voices |
| `--sig-green` / `--sig-amber` / `--sig-red` | ok / warn / bad |
| `--font-heading` (Barlow Condensed) / `--font-body` (Barlow) | headings and prose |

Three rules the screens hold to:

- **Numbers are monospace.** Every score, threshold, duration and id sits in `--font-mono`,
  so a column of them lines up and a changed digit is visible.
- **A refusal is not a result.** Anything the system declined to judge is drawn hatched,
  never as a fifth colour in the speaker palette.
- **Consequences precede confirmation.** Every dialog states what saving does, in words, in
  its footer — next to the button that does it.

`web/src/lib/ui.ts` holds the vocabulary in code: the outcome map, the voice palette, and
the formatters. Colours are CSS variables all the way down, so light and dark
need no JavaScript.

## The pipeline, stage by stage

`worker/worker/pipeline.py` runs ten stages in order, each in its own file under
`worker/worker/stages/`. Every stage emits a start/done event with timings; the progress
bar uses fixed relative weights (shown below) rather than live averages.

| # | Stage | Weight | What happens |
|---|---|---|---|
| 1 | `VALIDATING` | 2 | Container/codec probe, duration and size limits, corrupt-file rejection |
| 2 | `PREPROCESSING` | 15 | Decode to 16kHz mono, highpass, denoise, loudness-normalize to target LUFS, clipping check, VAD, quality grade (good/fair/poor) |
| 3 | `TRANSCRIBING` | 35 | faster-whisper with word timestamps and logprobs, hallucination filtering |
| 4 | `DIARIZING` | 25 | pyannote 3.1, turns snapped to VAD boundaries, short turns dropped, near turns merged |
| 5 | `RECONCILING` | 3 | Words assigned to turns by overlap, low-confidence words smoothed against neighbours, regrouped into utterances |
| 6 | `EMBEDDING` | 5 | ECAPA embedding per speaker, pooled from that speaker's cleanest turns up to `EMBED_TARGET_S` |
| 7 | `IDENTIFYING` | 3 | Match against profile centroids, reliability-gated (see below) |
| 8 | `POSTPROCESSING` | 5 | Talk shares, review flags, per-clip speaker rows |
| 9 | `INDEXING` | 4 | Text embeddings + tsvector for search |
| 10 | `TAGGING` | 1 | Automatic clip tags (language, voices, named speakers, quality caveats, topics) |

A stage may raise `RejectError` to mark a clip `REJECTED` with a code and detail (too short,
no speech, unreadable) — distinct from `FAILED`, which is an unexpected exception. Every run
is recorded in `processing_runs` with its config snapshot, model versions, device, per-stage
timings and warnings, so any result is reproducible after the fact.

Reprocessing deletes derived rows first (`quality_metrics`, `vad_regions`, `speaker_turns`,
`transcripts`, `clip_speakers`) so reruns don't accumulate duplicates.

## Speaker identification model

Identification is deliberately conservative — it would rather abstain than assert.

1. **Reliability** — each embedding gets a 0–1 reliability score from speech duration, SNR
   and quality grade. Below `RELIABILITY_POOR`, identification abstains outright.
2. **Threshold penalty** — a low-reliability embedding faces a *higher* similarity bar
   (`ID_THRESHOLD` + up to `ID_THRESHOLD_PENALTY`), not the same one.
3. **Margin** — the top match must beat the runner-up by at least `ID_MIN_MARGIN`. Two
   similar-sounding profiles produce an abstention, not a coin flip.
4. **Outcome** — one of `confident`, `suggested` (within `ID_SUGGEST_DELTA` of the bar),
   `unknown`, or `abstained`. Only `confident` auto-labels; the rest surface for review.
5. **Unknown voices** — cluster across clips at `CLUSTER_THRESHOLD`. Promote a cluster to a
   named profile and every clip it appears in updates.
6. **Calibration** — `POST /v1/admin/calibrate` sweeps thresholds over enrolled data and
   reports EER, so `ID_THRESHOLD` is a measured choice rather than a guess.

Enrollment maintains a running centroid per profile with outlier detection — an enrollment
that disagrees with its own profile gets flagged rather than silently dragging the centroid.

## Search

`POST /v1/search` with `mode`:

- **`fts`** — Postgres `websearch_to_tsquery` + `ts_rank_cd` over the utterance tsvector.
- **`semantic`** — multilingual MiniLM (384-dim) over pgvector with an HNSW cosine index.
- **`hybrid`** (default) — reciprocal rank fusion: `1/(60+rank_fts) + 1/(60+rank_vec)` over
  the top 200 of each. Catches both exact quotes and paraphrases.

All three accept `speaker_id` and `clip_id` to scope results to one person or recording,
and `tags` to clips carrying every one of them.

## Live transcription

`POST /v1/live/{session_id}/chunk` takes a numbered chunk; the worker runs a condensed
diarize → embed → identify chain on it (`worker/worker/live.py`).

Two thresholds deliberately differ from batch, both because a single short chunk is a worse
sample than a whole clip:

- `LIVE_MATCH_THRESHOLD = 0.60` (batch clustering uses 0.76) — at the batch bar, one
  person's own voice fell below it chunk to chunk and fragmented into a new "Speaker N"
  almost every time.
- `LIVE_EMBED_MIN_S = 0.5` (batch uses 1.5s) — the batch floor exists to keep bad data out
  of *enrollment*; a read-only live *match* can attempt shorter speech, since reliability
  scoring already penalizes it and scales the match threshold accordingly.

Enrolled profiles are matched exactly as in batch. Unmatched voices get an in-memory,
session-scoped running centroid — stable within a session, gone when it ends. A turn too
short to embed is labeled `Speaker (unclear)` rather than being given a fake stable identity.

## Model catalog

`models/REGISTRY.yaml` is the source of truth for what's baked into the image:

| Purpose | Model | Notes |
|---|---|---|
| ASR | `deepdml/faster-whisper-large-v3-turbo-ct2` | CTranslate2, word timestamps + logprobs |
| VAD | `silero-vad` | Weights ship inside the PyPI wheel — no HF repo, no fetch |
| Diarization | `pyannote/speaker-diarization-3.1` | Gated; needs `.hf_token` once at build |
| Embedding | `speechbrain/spkrec-ecapa-voxceleb` | 192-dim speaker vectors |
| Text embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384-dim, ~50 languages |

At runtime, Settings → Models lists alternates per category, downloads one on demand
(`POST /v1/admin/models/pull`) and activates it (`POST /v1/admin/models/activate`), which
persists the choice and reloads the worker pool. Downloads are recorded in
`downloaded_models`.

Two upstream quirks worth knowing, both handled: pyannote's segmentation submodel and
speechbrain's ECAPA sub-checkpoints resolve their dependencies through the standard HF cache
by repo id regardless of a local `source` override, so both must exist in the cache and not
only in the flat `local_dir`.

## API reference

Every route requires `X-API-Key` (`api/app/auth.py:get_current_user`).

**Clips** — `/v1/clips`

| Method | Path | Purpose |
|---|---|---|
| `POST` | `` | Upload a clip (multipart, plus `tags`, `notes`) |
| `GET` | `` | List with `q`, `status`, `needs_review`, repeatable `tag` (all must match, operator or auto), offset/limit |
| `GET` | `/tags` | Every tag in use with its clip count, `manual` and `auto` reported apart |
| `GET` | `/{id}` | Clip metadata and status |
| `POST` | `/{id}/retag` | Re-derive auto tags from the stored analysis (no reprocess) |
| `GET` | `/{id}/result` | Full result: quality, turns, utterances, words, speakers |
| `GET` | `/{id}/audio?variant=original\|processed` | Stream audio |
| `GET` | `/{id}/export/{srt\|vtt\|rttm\|json\|txt}` | Export |
| `POST` | `/{id}/reprocess` | Rerun with current settings |
| `PATCH` | `/{id}` | Rename, edit tags/notes/review flag |
| `POST` | `/bulk` | `{ids, action, tag}` — `delete`, `reprocess`, `add_tag`, `remove_tag`, `mark_reviewed`, `flag_review` |
| `PATCH` | `/{id}/utterances/{uid}` | Correct one transcript line; re-embeds it for search |
| `DELETE` | `/{id}` | Delete, writing a deletion receipt |
| `POST` | `/{id}/speakers/{local_label}/assign` | Correct a speaker — enrolls and recomputes |

**Speakers** — `/v1/speakers`: list (with recordings, last heard), create (409 with the
existing id on a duplicate name), get, `PATCH` (name, status, notes, tags), `DELETE`
(optionally `?reassign_to=`), `/{id}/clips`, `/{id}/utterances`, `/{id}/relations`,
`POST /{id}/merge` (`source_ids` — several at once), `GET /merge-suggestions`,
`/{id}/candidates`, `POST /{id}/claim`, `DELETE /{id}/enrollments/{eid}`.

**Clusters** — `/v1/clusters`: list, get, montage, `POST /{id}/promote` (a new name, or
`profile_id` to add the voice to someone enrolled), `DELETE /{id}` to dismiss.

**Search** — `POST /v1/search` (`q`, `mode`, `speaker_id`, `clip_id`, `tags`, `limit`). Each
hit carries its clip's `filename`.

**Auto tags.** The TAGGING stage writes `clips.auto_tags`, kept apart from the operator's
`clips.tags`: `lang:`, `voices:solo|dialogue|group`, `speaker:<name>` (confident matches
only), `audio:clean|poor`, `length:short|long`, `crosstalk`, `asr:low-confidence`, and up to three `topic:` keyphrases (English; KeyBERT-style over the
search embeddings). Speaker corrections refresh the `speaker:` tags immediately.

**Agent** — `POST /v1/agent/ask` (`question`, optional `history`, optional
`conversation_id`). Runs against either Anthropic or any OpenAI-compatible endpoint
(Groq, NVIDIA NIM, local vLLM/Ollama) via `LLM_PROVIDER`; the tools are shared, only the
loop differs: answers a question by querying the corpus with five read-only tools,
returning the answer plus the speech ids it cited, the tools it used and token usage.
Blocking, with progress mirrored to `WS /v1/ws/jobs/{conversation_id}`. 503 when
`LLM_ENABLED=false`.

### The agent's tools

Five read-only tools, declared in `api/app/services/agent.py`. Each is plain async code
that knows nothing about who is calling it, so the schema for both wire formats is derived
from one declaration.

| Tool | Arguments | What it does |
|---|---|---|
| `search_speech` | `query`, `spoken_by`, `limit` | Finds speech by topic, optionally from one speaker. One line per hit: id, speaker, recording and offset, and a quote windowed on the matched words. Says so when no hit actually contains the query's words. |
| `expand_speech` | `speech_id` (one id, or several comma-separated) | Everything the graph knows around a piece of speech: speaker, recording, names it mentions, the recording's topics, and the lines either side with who said them. |
| `read_recording` | `recording`, `from_minute` | One recording's transcript in order, paged, for "summarise X" questions search cannot answer. |
| `related` | `name` | Looks a person, recording, topic or named entity up in the knowledge graph and lists its connections with counts, plus citable lines. |
| `compare_speakers` | `profile_id_a`, `profile_id_b` (comma-separated lists, names, or `all`) | Contrasts enrolled speakers on how much they talk and where. |

Some OpenAI-compatible endpoints return a model's tool call as text instead of
`tool_calls` (`search_speech(query="x")`, Hermes `<tool_call>` JSON, Llama
`<function=…>`, Gemma `call:name{…}`); the loop recognises those and runs them rather than
showing the call to the user as an answer.

Nothing writes. A test asserts that — the tool bodies and every query constant they reach
for are scanned for write keywords.

**Graph** — `POST /v1/graph/context` (`q` or explicit `speech_ids`, `limit`, `mode`,
`speaker_id`, `max_chars`): ranks speech in Postgres, expands each hit's neighbourhood in
Neo4j, and returns both the structured rows and a rendered text block. 503 when
`GRAPH_ENABLED=false`.

**Admin** — `/v1/admin`: `GET /config`, `GET|PATCH /settings`, `DELETE /settings/{key}`,
`POST /calibrate`, `GET /calibration/latest`, `GET /corrections?days=`, `GET /stats`,
`GET /audit?limit=`, `POST /graph/sync`, `GET /graph/summary` (both 503 when
`GRAPH_ENABLED=false`), `POST /retag` (every clip), `GET /maintenance` (what the
background pass has done, from the audit log), `POST /maintenance/run`.

**Background maintenance** (`worker/worker/maintenance.py`, every 5 minutes while
`auto_maintenance` is on): marks clips a lost worker left mid-pipeline `FAILED/STALLED`,
re-queues `QUEUED` clips whose arq job vanished, enforces `retention_days`, tags untagged
clips, and — only with `auto_consolidate_clusters` on — merges duplicate voice clusters.
Every change it makes is audited as `maintenance.*`.

**Models** — `/v1/admin/models`: `GET ``, `POST /pull`, `POST /activate`.

**Graph** — `GET /v1/graph/map?limit=`, `GET /v1/graph/node/{id}` (neighbours + lines) for
the Graph page.

**Live** — `POST /v1/live/{session_id}/chunk?seq=`.

**Progress** — `WS /v1/ws/jobs/{job_id}`.

**Ops** — `GET /healthz`, `GET /readyz`, `GET /metrics`.

## Configuration

`.env` drives everything (see `.env.example` for the annotated full list). Anything
pipeline-shaped is also editable live in Settings and stored as a database override, which
wins over the environment for the next job.

| Group | Keys |
|---|---|
| Ingest | `MAX_UPLOAD_MB`, `MAX_DURATION_S`, `TARGET_DURATION_S`, `MIN_DURATION_S` |
| Compute | `DEVICE` (auto/cpu/cuda), `PRECISION`, `WORKER_CONCURRENCY`, `JOB_TIMEOUT_S`, `JOB_MAX_ATTEMPTS` |
| Models | `MODEL_DIR`, `ASR_MODEL`, `ASR_BEAM_SIZE`, `ASR_LANGUAGE`, `VAD_MODEL`, `DIAR_MODEL`, `EMBED_MODEL`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` |
| Pre-processing | `HIGHPASS_HZ`, `TARGET_LUFS`, `DENOISE_ENABLED`, `DENOISE_PROP_DECREASE`, `CLIPPING_THRESHOLD`, `VAD_*`, `MIN_TOTAL_SPEECH_S` |
| Quality | `QUALITY_GOOD_SNR_DB`, `QUALITY_FAIR_SNR_DB`, `QUALITY_MAX_CLIPPING`, `QUALITY_MIN_BANDWIDTH_HZ` |
| Diarization | `DIAR_MIN_SPEAKERS`, `DIAR_MAX_SPEAKERS`, `MIN_TURN_S`, `MERGE_GAP_S`, `VAD_SNAP_TOL_S`, `OVERLAP_WARN_RATIO` |
| Identification | `RELIABILITY_*`, `ID_THRESHOLD`, `ID_SUGGEST_DELTA`, `ID_MIN_MARGIN`, `ID_THRESHOLD_PENALTY`, `VERIFY_THRESHOLD`, `CLUSTER_THRESHOLD`, `AUTO_ENROLL*` |
| Privacy/ops | `LOG_TRANSCRIPTS` (off by default), `RETENTION_DAYS` (0 = keep forever), `API_KEY` |
| Agent (optional) | `LLM_ENABLED` (**off by default** — the only setting that sends text off the box), `LLM_PROVIDER` (`anthropic` \| `nvidia`), `ANTHROPIC_API_KEY`, `NVIDIA_API_KEY`, `NVIDIA_BASE_URL`, `LLM_MODEL`, `LLM_MAX_ITERATIONS`, `LLM_MAX_TOKENS`, `LLM_EFFORT` |
| Graph | `GRAPH_ENABLED` (on by default), `graph_entity_min_mentions`, `graph_utterance_nodes` (both live-tunable), `GRAPH_URI`, `GRAPH_USER`, `GRAPH_PASSWORD` (named `GRAPH_*`, not `NEO4J_*`: the neo4j container reads any `NEO4J_`-prefixed variable as one of its own config settings and won't start on an unknown one) |

## Data model

`db/init.sql` is the whole schema — pgvector and full-text search included.

```
clips ──┬── quality_metrics        SNR, clipping, bandwidth, grade
        ├── vad_regions            speech regions
        ├── speaker_turns          diarization output
        ├── transcripts ── utterances ── words
        ├── clip_speakers          per-clip speaker + embedding + id outcome
        └── processing_runs        config snapshot, model versions, timings, outcome

speaker_profiles ── speaker_enrollments      centroid (192-dim) + member embeddings
speaker_clusters                             unclaimed voices, promotable to a profile

audit_log            hash-chained, append-only
calibration_runs     EER sweeps
deletion_receipts    proof of deletion after the data is gone
settings_overrides   live tuning
downloaded_models    model catalog state
```

`db/retire_f1_sentiment.sql` drops the tables and columns of the removed F1 (OpenF1
sessions, laps, team radio, races, on-track events) and sentiment/tone features from
databases that predate their removal; on a fresh volume it does nothing.

HNSW cosine indexes on `utterances.embedding`, `speaker_profiles.centroid`,
`speaker_clusters.centroid` and `clip_speakers.embedding`; a GIN index on
`utterances.tsv`.

## Operations

- **Logs** — structured JSON with a per-run correlation ID threaded through every stage.
  Transcript text is never logged unless `LOG_TRANSCRIPTS=true`.
- **Metrics** — `GET /metrics`: `clip_stage_seconds` (histogram, by stage),
  `clip_processing_total` (by outcome), `speaker_identification_total` (by result).
- **Health** — `/healthz` is liveness; `/readyz` runs a Postgres query.
- **Audit** — every mutating action appends a row to `audit_log` carrying `prev_hash` and
  `row_hash`, so tampering breaks the chain. Deletions leave a `deletion_receipts` row after
  the data is gone.
- **Retention** — `RETENTION_DAYS` is a configured, surfaced setting; the sweeper that acts
  on it is not implemented (`0` = keep forever is the current behaviour either way).

## Testing

```bash
make test                          # python -m pytest tests/unit -q
```

16 unit tests, covering the logic with the sharpest edges: word/turn reconciliation and
smoothing, reliability scoring and abstention, turn cleanup, device selection and model
resolution.

## Offline verification

```bash
docker compose exec worker env | grep OFFLINE     # HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1
docker compose logs worker | grep models_loaded   # models loaded from local disk only
```

`worker.pool.verify_models` runs at worker startup and raises if any model in `models/REGISTRY.yaml` is missing —
see its `verify_models` step. The core pipeline makes no outbound calls. One optional
feature does, and it is off unless you turn it on: the agent layer (`LLM_ENABLED=false` by
default), the only thing in the system that sends **transcript text** to a third party. `LLM_ENABLED` is
deliberately not editable from the Settings page — it lives in `.env`, where enabling it is
a deliberate, reviewable act rather than a checkbox anyone with the API key can tick.

## What's scoped out (and where the seam is)

Everything below is a deliberate simplification with a known ceiling, not an oversight:

- **Object storage** — local disk under `DATA_DIR`, not MinIO/S3. `common/storage.py` is the
  seam: swap its four functions for an S3 client to go multi-node.
- **Migrations** — one `db/init.sql` run by Postgres's own `docker-entrypoint-initdb.d`, not
  Alembic. Fine until the schema needs versioned upgrades against live data.
- **Auth** — a single static API key checked in `api/app/auth.py:get_current_user`. That one
  function is the documented seam for real SSO/RBAC; every route depends on it, not on ad-hoc
  checks.
- **Pagination** — offset/limit, not cursor. Correct up to a few thousand clips.
- **Entity extraction** — capitalised spans with abstention rules, not an NER model.
  `common/entities.py` is the seam: swap `extract`/`mentions` for a model and the graph
  projection is unchanged.
- **Phonetic search** (`dmetaphone` for ASR-mangled names) and the **2D embedding plot** —
  not implemented. FTS, semantic and hybrid search all work.
- **GPU OOM recovery** — arq's normal backoff, not a lower-precision `pool.degraded()`
  variant. Add if OOM under load turns out to be common.
- **Multi-tenancy / RBAC beyond admin** — out of scope; no `tenant_id` column exists. Add it
  plus a row-level-security policy if needed.

## Layout

```
common/          shared by api + worker: config, db (asyncpg+pgvector), storage, audit, speaker logic
api/             FastAPI — upload, listing, search, admin, correction, model catalog, WS progress
  app/routers/   clips, speakers, clusters, search, graph, agent, admin, models, live, ws
worker/
  worker/stages/ one file per pipeline stage
  worker/audio/  decode, enhance, quality, vad
  worker/export/ srt, vtt, rttm, json, txt
  worker/live.py live-mode diarize + identify
  worker/pool.py model pool, loaded once at startup
web/
  src/style.css  the Industry design system: tokens, light/dark palettes, component classes
  src/lib/ui.ts  outcome/voice colour vocabulary + the monospace formatters
  src/pages/     Library (+ Review queue), ClipDetail, Ask, Upload, Live, Speakers, Graph, Settings
db/init.sql      full schema (pgvector + FTS)
models/          REGISTRY.yaml + prefetch.py, fetched into MODEL_DIR
tests/unit/      reconciliation, reliability, diarization cleanup, device/model resolution
scripts/dev.ps1  native Windows dev runner
```
