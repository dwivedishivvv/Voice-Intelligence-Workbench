# Technical Reference — Pipeline & Models

Facts pulled from the actual worker/API source, not the marketing copy on the landing page.
Source of truth: `worker/worker/pipeline.py`, `worker/models/REGISTRY.yaml`,
`common/models_catalog.py`, `common/speaker.py`, `db/init.sql`.

## Pipeline stages

`worker/worker/pipeline.py` runs these ten stages in order for every clip. `STAGE_WEIGHTS`
is the fixed (not measured) share of wall-clock time each stage gets in the UI progress bar.

| Stage | Weight | What it does |
|---|---|---|
| VALIDATING | 2% | `stages/validate.py` — format/duration checks before any model runs. |
| PREPROCESSING | 14% | Decode, downmix to 16 kHz mono, normalize, VAD (`silero-vad`). |
| TRANSCRIBING | 33% | faster-whisper ASR, word-level timestamps. |
| DIARIZING | 23% | pyannote turn segmentation. |
| RECONCILING | 3% | Assign words to turns by overlap; merge into utterances. |
| EMBEDDING | 5% | ECAPA-TDNN speaker embedding per turn/label. |
| IDENTIFYING | 3% | Match embeddings against enrolled profiles; margin-gated. |
| POSTPROCESSING | 5% | `stages/postprocess.py` — cleanup pass over the assembled transcript. |
| SENTIMENT | 7% | Per-utterance text + acoustic sentiment, fused. |
| INDEXING | 5% | Text embeddings for semantic search. |

Each stage's timing is recorded per run in `processing_runs.stage_timings`. A `RejectError`
in any stage marks the clip `REJECTED` with a code/detail; any other exception marks it
`FAILED` and re-raises.

## Models (default set, from `worker/models/REGISTRY.yaml`)

| Category | Model | Notes |
|---|---|---|
| ASR | `deepdml/faster-whisper-large-v3-turbo-ct2` (CTranslate2) | Word timestamps + per-word logprob. |
| VAD | `silero-vad` | Bundled in the PyPI wheel — no HF repo, no network call. |
| Diarization | `pyannote/speaker-diarization-3.1` (gated) | Depends on `pyannote/segmentation-3.0` and `pyannote/wespeaker-voxceleb-resnet34-LM`. |
| Speaker embedding | `speechbrain/spkrec-ecapa-voxceleb` | 192-dim. |
| Text embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384-dim, ~50 languages, used for semantic search. |
| Sentiment | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | 3-class (negative/neutral/positive), multilingual, ~1.1GB. |

All are swappable per-category from Settings → Models (`common/models_catalog.py` lists the
alternatives, e.g. whisper `tiny`→`large-v3`, pyannote 3.0 vs 3.1, X-Vector vs ECAPA, two
other sentiment checkpoints). Swapping a model updates the relevant `*_model` Settings field;
`ModelPool` only reads it at worker startup, so activation requires a restart by design — no
hot-swap of a loaded torch model.

### Sentiment fusion (`worker/worker/stages/sentiment.py`)

Two signals stored separately, then fused — not averaged:

- **Text** — the XLM-R classifier above. Sees content ("we have a problem" is negative
  regardless of delivery).
- **Acoustic** — `worker/worker/audio/tone.py`. **Not a trained model** — a heuristic
  threshold classifier over pitch (F0) mean/variability and speech rate via `librosa.pyin`,
  calibrated against a clip-scoped baseline. Below `MIN_DURATION_S` (1.5s) it returns
  neutral features rather than guessing off too little audio.
- Fusion nudges the text score by a small, tunable `MOOD_SHIFT` rather than blending the two
  into one number, so a flat delivery of bad news (or a cheerful-sounding complaint) shows up
  as a disagreement between the two columns instead of being averaged away.
- Three corrections sit in front of the classifier because short radio-traffic utterances
  break a sentence model badly (measured on 223 F1 utterances: a quarter were ≤4 words, half
  scored exactly 0.00): a protocol-word lexicon, short-turn merging for same-speaker turns,
  and a damped context window for turns too short to judge alone.

### Speaker identification (`common/speaker.py`)

- Match score is cosine similarity against enrolled profile centroids.
- `margin` = top match's similarity − runner-up's similarity.
- Effective threshold = `cfg.id_threshold + cfg.id_threshold_penalty * (1 - reliability)` —
  a less reliable embedding needs a higher bar.
- Result is `"confident"` only if similarity ≥ threshold **and** margin ≥
  `cfg.id_min_margin`; margin alone (not a fixed cosine cutoff) is what separates "confident"
  from "suggested", because it looks at the shape of the whole score distribution, not just
  the top score.
- Below `cfg.reliability_fair`, a would-be "confident" result is downgraded to "suggested" —
  a low-reliability embedding is never allowed to produce the strongest label.
- No embedding at all (silence, embed failure) → `"abstained"` before matching is even
  attempted.
- Auto-enrollment (writing a new sample back into a profile) additionally requires the
  embedding's own two halves to agree (`SPLIT_HALF_SUSPECT` gate in `stages/embed.py`) — a
  blended two-speaker embedding is barred from enrolling even if the match score alone would
  pass, since a bad auto-enroll poisons every future match against that profile.

## Search & storage

- **Full-text**: Postgres `tsvector`/`GIN`, `simple` config, via `websearch_to_tsquery` and
  `ts_rank_cd`.
- **Vector**: `pgvector` `hnsw` indexes with cosine ops on `utterances.embedding`,
  `speaker_profiles.centroid`, `speaker_clusters.centroid`, and `clip_speakers.embedding`
  (`db/init.sql`).
- **Hybrid ranking**: reciprocal rank fusion (RRF, k=60) combining the FTS rank and vector
  rank — chosen because `ts_rank_cd` and vector similarity aren't on comparable scales, so
  fusing by rank rather than raw score avoids one list dominating for no principled reason
  (`api/app/services/search.py`).

## Graph projection (`common/graph_sync.py`)

An optional Neo4j read model, on by default (`cfg.graph_enabled = True`) but strictly
derived — rebuilt wholesale from Postgres on every sync (`graph_sync.rebuild()`), never
written to directly, and a failed/unavailable Neo4j never fails a clip (`pipeline.py`'s
`_sync_graph` swallows `GraphUnavailable`). With it off, search, the pipeline, and every
other page behave identically.

Core node labels: `Speaker`, `Clip`, `Utterance`, `Driver`, `Team`, `Session`, `Circuit`,
`Lap`, `Race`, `RadioCall`/`Speech`, `EventType`. Key edges: `Utterance-[:IN_CLIP]->Clip`,
`Utterance-[:SPOKEN_BY]->Speaker`, `Driver-[:VOICE_OF]->Speaker`,
`Driver-[:DRIVES_FOR]->Team`, `Driver-[:DROVE_IN]->Session`, `Lap-[:BY]->Driver`,
`Lap-[:IN]->Session`, `Speech-[:MENTIONS]->{Driver,Team,Circuit}` (entity-linked via the
`f1_aliases` dictionary), `Speech-[:READS_AS]->EventType`.

**Sentiment reaches the graph as node properties, not a separate node type**: `Clip` nodes
carry `sentiment`, `sentiment_score`, `mood`; `Utterance` nodes carry `sentiment`,
`sentiment_score`, `text_sentiment`, `mood` (mirrored straight off the `clips`/`utterances`
columns the SENTIMENT stage writes). This means a graph query can combine an entity
traversal with an emotional filter in one hop — e.g. every stressed utterance that
`MENTIONS` a given driver this session — without a join across the relational schema.

## Offline posture

`HF_HUB_OFFLINE=1` by default; models are loaded from `MODEL_DIR` on disk and the core
pipeline makes no outbound network call. The only two features that can reach the network —
Race Radio (OpenF1) and the agent/LLM layer — are opt-in and off by default.
