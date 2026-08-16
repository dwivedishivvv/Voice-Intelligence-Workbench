# Speaker Intelligence Workbench

An on-premises voice-intelligence workbench. Drop in an audio or video clip (≤90s) and it
runs a full offline pipeline — preprocess → transcribe → diarize → reconcile → embed →
identify speakers → index — then lets you review, correct, search, and export the result.
Models live on disk under `MODEL_DIR` and run with `HF_HUB_OFFLINE=1`:
no audio and no text ever leaves the box.

Two optional features are the exceptions, both off by default: **Race Radio** fetches F1
session data from OpenF1, and the **agent layer** (`LLM_ENABLED`) sends transcript excerpts
to a hosted model. With both disabled — the default — the core pipeline makes no
outbound calls at all.

Two ways in: **batch upload** and **live microphone transcription**. Race Radio is an
API-only ingestion path (`/v1/f1`) with no screen of its own.

The interface is built on the **Industry** design system: square hairline frames, Barlow
Condensed headings, monospace for every number, and a four-state vocabulary for
identification — named, suggested, unknown, and *declined to judge*, the last drawn hatched
so a refusal never reads as a result.

![Clip Library](presentation/screenshots/library.png)

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
- [Race Radio (F1)](#race-radio-f1)
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
| Sentiment | multilingual XLM-R over the transcript, fused with the acoustic tone read; stored per utterance and rolled up per speaker |
| Races | group recordings by race, bulk upload, SVG track outline, per-voice filtering and analysis |
| Export | SRT, VTT, RTTM, JSON, TXT |
| Live | mic capture, pause-aligned chunking, transcript + tone read (no per-chunk speaker ID) |
| Race Radio | OpenF1 session/driver/lap data, radio ingestion, heuristic tone classification (API only, no UI page) |
| Ops | structured logs with correlation IDs, Prometheus metrics, hash-chained audit log, deletion receipts |
| Graph | optional Neo4j projection of clips, speakers, drivers, sessions and laps — a derived read model rebuilt from Postgres |
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
uv pip install -r api/pyproject.toml -r worker/pyproject.toml
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
# db/init.sql only runs on a *fresh* postgres volume, so an existing database needs the
# new tables replayed. f1.sql is written to be idempotent precisely for this:
docker compose exec -T postgres psql -v ON_ERROR_STOP=1 -U $env:POSTGRES_USER -d $env:POSTGRES_DB < db/f1.sql
python -m common.graph_sync --full  # project Postgres into Neo4j, prints per-label counts
```

The sync prints what it *sent* next to what actually *landed*. Those two agreeing is the
check that matters: relationship projections MATCH their endpoints rather than creating
them, so a dropped endpoint shows up as a gap between the two columns rather than as an
error.

Then browse it at `http://localhost:7474`. `POST /v1/admin/graph/sync` does the same thing
over HTTP and writes an `audit_log` row.

What gets projected: speakers, clips, utterances, races, and the F1 side (sessions,
drivers, teams, circuits, laps, radio calls). Utterances and radio calls share a `:Speech`
label — they are the same kind of thing, a unit of transcribed speech with a tone read,
and only differ in how much of the pipeline has touched them.

Two edges do the analytical work. `DURING_LAP` ties a moment of speech to the lap it
happened on, which is what makes "was the tone stressed on the lap that lost time" a single
hop; it needs a wall-clock timestamp, so radio calls (which carry one from OpenF1) get it
and manually uploaded clips without `recorded_at` do not. `MENTIONS` links speech to the
drivers, teams and circuits named in it, via a dictionary matcher over `f1_aliases` rather
than an NER model — the vocabulary is closed, small, and already in the database.

`POST /v1/graph/context` is what reads it back: the query is ranked in Postgres (the same
hybrid search the search page uses, widened to cover radio calls), each hit's neighbourhood
is pulled from Neo4j in one round trip, and the result is rendered as short lines rather
than returned as raw JSON:

```
[abc-123] Monaco Grand Prix 2024 | VER (Red Bull Racing)
  tone: voice reads stressed, words read negative (-0.62)
  "I can't get past, the tyres are gone"
  lap 42: 74.8s (+0.9s vs previous)
  mentions: LEC (driver)
```

Tone is always phrased as a reading, never as a fact — the classifier behind it is a
threshold heuristic (`worker/worker/audio/tone.py`), and the block leads with its speech id
so any claim built on it can be traced back to the recording.

See `GRAPH_RAG_PLAN.md` for the full node/edge model and what is deliberately left out.

Dependencies are managed with [uv](https://docs.astral.sh/uv/) — see `api/pyproject.toml`
and `worker/pyproject.toml`. The worker needs Python 3.11 and a CUDA build of torch; the
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
uv venv --python 3.11 .venv;        $env:VIRTUAL_ENV=".venv";        uv pip install -r api/pyproject.toml
uv venv --python 3.11 .venv-worker; $env:VIRTUAL_ENV=".venv-worker"; uv pip install -r worker/pyproject.toml
cd web; npm install; cd ..
```

Two Windows-specific notes. Models land in `C:\models`, not `.runtime/models` —
`models/REGISTRY.yaml` hardcodes absolute `/models/...` paths, which resolve to `C:\models`
on Windows, and the script points `MODEL_DIR` there so downloads and `verify_models()` agree.
And creating symlinks needs admin or Developer Mode, so anything that would symlink the
Hugging Face cache copies instead (`WinError 1314` is the symptom when it doesn't).

## The screens

Eight screens behind one sidebar, which carries live counts for each: Library, Review queue,
Ask, Upload, Live, Races, Speakers, Settings. Light and dark are both first-class — the
palette is a set of CSS variables swapped by a `data-theme` attribute, toggled at the bottom
of the sidebar and remembered per browser.

> The screenshots below predate the Industry redesign and show the previous interface. The
> behaviour each one describes is current; the styling is not.

### Upload

Drop in a folder of audio or video. Validation runs server-side on ingest (container,
duration, size), then each file is queued and its progress streams back over its own
WebSocket — per-stage, in a batch table, with a weighted bar per file and a segmented bar
across the batch. Three files are uploaded at a time; the API streams each to disk, and
firing everything at once buys no throughput.

Duplicates are recognised by audio rather than filename and jump to the existing result.
A rejection states its reason and is never retried silently — `Why?` opens the worker's
own verdict on that file.

![Upload](presentation/screenshots/upload.png)

### Clip Library

Every processed recording: duration, status, detected language, voice count, upload time,
and a **Why it is here** column stating what the pipeline thought. One segmented control
flips between the whole library and the **Review queue** (`/review`) — clips the pipeline
itself flagged, because identification hesitated or the audio graded poor. Four counters
above the table come from `/v1/admin/stats`, split by identification outcome rather than by
a single accuracy number.

Deleting asks for the filename back, and counts what goes with the clip first — turns,
words, embeddings, and any profile enrollment that came from it.

![Clip Library](presentation/screenshots/library.png)

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

The transcript is speaker-tagged with a tone meter per turn; words below 0.5 confidence
carry a dotted amber underline and their score on hover, and any word seeks on click. The
side rail splits attribution coverage four ways and keeps text sentiment and voice tone
apart, counting where they disagree instead of averaging them.

A correction is training data, not a UI edit: it re-enrolls and recomputes the centroid,
and says so before you save — including that enrollment will be skipped, with a reason, if
the audio can't support it.

`Reprocess` reruns the pipeline with current settings. `Export SRT` and the other formats
(VTT, RTTM, JSON, TXT) come off `/v1/clips/{id}/export/{fmt}`.

![Clip detail](presentation/screenshots/clip-detail.png)

### Live transcription

Speak into the mic. Each chunk cuts on your next natural pause rather than a fixed timer,
so words don't get split mid-sentence. A live session is transcript and tone only — it does
not run identification per chunk, and the screen says so rather than showing a name it
cannot stand behind. The side rail carries the input level, the tone strip across the
session, and which tools ran locally versus off-box (everything here is local).

![Live transcription](presentation/screenshots/live.png)

### Ask

Question the corpus. The sidebar hands itself over to thread history while Ask is open;
threads live in the browser, because the API answers one question at a time and keeps no
conversation of its own.

Every claim comes back with the clip it came from: cited ids become numbered chips in the
prose that jump to their evidence card, and each card carries the quote, the voice and text
readings side by side, and a **Listen at** link that lands the player on that second.
Passages the model read but did not cite are shown separately, hatched, labelled as
carrying no citation authority. A provenance strip closes each answer with the tools that
ran, the model, and the token cost — this is the one component that leaves the box, so what
it did is not hidden behind a spinner.

### Races

A race groups its recordings; filing changes grouping only, and nothing is reprocessed.
The detail view charts traffic across the weekend by tone, lists the recordings, and
surfaces **moments worth hearing** — utterances where the two readings disagree, the tone
reads stressed, or nobody was attributed at all. The side rail splits attribution across
the weekend into named, grouped-but-unnamed, and unattributed, and links straight into the
review queue when clips from that race are waiting on a curator.

### Speaker Directory

Enrolled profiles with enrollment counts and intra-profile cohesion (a low-cohesion warning
means the enrollments disagree with each other), alongside unclaimed voice clusters awaiting
review. Each cluster can be **auditioned** before it is named: the montage endpoint stitches
that voice's own speech from every clip it appears in, and one plays at a time — two voices
overlapping is exactly what makes a cluster unrecognisable.

Merging states its arithmetic before you commit (enrollments, speech, cohesion recomputed
from the combined set) and says plainly that if the two are not the same person, cohesion
drops and matches get worse — the number after the merge tells you which happened.

![Speaker Directory](presentation/screenshots/speakers.png)

### Settings

Every pipeline knob, grouped: Ingest, Pre-processing, Quality grading, Diarization, Speaker
identification, Transcription, Jobs & retention, Graph, Models, System. Changes are stored
as overrides in the database and picked up by the next job — no restart. Each field shows
its type, default and state (`DEFAULT` / `OVERRIDDEN`) and resets individually; the header
counts how many are currently overridden.

**Calibrate from my profiles** runs the EER sweep over your own enrollments and shows what
it would change — `id_threshold` and `verify_threshold`, old value → new, with the measured
error rate and voice separation behind them. Nothing is written until you apply it.

![Settings — tuning](presentation/screenshots/settings-tuning.png)

The **Models** tab is the on-demand model catalog: see what's active, download an
alternative, activate it.

![Settings — models](presentation/screenshots/settings-models.png)

**System** shows device, precision, model versions and the read-only environment-level
settings.

![Settings — system](presentation/screenshots/settings-system.png)

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
| `--sig-green` / `--sig-amber` / `--sig-red` | calm, tired, stressed — and the same three for ok / warn / bad |
| `--font-heading` (Barlow Condensed) / `--font-body` (Barlow) | headings and prose |

Three rules the screens hold to:

- **Numbers are monospace.** Every score, threshold, duration and id sits in `--font-mono`,
  so a column of them lines up and a changed digit is visible.
- **A refusal is not a result.** Anything the system declined to judge is drawn hatched,
  never as a fifth colour in the speaker palette.
- **Consequences precede confirmation.** Every dialog states what saving does, in words, in
  its footer — next to the button that does it.

`web/src/lib/ui.ts` holds the vocabulary in code: the outcome map, the voice palette, tone
colours, and the formatters. Colours are CSS variables all the way down, so light and dark
need no JavaScript.

## The pipeline, stage by stage

`worker/worker/pipeline.py` runs nine stages in order, each in its own file under
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
| 9 | `INDEXING` | 7 | Text embeddings + tsvector for search |

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

All three accept `speaker_id` to scope results to one person.

## Race Radio (F1)

An ingestion path, not a screen — the UI page was removed in the Industry redesign; the
endpoints below are unchanged and still callable.

The one component that talks to the network. `api/app/routers/f1.py` proxies
[OpenF1](https://openf1.org) for sessions, drivers, laps and team-radio metadata, then
`POST /v1/f1/ingest` downloads a radio call and queues it for analysis. Ingestion refuses
any URL that isn't an official `livetiming.formula1.com` asset.

Analysis reuses the normal transcription path, then adds a **heuristic** tone classifier
(`worker/worker/audio/tone.py`): speech rate, pitch mean/variability, energy variance and
voiced ratio, bucketed into calm / stressed / tired. When a `session_key` and
`driver_number` are supplied, tone calibrates against that driver's other calls in the same
session — an EMA baseline over their calm chunks — rather than fixed global thresholds,
because a driver's normal pitch is not everyone's normal pitch.

This is a threshold heuristic over acoustic features, not a trained speech-emotion model.
It shows a signal; treat it as one. The seam for swapping in a real SER model is
`extract_features` + the classifier beneath it.

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
| `GET` | `` | List with `q`, `status`, `needs_review`, offset/limit |
| `GET` | `/{id}` | Clip metadata and status |
| `GET` | `/{id}/result` | Full result: quality, turns, utterances, words, speakers |
| `GET` | `/{id}/audio?variant=original\|processed` | Stream audio |
| `GET` | `/{id}/export/{srt\|vtt\|rttm\|json\|txt}` | Export |
| `POST` | `/{id}/reprocess` | Rerun with current settings |
| `PATCH` | `/{id}` | Edit tags/notes |
| `DELETE` | `/{id}` | Delete, writing a deletion receipt |
| `POST` | `/{id}/speakers/{local_label}/assign` | Correct a speaker — enrolls and recomputes |

**Speakers** — `/v1/speakers`: list/create/get/patch/delete profiles, `/{id}/clips`,
`/{id}/merge`, `DELETE /{id}/enrollments/{eid}`.

**Clusters** — `/v1/clusters`: list, get, `POST /{id}/promote` to turn a cluster into a profile.

**Search** — `POST /v1/search` (`q`, `mode`, `speaker_id`, `limit`).

**Agent** — `POST /v1/agent/ask` (`question`, optional `history`, optional
`conversation_id`). Runs against either Anthropic or any OpenAI-compatible endpoint
(NVIDIA NIM, local vLLM/Ollama) via `LLM_PROVIDER`; the four tools are shared, only the
loop differs: answers a question by querying the corpus with four read-only tools,
returning the answer plus the speech ids it cited, the tools it used and token usage.
Blocking, with progress mirrored to `WS /v1/ws/jobs/{conversation_id}`. 503 when
`LLM_ENABLED=false`.

### The agent's tools

Four read-only tools, declared in `api/app/services/agent.py`. Each is plain async code
that knows nothing about who is calling it, so the schema for both wire formats is derived
from one declaration. Every filter is optional and composes with every other, because
questions arrive in combinations nobody enumerated in advance.

| Tool | Arguments | What it does |
|---|---|---|
| `search_speech` | `query`, `spoken_by`, `mood`, `limit` | Finds speech by topic, by tone, or both, across processed utterances *and* F1 radio calls. Returns one line per hit: id, speaker, session and lap where known, the tone reading, and a clipped quote. |
| `expand_speech` | `speech_id` (one id, or several comma-separated) | Everything the graph knows around a piece of speech: speaker, session, the lap and its time against the previous one, entities named, and the lines either side. Names any id the graph did not have rather than returning fewer rows silently. |
| `driver_timeline` | `session_key`, `driver_number`, `from_lap`, `to_lap` | One driver through one session: lap times with the radio spoken on each lap. The comparison search cannot make, because it ranks speech without knowing what the car was doing. |
| `compare_speakers` | `profile_id_a`, `profile_id_b` (either accepts a comma-separated list; names work as well as ids) | Contrasts enrolled speakers on counts and aggregate readings. For what was actually said, search instead. |

**Topic and tone are different axes.** Tone is a stored label, so no amount of searching
for the word "stressed" finds speech whose voice reads stressed — that is `mood`, which
takes `calm`, `stressed` or `tired` and works alone or alongside a topic. With no topic
the results are ordered by recency, not intensity: the classifier emits a label, not a
magnitude, so there is no "most stressed". A tone word typed into `query` that matches no
text falls back to the filter and says so in the result, because a model that reaches for
the word instead of the filter should not be told the corpus is empty.

Nothing writes. A test asserts that — the tool bodies and every query constant they reach
for are scanned for write keywords.

**Graph** — `POST /v1/graph/context` (`q` or explicit `speech_ids`, `limit`, `mode`,
`speaker_id`, `max_chars`): ranks speech in Postgres, expands each hit's neighbourhood in
Neo4j, and returns both the structured rows and a rendered text block. 503 when
`GRAPH_ENABLED=false`.

**Admin** — `/v1/admin`: `GET /config`, `GET|PATCH /settings`, `DELETE /settings/{key}`,
`POST /calibrate`, `GET /calibration/latest`, `GET /corrections?days=`, `GET /stats`,
`GET /audit?limit=`, `POST /graph/sync`, `GET /graph/summary` (both 503 when
`GRAPH_ENABLED=false`).

**Models** — `/v1/admin/models`: `GET ``, `POST /pull`, `POST /activate`.

**F1** — `/v1/f1`: `GET /sessions`, `/drivers`, `/laps`, `/team_radio`, `POST /ingest`.

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
| Graph (optional) | `GRAPH_ENABLED` (off by default), `GRAPH_URI`, `GRAPH_USER`, `GRAPH_PASSWORD` (named `GRAPH_*`, not `NEO4J_*`: the neo4j container reads any `NEO4J_`-prefixed variable as one of its own config settings and won't start on an unknown one) |

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

f1_sessions ──┬── f1_drivers       OpenF1 session/driver/lap data, cached write-through
              └── f1_laps          on every proxy call rather than re-fetched each time
radio_calls          one row per team-radio recording; unique on recording_url, so a
                     repeat analysis is a database read instead of a re-transcription
f1_aliases           entity-linking vocabulary for graph MENTIONS edges ('auto' rows are
                     re-derived each sync, 'manual' rows are yours and are never touched)
```

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
see its `verify_models` step. The core pipeline makes no outbound calls. Two optional
features do, and both are off unless you turn them on: the OpenF1 proxy and radio download
behind Race Radio, and the agent layer (`LLM_ENABLED=false` by default), which is the only
thing in the system that sends **transcript text** to a third party. `LLM_ENABLED` is
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
- **Tone classification** — heuristic thresholds over acoustic features, not a trained SER
  model. See `worker/worker/audio/tone.py`.
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
  app/routers/   clips, speakers, clusters, search, admin, models, f1, live, ws
worker/
  worker/stages/ one file per pipeline stage
  worker/audio/  decode, enhance, quality, vad, tone
  worker/export/ srt, vtt, rttm, json, txt
  worker/live.py live-mode diarize + identify
  worker/pool.py model pool, loaded once at startup
web/
  src/style.css  the Industry design system: tokens, light/dark palettes, component classes
  src/lib/ui.ts  outcome/voice/tone colour vocabulary + the monospace formatters
  src/pages/     Library (+ Review queue), ClipDetail, Ask, Upload, Live, Races, Speakers, Settings
db/init.sql      full schema (pgvector + FTS)
models/          REGISTRY.yaml + prefetch.py, fetched into MODEL_DIR
tests/unit/      reconciliation, reliability, diarization cleanup, device/model resolution
scripts/dev.ps1  native Windows dev runner
presentation/    deck source + screenshots
```
