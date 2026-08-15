# Speaker Intelligence Workbench

**Architecture & System Design Overview**
*On-premises speech intelligence platform — transcription, speaker identity, sentiment, search*

---

## 1. What the System Does

The Workbench turns raw audio recordings into structured, searchable intelligence. An operator uploads a recording; the system returns a time-aligned transcript, a breakdown of who spoke when, named identification of known speakers, a sentiment and tone read of every utterance, and full search across the resulting library.

Every model runs locally. There are no external inference calls, no API keys to a cloud provider, and no audio leaves the deployment. This is a hard design constraint, not a configuration option: the runtime starts in a fully offline mode and refuses to boot if any required model is missing from local storage.

### Core capabilities

- Speech-to-text with word-level timing and confidence.
- Speaker diarization — segmentation of the audio into speaker turns.
- Voice fingerprinting and identification against an enrolled speaker roster.
- Automatic clustering of recurring unknown voices, promotable to named profiles.
- Dual-signal sentiment: what was said (text) fused with how it was said (acoustic tone).
- Hybrid search — keyword, semantic, and a fused ranking of both.
- Live streaming mode for near-real-time transcription of an in-progress session.
- Grouping of recordings into events (e.g. a race weekend) with shared speaker identity across them.
- Export to subtitle, annotation, and data-interchange formats.
- Tamper-evident audit trail over every operator action.

---

## 2. Architecture at a Glance

Four cooperating tiers, deliberately kept small. Nothing is split into a service unless it has a genuinely different scaling or hardware profile.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  BROWSER CLIENT (single-page app)                                        │
│  Library · Upload · Recording Detail · Speakers · Events · Live · Config  │
└───────────────┬──────────────────────────────────────┬───────────────────┘
                │ REST (request/response)              │ WebSocket (progress)
┌───────────────▼──────────────────────────────────────▼───────────────────┐
│  API TIER  (stateless, horizontally scalable, no ML models loaded)       │
│  auth · upload & dedupe · query & search · admin · exports · live intake │
└───────┬───────────────────────────┬──────────────────────────┬───────────┘
        │ enqueue job               │ read/write               │ publish
┌───────▼─────────┐   ┌─────────────▼──────────────┐   ┌───────▼──────────┐
│  JOB QUEUE      │   │  RELATIONAL STORE          │   │  EVENT CHANNEL   │
│  (in-memory     │   │  transcripts · speakers ·  │   │  progress fan-out│
│   broker)       │   │  vectors · audit · config  │   │  to connected UIs│
└───────┬─────────┘   └─────────────▲──────────────┘   └──────────────────┘
        │ dequeue                   │ persist stage output
┌───────▼──────────────────────────────────────────────────────────────────┐
│  WORKER TIER  (GPU-attached, models resident in memory)                  │
│  the processing pipeline — one concern per stage, run in fixed order     │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │ read/write media
                     ┌──────────▼───────────┐
                     │  OBJECT / FILE STORE │
                     │  raw + normalized    │
                     │  audio, exports      │
                     └──────────────────────┘
```

| Tier | Responsibility | Why it is separate |
|---|---|---|
| **Client** | Presentation and operator workflow only. Holds no business rules. | Ships independently of the backend; can be replaced without touching logic. |
| **API** | Request validation, authorization, persistence, querying, job submission. | Must stay fast and cheap to scale; loads no ML models so it starts in seconds. |
| **Worker** | The heavy ML pipeline. One recording at a time per worker slot. | Needs a GPU and gigabytes of resident model weights — the opposite profile to the API. |
| **Datastores** | Durable state: relational data, vectors, queue, media files. | Operational concerns (backup, retention) belong to infrastructure, not application code. |

The API and the worker share a common library of primitives — configuration, database access, storage paths, audit writing, and speaker mathematics. Anything both tiers must agree on lives there exactly once, so drift between them is structurally impossible rather than merely discouraged.

---

## 3. Request and Processing Flow

1. The operator uploads a recording. The API streams it to disk in chunks, hashing as it goes — the file is never held whole in memory, so upload size is bounded by disk, not RAM.
2. The content hash is checked against existing recordings. An exact re-upload is recognised as a duplicate and short-circuits rather than re-running an expensive pipeline.
3. A record is registered, the media is moved into permanent storage, and a job is enqueued. The API responds immediately with an identifier and a subscription endpoint — the upload never blocks on processing.
4. A worker dequeues the job and runs the pipeline stage by stage, writing each stage's output to the store as it completes and publishing a progress event.
5. The client, subscribed over a WebSocket, renders live progress with a stage name and percentage. If the connection drops, state is recovered by polling — the socket is an optimisation, not a source of truth.
6. On completion the recording becomes searchable and exportable. On rejection or failure the reason is recorded against the recording and surfaced in the UI.

Every run is recorded with the exact configuration and model versions in force at the time, plus per-stage timings. Reprocessing a recording under new settings is therefore a first-class, repeatable operation, and any two runs can be compared to explain a difference in output.

---

## 4. The Processing Pipeline

The pipeline is a fixed, ordered list of stages. Each stage does one thing, reads what earlier stages produced, and writes its own results. Adding a capability means adding a stage, not editing a monolith — the sentiment capability was added exactly this way.

| # | Stage | What it produces |
|---|---|---|
| 1 | **Validate** | Format, duration and integrity checks. Bad input is rejected here with a specific reason, before any expensive work. |
| 2 | **Preprocess** | Normalized single-channel audio at a fixed sample rate, plus quality metrics and detected speech regions. |
| 3 | **Transcribe** | Text with word-level timestamps and per-word confidence. |
| 4 | **Diarize** | Speaker-change boundaries — "a different voice starts here" — independent of what was said. |
| 5 | **Reconcile** | Merges the two views above into coherent utterances: each block of words attributed to one speaker turn. The sharpest logic in the system, and the most heavily tested. |
| 6 | **Embed** | A voice fingerprint (fixed-length vector) per speaker per recording, plus a text embedding per utterance for semantic search. |
| 7 | **Identify** | Matches each voice fingerprint against the enrolled roster, with a confidence margin and an explicit abstention when the match is not decisive. |
| 8 | **Postprocess** | Cleanup of turn boundaries, punctuation and short-fragment merging. |
| 9 | **Sentiment** | Per-utterance text sentiment, acoustic tone, and a fused score; plus a weighted rollup for the whole recording. |
| 10 | **Index** | Search structures — keyword index and vector index — made queryable. |

Stages carry a relative cost weight used to drive an honest progress bar. A failure inside a stage distinguishes between a **rejection** (the input was unusable — final, no retry) and an **error** (something went wrong — retried with backoff). Confusing the two is how queues either lose work or spin forever on input that will never succeed.

A pipeline run always clears its own derived output first, so re-running is idempotent: a second run replaces the first rather than accumulating duplicates alongside it.

---

## 5. Speaker Identity Model

Identity is the part of the system with real consequences, so it is built to be cautious and reversible.

### Enrollment and matching

- A named profile is built from one or more enrolled voice samples, summarised into a representative centroid vector. Samples that sit too far from the rest are flagged as outliers and excluded, so one mislabelled clip cannot poison a profile.
- Matching is a nearest-neighbour search in vector space, accelerated by an approximate index so it stays fast as the roster grows.
- A match is only asserted when it clears both an absolute similarity threshold *and* a margin over the runner-up. Otherwise the system abstains and says "unknown" — a wrong name is far more damaging than an honest blank.
- Thresholds are calibrated from measured error rates rather than guessed, and recalibration is an operator-triggered run whose result is recorded.

### Unknown speakers

Voices that are never identified are not discarded. They are clustered by similarity, so a recurring unknown voice accumulates across recordings and can be promoted to a named profile in one action — at which point every past occurrence is already attributed to it. Operators can also correct any attribution manually; corrections are audited and feed back into the profile.

### Identity across events

Speaker profiles are global. Grouping recordings into an event is only a label on the recording — the pipeline is identical either way. That is precisely what makes a speaker enrolled during one event automatically recognised in the next, with no per-event setup.

---

## 6. Sentiment and Tone

Two independent signals are computed and stored separately, then fused — rather than collapsed into a single number that hides its own disagreement.

| Signal | Answers | Strength / weakness |
|---|---|---|
| **Text sentiment** | What was said. | A trained multilingual classifier. Understands content regardless of delivery, but blind to how it was spoken. |
| **Acoustic tone** | How it was said. | Pitch and speech-rate movement measured against a baseline learned from that same recording — so "agitated" means agitated *for this voice and this microphone*, not against a fixed global cutoff. Noisy on short or poor-quality audio. |
| **Fused score** | The combined read. | The text score nudged by tone. The nudge is deliberately bounded so delivery can push a borderline utterance over the line but can never, on its own, label a neutral one. |

The interface exposes all three, because the cases where they disagree — bad news delivered flatly, a cheerful-sounding complaint — are the interesting ones. Averaging them into one figure would erase exactly the signal an analyst is looking for.

The recording-level rollup is duration-weighted, so one long, heated exchange outweighs a burst of one-word acknowledgements. A plain average would say the opposite of the truth.

---

## 7. Search

Three retrieval modes over the same corpus of utterances:

- **Keyword** — full-text matching with linguistic stemming. Precise, fast, and the right tool when the operator knows the exact term.
- **Semantic** — nearest-neighbour search over utterance embeddings. Finds meaning without shared wording.
- **Hybrid** — a rank-fusion of both lists. Robust to the failure mode of either one alone, and the sensible default.

Both indexes live in the same relational store as the transcripts. Keeping vectors beside the rows they describe means a search result and its metadata come back in one query, with no second system to keep in sync and no possibility of the two disagreeing.

Results are paginated by offset, which is correct and simple at the scale this deployment targets.

---

## 8. Live Mode

For an in-progress session, the client captures audio in short chunks and posts each one as it is recorded. Each chunk becomes an independent job, so transcription trails the live audio by roughly one chunk instead of waiting for the session to end. Results stream back over the same progress channel used by uploads. The trade-off is deliberate: chunk-local context means slightly weaker accuracy than a full-file pass, in exchange for immediacy.

---

## 9. External Data Enrichment

The system can pull contextual reference data from a public read-only source — session listings, participant rosters, timing data, and published audio — through a thin server-side proxy. The proxy exists so that credentials and network egress stay under server control, the client never talks to third parties directly, and this enrichment path remains cleanly severable: disabling it removes a convenience, never a core capability. All analysis still runs locally on the retrieved audio.

---

## 10. Model Management

- All model weights live on local disk, declared in a single registry that names every model the system may load.
- At worker startup the registry is verified. A missing model is a startup failure, not a runtime surprise halfway through a job.
- Models are loaded once per worker process and held resident. Loading per job would dominate processing time.
- Alternate models for transcription, diarization, embedding and sentiment can be downloaded and activated from the settings interface, so accuracy-versus-speed is an operator decision rather than a redeploy.
- The exact model versions used are stamped onto every run, making any past result explainable after the fact.

---

## 11. Configuration

Configuration is layered: environment values provide the baseline, and operator overrides stored in the database take precedence. Tuning parameters — thresholds, weights, model selection — are therefore adjustable live, and every run captures a snapshot of the settings that produced it. Secrets and infrastructure endpoints remain environment-only and are never editable from the UI.

---

## 12. Security, Auditability and Operations

| Concern | Approach |
|---|---|
| **Authentication** | A single shared credential, checked in one place that every route depends on. Concentrating it in one function is what makes replacing it with enterprise sign-on a contained change rather than a rewrite. |
| **Data residency** | Nothing leaves the deployment. Models are local; inference is local; audio is local. |
| **Untrusted uploads** | User-supplied media and graphics are validated and, where they are served back to browsers, sanitised against a strict allowlist of permitted content. A denylist would be bypassed by the next construct nobody anticipated. |
| **Upload limits** | Size ceilings enforced during streaming, so an oversized file is rejected mid-transfer instead of after it has consumed disk. |
| **Audit trail** | Every consequential action is recorded in a hash-chained log — each entry commits to its predecessor, so silent after-the-fact tampering is detectable. |
| **Deletion** | Removal produces a receipt, giving a defensible record of what was destroyed and when. |
| **Observability** | Structured logs carry a correlation identifier through an entire run, so one recording's journey can be reconstructed from logs alone. Operational metrics are exposed in a standard scrape format. |
| **Health** | Separate liveness and readiness signals, so orchestration can distinguish "the process is up" from "dependencies are reachable and it can actually serve". |

---

## 13. Deployment Topology

In development the datastores run as containers while the application tiers run natively, so code changes reload instantly. In production the same components are containerised uniformly. The boundary is unchanged either way — only the packaging differs.

Scaling follows the tier split: the API scales horizontally on ordinary hardware and holds no session state; the worker scales by adding GPU-attached instances that pull from the shared queue; the datastores scale vertically first, as is usual. No component assumes it is the only instance.

---

## 14. Deliberate Scope Boundaries

These are stated choices with an identified upgrade path, not gaps. Each is isolated behind a single seam so it can be replaced without disturbing the rest of the system.

| Simplification | Holds until | Upgrade path |
|---|---|---|
| Local filesystem for media, not distributed object storage. | The deployment stays single-node. | Swap the storage layer's small surface for an object-store client. |
| A single schema definition applied at first start, not versioned migrations. | The schema is not being upgraded against live production data. | Introduce a migration tool. |
| One shared credential, not per-user accounts and roles. | Operators are a small, trusted team. | Replace the single authorization check with enterprise sign-on and role checks. |
| Offset pagination. | Library size is in the thousands. | Cursor-based pagination. |
| No tenant isolation. | One organisation per deployment. | Add a tenancy key plus row-level access policies. |
| Resource-exhaustion recovery relies on ordinary retry with backoff. | Contention is occasional. | Add a reduced-precision fallback path for retries. |

---

## 15. Design Principles in Force

- **One place per rule.** Anything two tiers must agree on lives in shared code, so they cannot drift apart.
- **One shared ingest path.** New entry points reuse it, so limits, deduplication and auditing apply to them by construction rather than by remembering.
- **Abstain rather than guess.** Where the system is not confident — identification especially — it says so.
- **Store signals separately, fuse at the edge.** Disagreement between signals is information; averaging it away destroys it.
- **Every simplification is named, bounded, and given an upgrade path**, so deferred work is tracked rather than forgotten.
- **Reprocessing is a first-class operation.** Runs are idempotent and stamped with their inputs, so any result can be reproduced or explained.
