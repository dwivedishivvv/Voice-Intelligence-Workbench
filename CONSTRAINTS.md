# Limitations & Constraints

Companion to `DESIGN_CONTEXT.md`, which covers what the application does. This one covers
what it **cannot** do, what it **refuses** to do, and what it does **imprecisely** — the
boundaries any design has to work inside rather than around.

Each entry says what the limit is, why it exists, and — where relevant — **what the user
sees**, since a constraint the interface hides becomes a lie the interface tells.

Same scope as the design doc: the Race Radio screen is excluded; the race data model it fed
(sessions, participants, laps, lap alignment) remains and is covered as *race context*.

---

## 1. Accuracy limits — the ones that reach the user

These matter most, because they are the difference between a system that is trusted
appropriately and one that is trusted too much.

### 1.1 Tone is a heuristic, not an emotion model

Tone (`calm` / `stressed` / `tired`) comes from thresholds over acoustic features — speech
rate, pitch mean and variability, energy variance, voiced ratio. It is **not** a trained
speech-emotion model, and it has never been validated against labelled emotional data.

It calibrates per speaker where it can: given a participant and a session, it builds a
running baseline from that person's calm speech rather than judging against fixed global
cutoffs, because one person's normal pitch is not everyone's.

**What the user must see:** a reading of the audio, never a statement about a person.
"Their voice reads stressed", not "they were stressed". Every surface that shows tone
carries this framing today, and it is load-bearing rather than decorative.

### 1.2 Transcription hallucinates on poor audio

The ASR model can invent fluent text from noise. Observed in this corpus: a noisy radio
clip transcribed as fluent Icelandic, and the tone classifier then labelled that invented
text *stressed*. Filters exist (no-speech probability, average log-probability, compression
ratio, a boilerplate list) and they catch the common cases — not all of them.

**Consequence:** a small share of transcript text is confident-sounding fiction, and any
downstream reading of it inherits the error.

### 1.3 Identification abstains — often, and by design

Speaker identification is deliberately conservative. It will decline rather than guess:

- Below a reliability floor it refuses to judge at all (`abstained`)
- A low-reliability sample faces a **higher** similarity bar, not the same one
- The top match must beat the runner-up by a margin, or the result is abstention rather
  than a coin flip

In the working corpus this produced roughly **7 in 10 voices unattributed** — far more
`unknown` and `abstained` than `confident`. That is the system behaving correctly, not
failing, but it means **most speech in the product has no name attached**, and the
interface has to make that its normal state rather than an error state.

### 1.4 Thresholds are tuned for short, compressed audio

The identification and clustering thresholds were calibrated on ~70 short radio-style clips
(median under 4 seconds of speech per speaker, heavily compressed). Clean long-form
recordings behave differently and would justify different numbers. The defaults are
**right for this data, not universal**, and are editable live in Settings for that reason.

### 1.5 Sentiment and tone can disagree

Text sentiment and acoustic tone are separate models, stored separately alongside their
fusion. They disagree regularly — calm delivery of negative words, and vice versa.

**This is a feature, not a defect.** The design must surface disagreement rather than
average it away; a single fused number destroys the most interesting signal in the data.

### 1.6 Diarization has an overlap ceiling

Overlapping speech is detected and flagged, not separated. Heavy overlap raises a warning
on the clip and pushes it into the review queue. Very short turns are dropped and nearby
turns merged, so rapid back-and-forth is smoothed.

---

## 2. Hard input limits

| Limit | Value | Why |
|---|---|---|
| Max clip duration | **90 seconds** | Pipeline is tuned for short recordings end to end |
| Min clip duration | 0.5 s | Below this there is nothing to analyse |
| Max upload size | **50 MB** | |
| Min total speech | 0.5 s | A clip with less real speech is rejected as empty |
| Job timeout | 300 s | |
| Retry attempts | 3 | |

A rejected clip is a **stated outcome**, not a failure: too short, no speech detected,
unreadable container. The reason is always given, and rejection is visually distinct from
an error.

**Design consequence:** this is a short-clip product. Anything resembling an hour-long
interview is out of scope for the current pipeline, and the UI's whole rhythm (waveform,
per-word transcript, single-screen review) assumes clips measured in seconds.

---

## 3. Coverage gaps — things that exist for some data and not others

This is the most design-relevant category: **the same screen shows different amounts of
information depending on where the recording came from.**

| Capability | Available when | Absent when |
|---|---|---|
| Word-level timing and confidence | Clip went through the full pipeline | Lightweight-analysed audio |
| Speaker identification | Full pipeline | Lightweight path (transcript + tone only) |
| Semantic search | Clip has embeddings | Lightweight-analysed audio — **keyword search only** |
| Placement on a timeline | The recording has a wall-clock timestamp | Manually uploaded audio has none |
| Participant / team context | Recording is linked to a known session | Standalone uploads |
| Adjacent lines (before/after) | Multi-utterance clip | Single-utterance recordings |

**The timestamp gap is the sharpest one.** Aligning speech to a moment on a timeline
requires knowing when the audio was actually recorded. Recordings that arrive with a
timestamp get it; manually uploaded files do not, because upload time is not recording
time. The system leaves the context **absent rather than estimated** — the design must
render "we don't know when this happened" as an ordinary state.

---

## 4. Search constraints

- **Three modes** — keyword, semantic, and a fused hybrid. Hybrid is the default and
  combines the two by rank, not by score, because their scores are not comparable.
- **Semantic search requires embeddings**, so it silently covers less of the corpus than
  keyword search does.
- **Phonetic search is not implemented.** ASR mangles proper nouns constantly — a surname
  comes back misspelled a dozen different ways across a corpus — and there is no fuzzy-name
  matching to recover from that. Searching for a name finds only the spellings the model
  happened to produce.
- **No filters in the search API beyond speaker** — no date range, no status, no tag.
- **Search has no UI at all.** The endpoint exists and works in all three modes; nothing in
  the interface calls it. Library filters by *filename* only. This is the single largest
  gap between what the backend can do and what a user can reach.

---

## 5. Entity linking constraints

Mentions of people, teams and places are found with a **dictionary matcher over a closed
vocabulary**, not a named-entity model. Aliases are derived automatically from known
participants and venues, and can be extended by hand.

Deliberate exclusions, each measured against the real corpus:

- **First names are not auto-derived.** "Max" and "Lando" are ordinary words in speech.
- **Short codes are not auto-derived.** Three-letter participant codes measured *zero*
  true positives (transcripts spell names out) while producing false ones — "three seconds
  **per** sector" linked to a participant named Perez. Zero signal, real noise.
- **Surnames that are ordinary words still slip through.** A participant named Stroll will
  match the word "stroll". This is accepted rather than fixed, because filtering it would
  remove a real person.

**Consequence:** mention links are high-precision and incomplete. Absence of a link does
not mean absence of a mention.

---

## 6. The graph layer

- **It is a derived read model.** Postgres is the source of truth; the graph is projected
  from it and can be wiped and rebuilt at any time.
- **It can be stale.** Projection runs on demand — manually or from an admin action — not
  on every pipeline completion. Anything processed since the last projection is missing
  from graph-backed views until someone re-syncs.
- **Full rebuild only.** There is no incremental sync. Fine at current scale, seconds to
  run; it is a rebuild-everything operation regardless.
- **Optional.** Disabled by default. With it off, graph-backed views degrade rather than
  break — search still works, the pipeline is untouched.

---

## 7. The Ask / agent layer

This is the newest and least proven part of the product.

- **Off by default.** It is the only feature that sends transcript text off the machine,
  and it is enabled in deployment configuration, not from Settings — deliberately, so
  turning it on is a considered act rather than a checkbox anyone with access can tick.
- **Read-only.** It can query the corpus; it cannot modify it. Corrections stay with the
  human.
- **Citation compliance is inconsistent.** The model is instructed to cite a source for
  every claim, and it does not always comply — observed answering *correctly* with no
  citation at all. The interface handles this by falling back to the sources its tools
  surfaced, labelled distinctly so an uncited claim never borrows a cited one's authority.
- **Answer quality is unmeasured.** There is no evaluation suite. Correctness has been
  confirmed on a small number of hand-checked questions, on one model. Others tried in the
  same harness produced wrong answers, and one fabricated a detail the data did not
  contain.
- **Latency is tens of seconds** per question, sometimes longer, because each answer
  involves several tool calls.
- **Bounded reasoning.** A hard cap on tool iterations stops runaway loops; a question that
  needs more hops returns a partial answer and says so.
- **No per-user cost accounting.** One shared key, no attribution, no quota.

---

## 8. Scale constraints

Current ceilings, in the order they would bite:

| Area | Ceiling | What happens past it |
|---|---|---|
| Processing concurrency | **One clip at a time** | Models are not thread-safe and share one pool; jobs queue rather than parallelise |
| Pagination | Offset/limit | Correct to a few thousand clips; deep pages get slow |
| Graph rebuild | Single transaction | Fine to ~100k nodes on the configured heap |
| Entity matching | Every transcript × every alias | Linear in both; fine at current corpus size |
| Storage | Local disk | Single node only; no object storage |
| Deployment | Single machine | No horizontal scaling path today |

Processing is serialised deliberately: running two jobs through one shared model pool
corrupted the process. Parallelism means more worker **processes**, not more threads.

---

## 9. Operational constraints

- **Single shared API key.** No user accounts, no roles, no multi-tenancy, no per-user
  data separation. Everyone with the key sees everything.
- **No schema migrations.** The database initialises once on a fresh volume; incremental
  schema files are written to be safely re-runnable against a live database. There is no
  versioned upgrade path.
- **Retention is configured but not enforced.** The setting exists and is surfaced; the
  process that would act on it does not exist. The current behaviour is keep-forever.
- **No GPU out-of-memory recovery** beyond ordinary job retry.
- **GPU strongly assumed.** It runs on CPU, roughly twenty times slower.
- **Desktop-only layout.** Fixed sidebar, single content column, no responsive or mobile
  handling anywhere.

---

## 10. Privacy posture — constraints that are the point

These are deliberate restrictions, not gaps. They constrain the design and should be
treated as requirements.

- **Transcripts are never logged** unless explicitly enabled. Diagnostics record shapes,
  counts and identifiers — not content.
- **The audit log is append-only and hash-chained.** Every mutating action is recorded with
  a hash of the previous entry, so tampering breaks the chain. It also means **audit
  entries cannot be deleted**, which is why user text is deliberately kept out of them.
- **Deletion leaves a receipt** — proof that something was deleted, after the data is gone.
- **The core pipeline makes no outbound network calls.** Two optional features do; both are
  off by default, and the one that transmits text is documented as such wherever it is
  configured.

**Design consequence:** anywhere the interface offers to "log", "share", "export" or
"send", it is touching a property the product otherwise guarantees. Those affordances need
to look different from ordinary actions.

---

## 11. What is deliberately not built

Listed so a design does not assume them:

- Phonetic / fuzzy name search
- A 2D embedding visualisation of the speaker space
- Object storage, multi-node deployment
- SSO, RBAC, multi-tenancy
- Cursor pagination
- Trained speech-emotion recognition
- Automatic retention enforcement
- Incremental graph sync
- Any search UI
- Mobile or responsive layouts
- A guided first-run experience

---

## 12. The five that most shape the interface

If a design accounts for nothing else:

1. **Most speech has no confirmed speaker.** Unattributed is the normal state, not an
   error state, and the four different reasons for it are distinguishable facts.
2. **Tone and sentiment are readings, not truths** — and they are allowed to disagree with
   each other.
3. **Context is uneven.** Two recordings on the same screen may carry very different
   amounts of surrounding information, and the missing parts are genuinely unknown rather
   than not-yet-loaded.
4. **Clips are short.** Seconds to ninety seconds, never hours.
5. **Corrections are the product's engine.** They are training data, and the interface is
   where they are either invited or buried.
