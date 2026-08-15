# Speaker Intelligence Workbench — Design Context

Reference for designing the application's interface. Describes what the product does, every
screen, every state a screen can be in, and the data available to each view.

**Scope note.** The Race Radio screen is deliberately excluded. The underlying race data it
populated — sessions, participants, laps, and the alignment of speech to a lap — remains
and surfaces elsewhere (Races, Ask), so it is documented here as *race context* rather than
as a feature with a screen of its own.

---

## 1. What the product is

An on-premises voice-intelligence workbench. You give it a recording of people talking; it
tells you **who spoke, what they said, and how they sounded**, then lets you correct it,
search it, group it, and ask questions about it.

Everything runs on the box. Models sit on local disk. The core pipeline makes no outbound
network calls.

### The single idea the design has to carry

**The system would rather say "I don't know" than guess.**

This is not a limitation to hide — it is the product's core claim to trustworthiness, and
the interface is where that claim is either honoured or thrown away. It shows up
everywhere:

- Speaker identification **abstains** when the audio is too poor or two profiles are too
  close. Result: a large share of speech is attributed to nobody.
- Tone is a **reading of the audio**, never a statement about a person's feelings.
- Text sentiment and voice tone are stored **separately** so they can disagree, and the
  disagreement is shown rather than averaged away.
- Confidence is attached to almost everything — per word, per speaker match, per profile.

A design that renders uncertainty as though it were fact breaks the product. A design that
makes uncertainty legible is the product.

---

## 2. Who uses it, and why

| User | Comes to do | Leaves with |
|---|---|---|
| **Analyst** | Understand one recording in depth | Who spoke, what was said, where the doubt is |
| **Curator** | Correct wrong or missing attributions | A corpus that gets more accurate over time |
| **Researcher** | Find moments across many recordings | Search results, or an answer with sources |
| **Operator** | Tune quality and manage models | Thresholds adjusted, models activated |

The curator role is worth designing for explicitly: **a correction is training data, not a
UI edit.** Fixing a speaker re-enrolls that voice and recomputes the profile, so the next
recording is identified better. The interface should make corrections feel consequential
rather than clerical.

---

## 3. Core capabilities

| Capability | What the user sees |
|---|---|
| Transcription | Word-level text with per-word confidence and automatic language detection |
| Diarization | The recording split into speaker turns, colour-coded |
| Speaker identification | Turns matched to enrolled people, with a confidence outcome |
| Enrollment | Naming a voice; building a profile from multiple samples |
| Clustering | Unnamed voices grouped across recordings, promotable to a named person |
| Sentiment & tone | Per-utterance text sentiment plus an acoustic tone reading |
| Search | Keyword, meaning-based, or both fused |
| Ask | Natural-language questions answered with cited evidence |
| Races | User-created groupings of recordings around one event |
| Live | Real-time microphone transcription |
| Export | SRT, VTT, RTTM, JSON, TXT |
| Settings | Every processing threshold, editable live |

---

## 4. Information architecture

Fixed left sidebar (240px), single content column. Current navigation:

```
Library      (/)              every processed recording
Ask          (/ask)           question the corpus
Upload       (/upload)        add a recording
Live         (/live)          microphone capture
Races        (/races)         event groupings
Speakers     (/speakers)      people and unclaimed voices
Settings     (/settings)      pipeline tuning
                              └── Clip detail (/clips/:id) — reached, not navigated to
```

Clip detail has no nav entry: it is always arrived at from Library, Races, search, or an
Ask citation. It is the **most-visited screen in the product** and the natural centre of
gravity — worth weighing in any restructure.

The sidebar footer shows a live "All systems operational" pulse.

---

## 5. The core object model

Understanding these five nouns is enough to design every screen.

```
Clip ─────── one uploaded recording
 ├─ quality      SNR, clipping, bandwidth → a grade: good | fair | poor
 ├─ turns        who spoke when (diarization)
 ├─ utterances   sentences, each with speaker, timing, sentiment, tone
 │   └─ words    each with its own confidence
 └─ speakers     one row per distinct voice in this clip, with a match outcome

Speaker profile ── a named person, built from enrollments
Cluster ────────── an unnamed recurring voice, promotable to a profile
Race ───────────── a user-made grouping of clips around one event
Speech ─────────── any unit of transcribed speech (used by Ask)
```

### Values the UI renders constantly

**Clip status** — the pipeline's position. Terminal states are `COMPLETE` (green),
`FAILED` / `DEAD` (red), `REJECTED` (amber — refused for a stated reason, e.g. too short,
no speech), `QUEUED` (grey). Anything else is an in-progress stage and renders with a
spinner: `VALIDATING`, `PREPROCESSING`, `TRANSCRIBING`, `DIARIZING`, `RECONCILING`,
`EMBEDDING`, `IDENTIFYING`, `POSTPROCESSING`, `SENTIMENT`, `INDEXING`.

**Identification outcome** — per speaker per clip. Four values, and they are the
uncertainty model in miniature:

| Outcome | Meaning | Design implication |
|---|---|---|
| `confident` | Matched and auto-labelled | Show the name |
| `suggested` | Close, but under the bar | Show the name *as a proposal* awaiting confirmation |
| `unknown` | No profile close enough | Show as an unnamed voice, invite naming |
| `abstained` | Audio too poor to judge | Say *why* it declined — this is not the same as `unknown` |

**Tone** — `calm` (green) / `stressed` (red) / `tired` (amber). Always phrased as a reading
of the voice.

**Sentiment** — `negative` / `neutral` / `positive`, plus a signed score. Stored as three
values: text sentiment, acoustic tone, and their fusion.

**Reliability** — 0–1, from speech duration, SNR and quality grade. Drives whether
identification is even attempted.

**Cohesion** — how much a profile's enrollments agree with each other. Low cohesion is a
warning that a profile has been polluted.

---

## 6. Screens

### 6.1 Library `/`

The default landing screen: a table of every recording.

**Shows** — filename, status badge, duration, detected language, speaker count, upload time.

**Actions** — filter by filename; toggle **Needs review**; open a clip; edit tags/notes in
a dialog; delete.

**States** — loading (skeleton rows) · empty ("No clips yet" with an Upload call-to-action)
· populated · filtered-to-empty.

**Needs review** is the curator's queue: the pipeline flags clips where it abstained, found
heavy overlap, or graded the audio poor. It is currently a toggle button and reads as a
minor filter, despite being the primary entry point for the highest-value user task.

### 6.2 Clip detail `/clips/:id`

The heart of the product. Everything known about one recording.

**Shows**
- Header: filename, status, warnings as amber alerts (each with a code and explanation)
- Audio player with a waveform **segmented and coloured per speaker**
- Toggle between **original** and **processed** (denoised, loudness-normalised) audio
- Speakers panel: one card per voice — talk share, reliability, match score, outcome
- Transcript: speaker-tagged utterances; **per-word confidence on hover**; clicking any
  word or utterance seeks the audio
- The currently playing utterance is highlighted and auto-scrolled into view

**Actions** — **Correct** a speaker (name it, confirm it, or mark unknown) · Reprocess with
current settings · Export (SRT/VTT/RTTM/JSON/TXT) · edit tags and notes.

**Deep link** — `?t=<seconds>` opens the clip and seeks to that moment. This is what makes
an Ask citation verifiable rather than decorative.

**States** — loading skeleton · processing (live per-stage progress over a websocket) ·
complete · rejected (with the reason) · failed (with the error).

### 6.3 Ask `/ask`

Natural-language questions answered from the corpus, with sources.

**Shows** — a conversation. Each answer carries:
- Inline numbered chips where the answer cites a source; clicking scrolls to that source
- **Evidence cards** — speaker, session, lap and its delta, the quote, tone badges, any
  entities mentioned, and a **"Listen at 1:12"** link into the clip at that moment
- A provenance strip: which tools ran, which model, token counts

**Actions** — ask; click an example prompt; follow a citation into the audio.

**States** — empty (three example questions) · thinking (live tool-by-tool progress:
"Searching the corpus" → "Pulling the surrounding context") · answered with evidence ·
answered *without* explicit citations, where the panel is relabelled **"Sources consulted
(the model did not cite directly)"** — deliberately distinct, so an uncited claim never
borrows a cited one's authority · declined · **disabled**, explaining that this is the one
feature that sends text off the box and is switched on in configuration, not in Settings.

A standing footnote reads: *tone labels are heuristic readings of the audio, not
measurements of how anyone felt.*

### 6.4 Upload `/upload`

**Shows** — a drop zone (audio or video, ≤90s), file details once chosen, and live
per-stage progress with a weighted progress bar once submitted.

**Actions** — drag or browse · submit · jump to results.

**States** — idle · file selected · uploading · processing (per-stage) · complete (routes
to the clip) · rejected (named reason) · failed · **duplicate** (same audio already
processed — jumps to the existing result rather than reprocessing).

### 6.5 Live `/live`

Real-time microphone transcription.

**Shows** — recording indicator, a running transcript in chunks, a current tone badge, and
a per-chunk tone strip across the session.

**Actions** — start · stop.

**States** — idle · requesting microphone permission · recording · chunk processing ·
error.

Chunks are cut at natural pauses rather than on a fixed timer, so text does not split
mid-sentence. Speaker identification is **not** run per chunk — voices unmatched to an
enrolled profile are labelled for this session only, and anything too short to judge is
labelled "Speaker (unclear)" rather than given a fabricated identity.

### 6.6 Speakers `/speakers`

Two stacked panels: people, and voices that might become people.

**Enrolled speakers** — name, enrollment count, total speech, status
(`provisional` / `confirmed` / `locked`), and a **cohesion warning** when a profile's
samples disagree with each other.

**Unclaimed clusters** — recurring unnamed voices, each with a playable **montage** (a
stitched sample of that voice across clips) and a one-click **Promote** to a named profile,
which retroactively updates every clip the voice appears in.

**Actions** — promote a cluster · merge profiles · delete with reassignment · drop a bad
enrollment · play a montage.

**States** — loading · no speakers yet ("promote a cluster below") · no clusters · both
populated · promoting (inline name entry).

### 6.7 Races `/races`

A grouping of recordings around one event, with two tabs.

**Recordings tab** — every clip in the race, searchable by transcript, filterable by
sentiment, mood and status. Bulk drag-and-drop upload.

**Analysis tab** — aggregate view across the race: per-voice filtering, sentiment and tone
distribution, and an optional SVG track outline as a visual anchor.

**Race context** — a race may be linked to a real session, which brings in participants and
a lap timeline. Where that link exists, speech can be placed **on the lap it happened on**,
and evidence cards show the lap and its time delta. Where it does not (most manually
uploaded audio, which has no recording timestamp), that context is simply absent — it is
never estimated.

**Voices** in a race can be named inline, which promotes the cluster globally.

**States** — no races · creating a race (name, circuit, date, optional session link,
optional SVG) · race with no recordings ("drop some audio above") · populated · uploading.

### 6.8 Settings `/settings`

Every processing threshold, grouped into tabs: Ingest, Pre-processing, Quality grading,
Diarization, Speaker identification, Transcription, Jobs & retention, Graph.

**Shows** — per field: current value, default, whether it is overridden, and when it was
changed. A header counts how many settings are currently overridden. Each field resets
individually.

Plus two special tabs:
- **Models** — what is active per category, alternatives available to download, and a
  one-click activate.
- **System** — read-only: device (CPU/GPU), precision, model versions, and the
  configuration that can only change by restart, shown *with the reason* rather than
  silently omitted.

Changes apply to the **next job** — no restart. The distinction between live-tunable and
restart-required is surfaced honestly rather than hidden.

---

## 7. Cross-cutting patterns

### Progress
Long operations stream per-stage progress over a websocket: a weighted progress bar with
the named stage, not a spinner. Used by upload, live, and Ask (which streams the tools it
is running).

### Correction as a first-class act
Wherever an attribution appears, correcting it is one click away: name it, confirm a
suggestion, or mark it unknown. The interface should convey that the correction improves
future results.

### Confidence made visible
Per-word confidence on hover; per-speaker reliability and match score; profile cohesion
warnings. The design language for "how sure is this" is currently inconsistent across
screens and is a prime candidate for unification.

### Audio as the source of truth
Every claim can be traced back to the sound: clicking a word seeks the player, an Ask
citation deep-links to the moment, and a cluster can be auditioned as a montage before
naming.

---

## 8. Visual system in place

**Theme** — dark, with a light/dark variant hook (`.dark` class variant).

**Tokens** (CSS custom properties): `background`, `foreground`, `card`, `popover`,
`primary`, `secondary`, `muted`, `accent`, `border`, `input`, `ring`, `sidebar`,
`sidebar-border`, `sidebar-foreground`, plus semantic `success`, `warning`, `destructive`
and a five-step categorical `chart-1…5`. Radii: `sm`/`md`/`lg`/`xl`. Two font families:
`sans` and `mono` (mono for timings, scores, ids).

**Semantic colour usage** is consistent and worth preserving:
- success/green → calm, complete, confident
- warning/amber → tired, rejected, low cohesion, needs attention
- destructive/red → stressed, failed
- primary → active state, links, citations

**Components** (shadcn/ui): alert, avatar, badge, button, card, dialog, dropdown-menu,
input, label, progress, scroll-area, select, separator, skeleton, slider, sonner (toasts),
switch, table, tabs, textarea, tooltip.

**Motion** — `fade-in` on page transitions, `slide-up` on list items, spin for processing,
ping for the live status dot. Restrained and consistent.

**Speaker colour** — each speaker gets a stable colour from a fixed palette, used
identically in the waveform and the transcript so the two read as one thing.

---

## 9. What each view needs from the data

For designing around real payload shapes.

| View | Available per item |
|---|---|
| Library row | filename, status, duration, language, speaker count, created, tags, needs-review flag |
| Clip detail | quality metrics + grade, turns (speaker, start, end, overlap), utterances (text, timing, speaker, sentiment, tone, confidence), words (text, timing, confidence), per-clip speakers (talk share, reliability, match score, margin, outcome, runner-up) |
| Speakers | display name, status, enrollment count, total speech, cohesion, per-enrollment quality and outlier flag |
| Clusters | member count, total speech, cohesion, montage audio |
| Races | clips with sentiment/tone rollups, per-voice breakdown, optional session link, optional track SVG |
| Ask evidence | speech id, quote, speaker/participant, team, session, lap + previous-lap delta, tone, sentiment (fused and text-only), entities mentioned, adjacent lines, clip id + offset for deep linking |

---

## 10. Open design questions

Worth resolving in any redesign:

1. **How should abstention look?** Four identification outcomes exist but only two read
   clearly today. `abstained` (declined to judge, with a reason) and `unknown` (no match
   found) are different facts and currently look alike.
2. **Where does the curator live?** "Needs review" is a filter toggle on Library, but
   reviewing flagged clips is the highest-value recurring task in the product.
3. **Is clip detail one screen or several?** It carries audio, waveform, transcript,
   speakers, corrections, warnings and exports at once.
4. **How prominent should Ask be?** It answers questions the other screens require manual
   work to answer, but it is also the only feature that leaves the box, and its evidence
   panel is what makes it trustworthy.
5. **Should confidence have one visual language?** Word confidence, match score,
   reliability and cohesion are four expressions of the same underlying idea.
6. **What does an empty install look like?** Every screen currently has its own empty
   state; there is no guided first run.
