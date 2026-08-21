# Graph RAG Engine — Design Plan

Adding a Neo4j-backed relationship layer over the existing Postgres/pgvector store, so
driver history and race events can be traversed as a graph and later fed to an LLM.

---

## The one architectural decision

**Postgres stays the source of truth. Neo4j is a derived read model, rebuildable from
scratch by one script.**

Everything else in this plan follows from that. It means:

- No dual-write consistency problem. The pipeline never writes to Neo4j.
- No Neo4j migrations. Schema changed? `DETACH DELETE` everything and re-project.
- Neo4j going down degrades the graph endpoints, not the pipeline, search, or UI.
- The sync script is idempotent by construction (`MERGE` on stable Postgres IDs).

The failure mode this avoids is the one that kills graph projects: two stores that are
each half-authoritative, drifting, with no way to tell which is right.

## Does this need Neo4j at all?

Honest answer: the entity graph here is small (thousands of nodes) and mostly a star
schema Postgres already joins fine. Neo4j earns its place on exactly three things:

1. **Variable-length path queries** — "which drivers appear in radio calls that mention
   the driver who caused the incident on lap 34" is one Cypher line and a recursive CTE
   nightmare.
2. **A schema an LLM can describe and traverse.** Cypher over a labelled property graph
   is far easier to generate correctly than joins across 15 tables.
3. **Cheap neighbourhood expansion** — pull everything within 2 hops of an utterance in
   one round trip, which is the core primitive of graph RAG.

If none of those materialise, Phase 0 alone still delivers most of the value. Phase 0
does not depend on Neo4j and should ship regardless.

---

## Implementation status

Phases 0 through 3 are built. Three things came out different from the design below; each
is recorded here rather than quietly amended in place, because the reasoning is the useful
part.

| Deviation | Why |
|---|---|
| **No `--since` incremental sync.** `rebuild()` is full-only. | The ceilings section already said a full pass is fine at this scale, and incremental would need change-tracking timestamps that `f1_drivers` and `f1_laps` do not have. Marked in the code with the upgrade path. |
| **Added a `RadioCall` label** (not in the node model below) **and a shared `:Speech` label** on it and `Utterance`. | Without `RadioCall` the graph had F1 sessions, drivers and laps but no F1 *speech* — `radio_calls` holds the transcripts and they have no clip row unless pipelined. The shared `:Speech` label then let `DURING_LAP` and `MENTIONS` be one projection each instead of one per source type. |
| **Radio audio is not retained.** The plan said keep it under `DATA_DIR/f1/`. | The F1 page plays each call straight from its livetiming URL, so a local copy serves nothing today. Reverse it if radio ever goes through the full clip pipeline, which needs a file. |

Phase 3 landed with two additions the design did not anticipate: the hybrid search was
extracted out of `routers/search.py` into `services/search.py` so the context endpoint
anchors on the same ranking rather than a second copy of it, and `radio_calls` gained a
generated `tsv` column — without it, search covered only `utterances` and would have missed
almost the entire F1 corpus.

Phase 4 (the agent layer, `AGENT_LAYER_PLAN.md`) is not built.

**Verified end to end against live Postgres + Neo4j**, on real OpenF1 data (30 sessions,
20 drivers, 1129 laps, 137 radio calls) plus the existing 674-utterance clip corpus. The
run found five bugs that no amount of static checking would have reached:

| Bug | Why only a live run found it |
|---|---|
| `NEO4J_URI` in `.env` stopped the neo4j container booting | The image reads *every* `NEO4J_`-prefixed variable as one of its own config settings. App-side settings are `GRAPH_*` now, and the service no longer inherits `env_file`. |
| OpenF1 timestamps are ISO **strings**; asyncpg binds `TIMESTAMPTZ` from `datetime` only | Sessions, laps and radio calls all persisted zero rows while drivers (no timestamp column) worked. `_persist` logs and continues by design, so the page looked healthy over an empty cache. |
| Auto-seeded 3-letter driver codes were pure noise | Measured on the corpus: **zero** true positives (ASR spells names out, never emits "VER") and real false ones — "three seconds *per* sector" linked to Perez. Seeding now derives surnames only. |
| `refresh_aliases` never pruned stale `auto` rows | Insert-only seeding meant the acronym fix could not take effect on an existing database. Auto rows are now deleted and rebuilt each sync; `manual` rows are untouched. |
| `/v1/search` returned `id` in hybrid mode but `utterance_id` in the other two | A pre-existing inconsistency, inherited when the ranking was extracted. Normalised to `utterance_id`; nothing in the UI consumed it yet. |

---

## Phase 0 — Persist F1 data (no Neo4j)

**Required before anything graph-shaped is possible.** Today the Race Radio feature is
fully ephemeral: audio deleted, results emitted over WebSocket, nothing stored. Every
page load re-hits OpenF1; nothing accumulates.

New file `db/f1.sql`, written idempotently and `\ir`'d from `db/init.sql` — the same
pattern `db/races.sql` already uses so it can be replayed against a live database.

```sql
CREATE TABLE IF NOT EXISTS f1_sessions (
  session_key  INT PRIMARY KEY,          -- OpenF1 key, natural PK
  year         INT, name TEXT, session_type TEXT,
  circuit      TEXT, country TEXT,
  date_start   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS f1_drivers (
  session_key   INT REFERENCES f1_sessions(session_key) ON DELETE CASCADE,
  driver_number INT,
  full_name TEXT, name_acronym TEXT, team_name TEXT,
  -- the seam that links a driver to a voice we can recognize
  profile_id    UUID REFERENCES speaker_profiles(id) ON DELETE SET NULL,
  PRIMARY KEY (session_key, driver_number)
);

CREATE TABLE IF NOT EXISTS f1_laps (
  session_key   INT, driver_number INT, lap_number INT,
  date_start    TIMESTAMPTZ,
  lap_duration  REAL, sector_1 REAL, sector_2 REAL, sector_3 REAL,
  is_pit_out    BOOLEAN,
  PRIMARY KEY (session_key, driver_number, lap_number)
);

CREATE TABLE IF NOT EXISTS radio_calls (
  id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_key   INT, driver_number INT,
  recording_url TEXT NOT NULL,
  recorded_at   TIMESTAMPTZ,             -- OpenF1 team_radio.date
  clip_id       UUID REFERENCES clips(id) ON DELETE SET NULL,
  text          TEXT, mood TEXT, features JSONB,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS radio_url_uniq ON radio_calls(recording_url);
```

Code changes:

- `api/app/routers/f1.py` — upsert session/driver/lap rows on each proxy call (write-through
  cache; OpenF1 data is immutable once a session is over, so this is safe and makes the
  page fast on the second load). `POST /ingest` inserts the `radio_calls` row up front.
- `worker/worker/main.py:analyze_f1_radio_job` — take `radio_call_id`, `UPDATE` the row with
  `text`/`mood`/`features` before `emit_f1_result`. Keep the audio (drop the `os.remove`)
  under `DATA_DIR/f1/` so calls stay playable.
- Better: route radio through the **normal clip pipeline** so calls get diarization,
  speaker ID, utterances, sentiment, and search for free. `radio_calls.clip_id` is the
  seam. Bigger change; do it in Phase 0.5 if the ephemeral path is demo-critical.

`clips` also needs `recorded_at TIMESTAMPTZ` (nullable) — upload time is not recording
time, and every temporal edge in Phase 2 depends on knowing when audio actually happened.

**Ships alone:** Race Radio history stops evaporating; driver tone becomes queryable over
time; the F1 page loads from Postgres instead of round-tripping OpenF1.

---

## Phase 1 — The graph projection

### Infrastructure

`docker-compose.yml` gets one service (datastores only, matching the existing convention):

```yaml
  neo4j:
    <<: *common
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
      NEO4J_server_memory_heap_max__size: 1G
    volumes: [neo4jdata:/data]
    healthcheck:
      test: ["CMD-SHELL", "wget -qO- http://localhost:7474 || exit 1"]
      interval: 10s
      retries: 20
    ports: ["7474:7474", "7687:7687"]
```

`.env.example`: `NEO4J_URI=bolt://localhost:7687`, `NEO4J_USER`, `NEO4J_PASSWORD`,
`GRAPH_ENABLED=false`. Dependency: `neo4j` (official driver, async) in `pyproject.toml`'s `api` extra
only — the worker never touches it.

`common/graph.py` mirrors `common/db.py`: module-level async driver, `init_driver`,
`close_driver`, `run(cypher, **params)`. ~40 lines, no OGM. Same reasoning as the
existing "no ORM" note in `common/db.py` — the queries are the interesting part.

### Node and relationship model

```
(:Driver     {number, code, full_name, team})
(:Speaker    {profile_id, name})            -- speaker_profiles
(:Session    {session_key, name, type, year, date_start})
(:Race       {race_id, name, circuit, race_date})   -- races table
(:Circuit    {name, country})
(:Team       {name})
(:Clip       {clip_id, filename, duration_s, recorded_at, sentiment, mood})
(:Utterance  {utt_id, text, start_s, end_s, sentiment, sentiment_score, mood, ts})
(:Lap        {session_key, driver_number, number, duration_s, date_start})
```

```
(Driver)   -[:DRIVES_FOR]->    (Team)
(Driver)   -[:DROVE_IN]->      (Session)
(Driver)   -[:VOICE_OF]->      (Speaker)        -- from f1_drivers.profile_id
(Session)  -[:AT]->            (Circuit)
(Session)  -[:PART_OF]->       (Race)
(Lap)      -[:BY]->            (Driver)
(Lap)      -[:IN]->            (Session)
(Lap)      -[:NEXT]->          (Lap)            -- temporal spine, per driver
(Clip)     -[:IN_RACE]->       (Race)
(Clip)     -[:FROM_SESSION]->  (Session)
(Utterance)-[:IN_CLIP]->       (Clip)
(Utterance)-[:SPOKEN_BY]->     (Speaker)
(Utterance)-[:NEXT]->          (Utterance)      -- conversational order within a clip
(Speaker)  -[:TALKED_WITH {n_clips}]-> (Speaker) -- co-occurrence, non-F1 side too
```

Utterances carry no embedding. Vectors stay in pgvector, where HNSW + RRF hybrid search
already works (`api/app/routers/search.py`). A second copy in Neo4j's vector index would
be the same numbers with no added capability.

### The sync script

`scripts/graph_sync.py` — one file, two modes:

- `--full`: `MATCH (n) DETACH DELETE n`, then project everything. The recovery path.
- `--since <ts>`: project only clips/utterances/radio calls newer than a timestamp.

Reads Postgres with `common.db`, writes with `common.graph`. Every write is a `MERGE` on
the Postgres primary key, so re-running is a no-op. Constraints created first:

```cypher
CREATE CONSTRAINT IF NOT EXISTS FOR (u:Utterance) REQUIRE u.utt_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (c:Clip)      REQUIRE c.clip_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (s:Speaker)   REQUIRE s.profile_id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (d:Driver)    REQUIRE (d.number, d.session_key) IS UNIQUE;
```

Batched `UNWIND $rows AS row MERGE ...` — one round trip per few thousand rows, not one
per node.

Triggered three ways, cheapest first:

1. `python scripts/graph_sync.py --full` by hand.
2. `POST /v1/admin/graph/sync` on the existing admin router, so it's a Settings button and
   lands in `audit_log` like every other mutating action.
3. *Later, only if needed:* an arq job enqueued at the end of `pipeline.process_clip`.
   Skip until stale-by-minutes is actually a problem — a hackathon demo re-syncs on demand.

**Ships alone:** Neo4j Browser becomes a live visual of the whole dataset. That is,
frankly, the single best demo artifact in this plan.

---

## Phase 2 — The edges that make it a graph and not a join

Two edge types carry all the analytical weight.

### `(Utterance)-[:DURING_LAP]->(Lap)`

The money edge. Turns "was he stressed on the lap he lost three tenths?" into one hop.

```
utterance_ts = clip.recorded_at + utterance.start_s
lap covers    [lap.date_start, lap.date_start + lap.lap_duration]
```

Computed in the sync script, not at query time. **This depends entirely on
`clips.recorded_at` being populated** — for OpenF1 radio it comes free from
`team_radio.date`; for manual uploads it is null and the edge is simply absent. Say so in
the UI rather than guessing from upload time.

Clock skew between OpenF1's radio timestamps and lap timestamps is real. Add a
`GRAPH_LAP_MATCH_TOLERANCE_S` setting (default 2.0) to the existing
`settings_overrides` mechanism so it can be tuned from Settings without a restart, like
every other pipeline knob.

### `(Utterance)-[:MENTIONS]->(Driver|Team|Circuit)`

Entity linking. **Dictionary matcher, not an NER model.** The vocabulary is closed and
tiny — ~20 drivers, 10 teams, ~24 circuits per season, all already in `f1_drivers` /
`f1_sessions`. A normalized alias table plus word-boundary regex beats spaCy here on both
accuracy (ASR mangles names in ways a general NER has never seen) and dependency count.

Alias table lives in Postgres so it's editable:

```sql
CREATE TABLE IF NOT EXISTS f1_aliases (
  alias       TEXT PRIMARY KEY,   -- lowercased: "checo", "max", "seven", "vercetti"
  entity_type TEXT NOT NULL,      -- driver | team | circuit
  entity_key  TEXT NOT NULL
);
```

Seed from driver surnames, acronyms, and team names; append ASR mishearings as they show
up in real transcripts. The existing `dmetaphone` phonetic-search idea listed as scoped
out in the README is the natural upgrade path when the literal matcher misses.

### `(Speaker)-[:TALKED_WITH]->(Speaker)`

Co-occurrence in the same clip, weighted by count. Cheap, and it makes the non-F1 side of
the product (the general Speaker Directory) a graph too, not just an F1 feature.

**Ships alone:** "show me every stressed call within two laps of a pit stop" becomes a
single query, and the Race page can render tone against the lap chart from graph data
instead of the current ad-hoc join.

---

## Phase 3 — Retrieval: `POST /v1/graph/context`

The actual RAG mechanism. Four steps, and the last one is the one everyone skips.

**1. Anchor (Postgres).** Reuse `api/app/routers/search.py` verbatim — hybrid RRF over
FTS + pgvector returns the top-k utterance IDs. Do not rebuild retrieval in Neo4j.

**2. Expand (Neo4j).** One Cypher query pulls the neighbourhood of those IDs:

```cypher
UNWIND $utt_ids AS uid
MATCH (u:Utterance {utt_id: uid})
OPTIONAL MATCH (u)-[:SPOKEN_BY]->(sp:Speaker)<-[:VOICE_OF]-(d:Driver)
OPTIONAL MATCH (u)-[:IN_CLIP]->(c:Clip)-[:FROM_SESSION]->(s:Session)
OPTIONAL MATCH (u)-[:DURING_LAP]->(l:Lap)
OPTIONAL MATCH (l)-[:NEXT*1..2]-(nearby:Lap)
OPTIONAL MATCH (u)-[:MENTIONS]->(m)
OPTIONAL MATCH (prev:Utterance)-[:NEXT]->(u)-[:NEXT]->(next:Utterance)
RETURN u, sp, d, c, s, l, collect(DISTINCT nearby) AS laps,
       collect(DISTINCT m) AS mentions, prev, next
```

**3. Serialize.** `render_context(rows) -> str`. This is the step that decides whether the
whole thing works. Dumping raw JSON into a prompt wastes tokens and reads badly; emit
short natural-language lines instead:

```
[Monaco GP 2024, lap 42] VER (Red Bull) — stressed, negative (-0.6):
  "I can't get past, the tyres are gone"
  lap 42: 74.8s (+0.9s vs lap 41). Mentions: LEC.
  next call, lap 44: "box box, understood" — calm
```

**4. Return** both the rendered text and the structured rows, so the UI can render a graph
view from the same call the LLM will later consume.

Keep `GRAPH_ENABLED=false` as a hard gate: when Neo4j is down or unsynced, the endpoint
404s and search/Race pages are unaffected.

**Ships alone:** a "Context" panel on the clip and race pages showing the neighbourhood of
any utterance — no LLM required.

---

## Phase 4 — The LLM layer (future)

Do **not** stuff one giant retrieval blob into a prompt. Give the model tools and let it
navigate — that is the entire point of having built a graph.

Four tools, via the Anthropic SDK's tool runner (`client.beta.messages.tool_runner`), on
`claude-opus-5` with adaptive thinking:

| Tool | Purpose |
|---|---|
| `search_radio(query, driver?, session?, sentiment?)` | Hybrid search → anchor utterances |
| `neighborhood(utterance_id, depth=1)` | Phase 3 expansion, rendered |
| `driver_timeline(driver, session)` | Laps + calls in order, one driver, one race |
| `compare_drivers(a, b, session)` | Tone and pace side by side |

Notes that matter:

- **Read-only.** No tool writes to Postgres or Neo4j. The graph is a lens; corrections stay
  in the existing human-in-the-loop UI (`/clips/{id}/speakers/{label}/assign`).
- Cache the system prompt + tool definitions (`cache_control: {"type": "ephemeral"}`) —
  they are stable across every request and are exactly the prefix prompt caching is for.
- **This is the first thing in the product that sends data off the box.** The README's
  central claim is "no audio and no text ever leaves the box"; an LLM layer breaks it for
  text. Gate it behind an explicit `LLM_ENABLED=false` default and say so plainly in the
  README, or run a local model through the same tool interface. Do not let this become an
  accidental privacy regression — the tool boundary is the seam that makes either choice
  possible.

---

## Build order and honest sizing

| Phase | Depends on | Ships alone as | Rough size |
|---|---|---|---|
| 0 — Persist F1 | — | Race Radio history stops evaporating | 1 SQL file, 2 files touched |
| 1 — Projection | 0 | Neo4j Browser demo of the whole dataset | 1 compose service, `common/graph.py`, `scripts/graph_sync.py` |
| 2 — Temporal + entity edges | 0, 1 | Tone-vs-lap and mention queries | sync script + `f1_aliases` seed |
| 3 — `/v1/graph/context` | 2 | Context panel in the UI | 1 router, 1 renderer |
| 4 — LLM layer | 3 | Natural-language Q&A | tool defs + one agent loop |

Phase 0 is the only mandatory prerequisite and the only one with no Neo4j in it. If the
graph work is dropped entirely, Phase 0 still stands on its own.

## Known ceilings, stated up front

- **`recorded_at` is the weak link.** Every temporal edge depends on it. OpenF1 radio has
  it; manual uploads do not. Absent edges, not guessed ones.
- **Dictionary entity linking misses novel phrasings.** Upgrade path is phonetic matching
  (`dmetaphone`), already noted as a scoped-out item in the README.
- **Full re-sync is O(n) over all utterances.** Fine to tens of thousands of rows on a
  laptop. Past that, `--since` incremental becomes mandatory rather than optional.
- **Neo4j Community is single-database and unclustered.** Correct for a single-box
  on-prem deployment, which is what this product is.
- **Tone is still a heuristic** (`worker/worker/audio/tone.py`). The graph makes that
  signal traversable; it does not make it more accurate. Same caveat as the README's.
