"""Project Postgres into Neo4j.

Postgres is the source of truth; Neo4j is a derived read model rebuilt from it (see
GRAPH_RAG_PLAN.md). Nothing in the pipeline ever writes to Neo4j, so there is no
dual-write consistency problem to manage and no graph migrations to write — if the shape
is wrong, wipe and re-project.

Every write is a MERGE keyed on the Postgres primary key, so a re-run is a no-op rather
than a duplicate. Run it as `python -m common.graph_sync`, or from the api via
POST /v1/admin/graph/sync.

PROJECTIONS is deliberately data rather than a sequence of calls: it keeps each SQL query
next to the Cypher it feeds, and it lets tests/unit/test_graph_projections.py check that
the two halves agree without a database. Both ways they can disagree fail silently — a
`row.x` with no matching column writes nulls at a healthy row count, and a MATCH naming a
label nothing projects matches zero rows without raising.
"""
import argparse
import asyncio
import uuid
from typing import NamedTuple

from . import db, entities, graph
from .config import get_settings, get_effective_settings


class Projection(NamedTuple):
    name: str
    sql: str
    cypher: str
    # Settings field names bound to $1, $2, ... in `sql`. Named rather than positional
    # values so a projection stays a static, testable declaration instead of something
    # that has to be built at call time once a query needs a tunable threshold.
    params: tuple[str, ...] = ()


# Uniqueness constraints double as indexes, which is what makes the MERGEs below cheap:
# without them every MERGE is a label scan and the projection degrades quadratically.
CONSTRAINTS = [
    "CREATE CONSTRAINT utt_id    IF NOT EXISTS FOR (u:Utterance) REQUIRE u.utt_id IS UNIQUE",
    "CREATE CONSTRAINT clip_id   IF NOT EXISTS FOR (c:Clip)      REQUIRE c.clip_id IS UNIQUE",
    "CREATE CONSTRAINT spk_id    IF NOT EXISTS FOR (s:Speaker)   REQUIRE s.profile_id IS UNIQUE",
    "CREATE CONSTRAINT race_id   IF NOT EXISTS FOR (r:Race)      REQUIRE r.race_id IS UNIQUE",
    "CREATE CONSTRAINT sess_id   IF NOT EXISTS FOR (s:Session)   REQUIRE s.session_key IS UNIQUE",
    "CREATE CONSTRAINT call_id   IF NOT EXISTS FOR (c:RadioCall) REQUIRE c.call_id IS UNIQUE",
    "CREATE CONSTRAINT speech_id IF NOT EXISTS FOR (s:Speech)    REQUIRE s.speech_id IS UNIQUE",
    "CREATE CONSTRAINT team_nm   IF NOT EXISTS FOR (t:Team)      REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT evt_nm    IF NOT EXISTS FOR (e:EventType) REQUIRE e.name IS UNIQUE",
    "CREATE CONSTRAINT circ_nm   IF NOT EXISTS FOR (c:Circuit)   REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT drv_key   IF NOT EXISTS FOR (d:Driver)    REQUIRE (d.session_key, d.number) IS UNIQUE",
    "CREATE CONSTRAINT lap_key   IF NOT EXISTS FOR (l:Lap)       REQUIRE (l.session_key, l.driver_number, l.number) IS UNIQUE",
]

# Utterance and RadioCall both carry a second `:Speech` label. They are genuinely the same
# kind of thing — a unit of transcribed speech with a tone read — and differ only in how
# much of the pipeline has touched them. Modelling that shared identity explicitly is what
# lets the edges that don't care about the difference (DURING_LAP, MENTIONS) be one
# projection each instead of one per source label, and gives Phase 3 retrieval a single
# label to anchor on.

NODE_PROJECTIONS = [
    Projection("Speaker", """
        SELECT id AS profile_id, display_name, status, n_enrollments, total_speech_s
        FROM speaker_profiles""", """
        UNWIND $rows AS row
        MERGE (s:Speaker {profile_id: row.profile_id})
        SET s.name = row.display_name, s.status = row.status,
            s.n_enrollments = row.n_enrollments, s.total_speech_s = row.total_speech_s
    """),
    Projection("Race", """
        SELECT id AS race_id, name, circuit, race_date FROM races""", """
        UNWIND $rows AS row
        MERGE (r:Race {race_id: row.race_id})
        SET r.name = row.name, r.circuit = row.circuit, r.race_date = row.race_date
    """),
    # Only COMPLETE clips: anything else has no finished transcript, and a half-processed
    # clip in the graph is worse than an absent one.
    Projection("Clip", """
        SELECT id AS clip_id, filename, duration_s, language, n_speakers,
               sentiment, sentiment_score, mood, recorded_at, created_at
        FROM clips WHERE status = 'COMPLETE'""", """
        UNWIND $rows AS row
        MERGE (c:Clip {clip_id: row.clip_id})
        SET c.filename = row.filename, c.duration_s = row.duration_s,
            c.language = row.language, c.n_speakers = row.n_speakers,
            c.sentiment = row.sentiment, c.sentiment_score = row.sentiment_score,
            c.mood = row.mood, c.recorded_at = row.recorded_at, c.created_at = row.created_at
    """),
    Projection("Utterance", """
        SELECT u.id AS utt_id, u.idx, u.text, u.start_s, u.end_s,
               u.sentiment, u.sentiment_score, u.text_sentiment, u.mood
        FROM utterances u JOIN clips c ON c.id = u.clip_id
        WHERE c.status = 'COMPLETE'""", """
        UNWIND $rows AS row
        MERGE (u:Utterance {utt_id: row.utt_id})
        SET u:Speech, u.speech_id = row.utt_id,
            u.text = row.text, u.idx = row.idx,
            u.start_s = row.start_s, u.end_s = row.end_s,
            u.sentiment = row.sentiment, u.sentiment_score = row.sentiment_score,
            u.text_sentiment = row.text_sentiment, u.mood = row.mood
    """),
    # A vocabulary node per event type. Small and closed, which is what makes
    # (:EventType {name:"overtake_attempt"})<-[:READS_AS]-(:Speech) a one-hop query
    # instead of a scan for a relationship property.
    Projection("EventType", """
        SELECT DISTINCT event_type AS name, family FROM speech_events""", """
        UNWIND $rows AS row
        MERGE (e:EventType {name: row.name}) SET e.family = row.family
    """),
    Projection("Session", """
        SELECT session_key, name, session_type, year, date_start, circuit, country
        FROM f1_sessions""", """
        UNWIND $rows AS row
        MERGE (s:Session {session_key: row.session_key})
        SET s.name = row.name, s.session_type = row.session_type, s.year = row.year,
            s.date_start = row.date_start, s.circuit = row.circuit, s.country = row.country
    """),
    Projection("Circuit", """
        SELECT DISTINCT circuit AS name, country FROM f1_sessions
        WHERE circuit IS NOT NULL""", """
        UNWIND $rows AS row
        MERGE (c:Circuit {name: row.name}) SET c.country = row.country
    """),
    Projection("Team", """
        SELECT DISTINCT team_name AS name FROM f1_drivers
        WHERE team_name IS NOT NULL""", """
        UNWIND $rows AS row MERGE (t:Team {name: row.name})
    """),
    Projection("Driver", """
        SELECT session_key, driver_number, full_name, name_acronym, team_name
        FROM f1_drivers""", """
        UNWIND $rows AS row
        MERGE (d:Driver {session_key: row.session_key, number: row.driver_number})
        SET d.full_name = row.full_name, d.code = row.name_acronym, d.team = row.team_name
    """),
    Projection("Lap", """
        SELECT session_key, driver_number, lap_number, date_start, lap_duration, is_pit_out
        FROM f1_laps""", """
        UNWIND $rows AS row
        MERGE (l:Lap {session_key: row.session_key, driver_number: row.driver_number,
                      number: row.lap_number})
        SET l.date_start = row.date_start, l.duration_s = row.lap_duration,
            l.is_pit_out = row.is_pit_out
    """),
    # Radio calls get their own primary label rather than being Clips: until one is put
    # through the full pipeline it has no clip row, no diarization and no utterances —
    # only a transcript and a tone read. Calling that a Clip would mean a Clip node with
    # none of a Clip's relationships. radio_calls.clip_id is the seam if they are ever
    # pipelined; the shared :Speech label is what lets the rest of the graph ignore the
    # distinction where it doesn't matter.
    Projection("RadioCall", """
        SELECT id AS call_id, text, mood, recorded_at FROM radio_calls
        WHERE analyzed_at IS NOT NULL AND error IS NULL""", """
        UNWIND $rows AS row
        MERGE (c:RadioCall {call_id: row.call_id})
        SET c:Speech, c.speech_id = row.call_id,
            c.text = row.text, c.mood = row.mood, c.recorded_at = row.recorded_at
    """),
]

REL_PROJECTIONS = [
    Projection("IN_CLIP", """
        SELECT u.id AS utt_id, u.clip_id FROM utterances u
        JOIN clips c ON c.id = u.clip_id WHERE c.status = 'COMPLETE'""", """
        UNWIND $rows AS row
        MATCH (u:Utterance {utt_id: row.utt_id}), (c:Clip {clip_id: row.clip_id})
        MERGE (u)-[:IN_CLIP]->(c)
    """),
    Projection("SPOKEN_BY", """
        SELECT u.id AS utt_id, u.profile_id FROM utterances u
        JOIN clips c ON c.id = u.clip_id
        WHERE c.status = 'COMPLETE' AND u.profile_id IS NOT NULL""", """
        UNWIND $rows AS row
        MATCH (u:Utterance {utt_id: row.utt_id}), (s:Speaker {profile_id: row.profile_id})
        MERGE (u)-[:SPOKEN_BY]->(s)
    """),
    # Conversational order. Built with LAG in SQL rather than by sorting in Cypher: the
    # window function is the natural way to say "the previous utterance in this clip",
    # and it leaves the Cypher a flat pairwise MERGE.
    Projection("UTT_NEXT", """
        SELECT a, b FROM (
          SELECT LAG(u.id) OVER (PARTITION BY u.clip_id ORDER BY u.idx) AS a, u.id AS b
          FROM utterances u JOIN clips c ON c.id = u.clip_id WHERE c.status = 'COMPLETE'
        ) p WHERE a IS NOT NULL""", """
        UNWIND $rows AS row
        MATCH (a:Utterance {utt_id: row.a}), (b:Utterance {utt_id: row.b})
        MERGE (a)-[:NEXT]->(b)
    """),
    Projection("IN_RACE", """
        SELECT id AS clip_id, race_id FROM clips
        WHERE race_id IS NOT NULL AND status = 'COMPLETE'""", """
        UNWIND $rows AS row
        MATCH (c:Clip {clip_id: row.clip_id}), (r:Race {race_id: row.race_id})
        MERGE (c)-[:IN_RACE]->(r)
    """),
    Projection("PART_OF", """
        SELECT id AS race_id, session_key FROM races WHERE session_key IS NOT NULL""", """
        UNWIND $rows AS row
        MATCH (s:Session {session_key: row.session_key}), (r:Race {race_id: row.race_id})
        MERGE (s)-[:PART_OF]->(r)
    """),
    Projection("AT", """
        SELECT session_key, circuit FROM f1_sessions WHERE circuit IS NOT NULL""", """
        UNWIND $rows AS row
        MATCH (s:Session {session_key: row.session_key}), (c:Circuit {name: row.circuit})
        MERGE (s)-[:AT]->(c)
    """),
    Projection("DRIVES_FOR", """
        SELECT session_key, driver_number, team_name FROM f1_drivers
        WHERE team_name IS NOT NULL""", """
        UNWIND $rows AS row
        MATCH (d:Driver {session_key: row.session_key, number: row.driver_number}),
              (t:Team {name: row.team_name})
        MERGE (d)-[:DRIVES_FOR]->(t)
    """),
    Projection("DROVE_IN", """
        SELECT session_key, driver_number FROM f1_drivers""", """
        UNWIND $rows AS row
        MATCH (d:Driver {session_key: row.session_key, number: row.driver_number}),
              (s:Session {session_key: row.session_key})
        MERGE (d)-[:DROVE_IN]->(s)
    """),
    # The edge that makes a driver and a recognized voice the same entity. Rare on
    # purpose: f1_drivers.profile_id is only set once that driver has been enrolled.
    Projection("VOICE_OF", """
        SELECT session_key, driver_number, profile_id FROM f1_drivers
        WHERE profile_id IS NOT NULL""", """
        UNWIND $rows AS row
        MATCH (d:Driver {session_key: row.session_key, number: row.driver_number}),
              (s:Speaker {profile_id: row.profile_id})
        MERGE (d)-[:VOICE_OF]->(s)
    """),
    # One query, two edges: a lap's driver and its session come from the same row, so
    # splitting them would run the identical scan twice.
    Projection("LAP_EDGES", """
        SELECT session_key, driver_number, lap_number FROM f1_laps""", """
        UNWIND $rows AS row
        MATCH (l:Lap {session_key: row.session_key, driver_number: row.driver_number,
                      number: row.lap_number}),
              (d:Driver {session_key: row.session_key, number: row.driver_number}),
              (s:Session {session_key: row.session_key})
        MERGE (l)-[:BY]->(d)
        MERGE (l)-[:IN]->(s)
    """),
    # The temporal spine: lap N -> lap N+1, per driver per session. Phase 3's "lap +/- 2"
    # neighbourhood expansion walks this.
    Projection("LAP_NEXT", """
        SELECT session_key, driver_number, a, b FROM (
          SELECT session_key, driver_number,
                 LAG(lap_number) OVER (PARTITION BY session_key, driver_number
                                       ORDER BY lap_number) AS a,
                 lap_number AS b
          FROM f1_laps
        ) p WHERE a IS NOT NULL""", """
        UNWIND $rows AS row
        MATCH (a:Lap {session_key: row.session_key, driver_number: row.driver_number, number: row.a}),
              (b:Lap {session_key: row.session_key, driver_number: row.driver_number, number: row.b})
        MERGE (a)-[:NEXT]->(b)
    """),
    Projection("CALL_IN_SESSION", """
        SELECT id AS call_id, session_key FROM radio_calls
        WHERE analyzed_at IS NOT NULL AND error IS NULL AND session_key IS NOT NULL""", """
        UNWIND $rows AS row
        MATCH (c:RadioCall {call_id: row.call_id}), (s:Session {session_key: row.session_key})
        MERGE (c)-[:IN_SESSION]->(s)
    """),
    Projection("CALL_BY_DRIVER", """
        SELECT id AS call_id, session_key, driver_number FROM radio_calls
        WHERE analyzed_at IS NOT NULL AND error IS NULL
          AND session_key IS NOT NULL AND driver_number IS NOT NULL""", """
        UNWIND $rows AS row
        MATCH (c:RadioCall {call_id: row.call_id}),
              (d:Driver {session_key: row.session_key, number: row.driver_number})
        MERGE (c)-[:BY_DRIVER]->(d)
    """),
    # --- Phase 2 ------------------------------------------------------------------
    # The edge the whole graph exists for: what was said, on which lap. One projection
    # over both kinds of speech, via the shared :Speech label.
    #
    # Radio calls carry their own wall-clock time from OpenF1. Utterances have to derive
    # one (clip.recorded_at + start_s) and additionally need the clip filed under a race
    # bound to a session, and the speaker enrolled as a driver — so in practice almost all
    # of these edges come from radio today. That is the honest state of the data, not a
    # gap to paper over: a manually uploaded clip with no recorded_at has no place on a
    # lap timeline and gets no edge.
    #
    # Speech inside the tolerance window of a lap boundary links to *both* laps. Two edges
    # says "it was around here"; picking the nearer one would assert a precision the
    # timestamps do not have.
    Projection("DURING_LAP", """
        WITH speech AS (
          SELECT rc.id AS speech_id, rc.session_key, rc.driver_number, rc.recorded_at AS ts
          FROM radio_calls rc
          WHERE rc.analyzed_at IS NOT NULL AND rc.error IS NULL
            AND rc.recorded_at IS NOT NULL
            AND rc.session_key IS NOT NULL AND rc.driver_number IS NOT NULL
          UNION ALL
          SELECT u.id, r.session_key, fd.driver_number,
                 c.recorded_at + make_interval(secs => u.start_s)
          FROM utterances u
          JOIN clips c   ON c.id = u.clip_id
          JOIN races r   ON r.id = c.race_id
          JOIN f1_drivers fd ON fd.session_key = r.session_key AND fd.profile_id = u.profile_id
          WHERE c.status = 'COMPLETE' AND c.recorded_at IS NOT NULL
            AND r.session_key IS NOT NULL AND u.profile_id IS NOT NULL
        )
        SELECT sp.speech_id, l.session_key, l.driver_number, l.lap_number
        FROM speech sp
        JOIN f1_laps l
          ON l.session_key   = sp.session_key
         AND l.driver_number = sp.driver_number
         AND l.date_start IS NOT NULL AND l.lap_duration IS NOT NULL
         AND sp.ts >= l.date_start - make_interval(secs => $1)
         AND sp.ts <  l.date_start + make_interval(secs => l.lap_duration + $1)""", """
        UNWIND $rows AS row
        MATCH (sp:Speech {speech_id: row.speech_id}),
              (l:Lap {session_key: row.session_key, driver_number: row.driver_number,
                      number: row.lap_number})
        MERGE (sp)-[:DURING_LAP]->(l)
    """, params=("graph_lap_match_tolerance_s",)),
    # Who talks to whom, from co-occurrence in the same clip. Undirected in meaning, so
    # only the a<b half is stored — the other direction is the same fact written twice.
    Projection("TALKED_WITH", """
        SELECT a.profile_id AS a, b.profile_id AS b, count(*)::int AS n_clips
        FROM clip_speakers a
        JOIN clip_speakers b ON b.clip_id = a.clip_id AND a.profile_id < b.profile_id
        JOIN clips c ON c.id = a.clip_id
        WHERE c.status = 'COMPLETE'
          AND a.profile_id IS NOT NULL AND b.profile_id IS NOT NULL
        GROUP BY a.profile_id, b.profile_id""", """
        UNWIND $rows AS row
        MATCH (x:Speaker {profile_id: row.a}), (y:Speaker {profile_id: row.b})
        MERGE (x)-[t:TALKED_WITH]->(y) SET t.n_clips = row.n_clips
    """),
    # READS_AS, not IS: the edge records how a line was read by a similarity classifier
    # over one sentence of ASR output, and carries the score so a reader can discount it.
    # Calling it anything stronger would let a model state an inference as a fact.
    Projection("READS_AS", """
        SELECT coalesce(e.utterance_id, e.radio_call_id)::text AS speech_id,
               e.event_type, e.family, e.score, e.margin
          FROM speech_events e""", """
        UNWIND $rows AS row
        MATCH (sp:Speech {speech_id: row.speech_id}), (t:EventType {name: row.event_type})
        MERGE (sp)-[r:READS_AS]->(t)
        SET r.score = row.score, r.margin = row.margin, r.family = row.family
    """),
    # The other car, when the line named exactly one. This is the movement-between-cars
    # edge: it is what turns "who was anyone trying to pass" into a graph traversal.
    Projection("INVOLVES", """
        SELECT coalesce(e.utterance_id, e.radio_call_id)::text AS speech_id,
               e.target_session_key AS session_key, e.target_driver_number AS number,
               e.event_type, e.score
          FROM speech_events e
         WHERE e.target_driver_number IS NOT NULL AND e.target_session_key IS NOT NULL""", """
        UNWIND $rows AS row
        MATCH (sp:Speech {speech_id: row.speech_id}),
              (d:Driver {session_key: row.session_key, number: row.number})
        MERGE (sp)-[r:INVOLVES]->(d)
        SET r.event_type = row.event_type, r.score = row.score
    """),
]

PROJECTIONS = NODE_PROJECTIONS + REL_PROJECTIONS

# MENTIONS is projected separately because the rows come from the Python matcher in
# common/entities.py rather than from a single query — see _mention_rows(). One Cypher per
# entity type, because a Cypher label cannot be parameterised.
MENTION_CYPHER = {
    # Session-scoped: the same surname maps to different car numbers across seasons, so
    # which Driver node is meant is decided by the session the speech belongs to.
    "driver": """
        UNWIND $rows AS row
        MATCH (sp:Speech {speech_id: row.speech_id}),
              (d:Driver {session_key: row.session_key, number: row.entity_number})
        MERGE (sp)-[:MENTIONS]->(d)
    """,
    "team": """
        UNWIND $rows AS row
        MATCH (sp:Speech {speech_id: row.speech_id}), (t:Team {name: row.entity_key})
        MERGE (sp)-[:MENTIONS]->(t)
    """,
    "circuit": """
        UNWIND $rows AS row
        MATCH (sp:Speech {speech_id: row.speech_id}), (c:Circuit {name: row.entity_key})
        MERGE (sp)-[:MENTIONS]->(c)
    """,
}

# Every unit of transcribed speech, with the session it belongs to where one is known.
# session_key is null for clips outside a race; those can still mention a team or a
# circuit, just not a session-scoped driver.
SPEECH_TEXT_SQL = """
    SELECT rc.id AS speech_id, rc.session_key, rc.text
    FROM radio_calls rc
    WHERE rc.analyzed_at IS NOT NULL AND rc.error IS NULL
      AND rc.text IS NOT NULL AND rc.text <> ''
    UNION ALL
    SELECT u.id, r.session_key, u.text
    FROM utterances u
    JOIN clips c ON c.id = u.clip_id
    LEFT JOIN races r ON r.id = c.race_id
    WHERE c.status = 'COMPLETE' AND u.text IS NOT NULL AND u.text <> ''
"""


def _rows(records) -> list[dict]:
    """asyncpg Records -> plain dicts the neo4j driver accepts. UUIDs are the reason this
    exists: neo4j has no UUID type and rejects the objects asyncpg returns, while
    datetimes map to its temporal types natively and pass through untouched."""
    return [{k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in dict(r).items()}
            for r in records]


async def refresh_aliases() -> int:
    """Re-derive the 'auto' half of f1_aliases from current driver/session data.

    Deleted and rebuilt rather than topped up. Insert-only seeding leaves yesterday's
    vocabulary behind forever: a driver who leaves the grid keeps their alias, and a
    tightening of the seeding rules (as when 3-letter codes were dropped for producing
    false matches) never actually takes effect on an existing database.

    'manual' rows — the ASR mishearings and nicknames that are the whole reason the table
    is editable — are never touched, and win over an auto row for the same triple via the
    ON CONFLICT DO NOTHING in each seed statement."""
    await db.execute("DELETE FROM f1_aliases WHERE source = 'auto'")
    for sql in entities.SEED_ALIASES_SQL:
        await db.execute(sql)
    return await db.fetchval("SELECT count(*) FROM f1_aliases")


async def _mention_rows() -> dict[str, list[dict]]:
    """Run the alias matcher over every transcript, bucketed by entity type."""
    alias_rows = await db.fetch("SELECT alias, entity_type, entity_key FROM f1_aliases")
    table: entities.AliasTable = {}
    for r in alias_rows:
        table.setdefault(r["alias"], []).append((r["entity_type"], r["entity_key"]))
    compiled = entities.compile_aliases(table)

    out: dict[str, list[dict]] = {k: [] for k in MENTION_CYPHER}
    for r in await db.fetch(SPEECH_TEXT_SQL):
        for entity_type, entity_key in entities.find_mentions(r["text"], compiled):
            if entity_type not in out:
                continue
            if entity_type == "driver":
                # Needs a session to disambiguate, and a numeric car number to match on.
                if r["session_key"] is None or not entity_key.isdigit():
                    continue
                out["driver"].append({"speech_id": str(r["speech_id"]),
                                      "session_key": r["session_key"],
                                      "entity_number": int(entity_key)})
            else:
                out[entity_type].append({"speech_id": str(r["speech_id"]),
                                         "entity_key": entity_key})
    return out


async def rebuild(cfg=None) -> dict:
    """Wipe and re-project everything. Returns per-step row counts.

    ponytail: full rebuild only, no incremental mode. At a few thousand clips this is
    seconds, and `--since` would need change-tracking timestamps on tables that have none
    (f1_drivers, f1_laps). Add it when a full pass stops being fast enough — the
    projection statements themselves would not change, only their WHERE clauses.
    """
    cfg = cfg or await get_effective_settings()

    for c in CONSTRAINTS:
        await graph.run(c)

    # ponytail: single-transaction wipe, fine to ~100k nodes against the 1G heap the
    # compose file allocates. Past that, switch to
    # `MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS`.
    await graph.run("MATCH (n) DETACH DELETE n")

    counts: dict[str, int] = {"aliases": await refresh_aliases()}
    for p in PROJECTIONS:
        args = [getattr(cfg, name) for name in p.params]
        counts[p.name] = await graph.write(p.cypher, _rows(await db.fetch(p.sql, *args)))

    for entity_type, rows in (await _mention_rows()).items():
        counts[f"MENTIONS_{entity_type.upper()}"] = await graph.write(
            MENTION_CYPHER[entity_type], rows)

    return counts


async def summary() -> dict:
    """Node counts by label and relationship counts by type, read back from the graph.

    Worth having because the relationship projections MATCH their endpoints: a row whose
    endpoint is missing is skipped silently, so comparing what rebuild() *sent* against
    what actually *exists* here is the only way that surfaces."""
    nodes = await graph.run(
        "MATCH (n) UNWIND labels(n) AS l RETURN l AS label, count(*) AS n ORDER BY label")
    rels = await graph.run(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY type")
    return {"nodes": {r["label"]: r["n"] for r in nodes},
            "relationships": {r["type"]: r["n"] for r in rels}}


async def _main():
    ap = argparse.ArgumentParser(description="Project Postgres into Neo4j.")
    ap.add_argument("--full", action="store_true",
                    help="wipe and re-project (currently the only mode; see rebuild())")
    ap.parse_args()

    if not get_settings().graph_enabled:
        # plain ASCII: this goes to a Windows console, where an em-dash renders as mojibake
        raise SystemExit("GRAPH_ENABLED is false - set it to true in .env first.")

    await db.init_pool(get_settings())
    try:
        print("projected:")
        for k, v in (await rebuild()).items():
            print(f"  {k:<18} {v}")
        print("\nin graph:")
        for kind, d in (await summary()).items():
            print(f"  {kind}:")
            for k, v in d.items():
                print(f"    {k:<18} {v}")
    finally:
        await graph.close()
        await db.close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
