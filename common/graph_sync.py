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
import time
import uuid
from collections import Counter
from itertools import combinations
from datetime import datetime, timezone
from typing import NamedTuple

import structlog

from . import db, entities, graph
from .config import get_settings, get_effective_settings
from .graph import GraphUnavailable

log = structlog.get_logger(__name__)


class Projection(NamedTuple):
    name: str
    sql: str
    cypher: str
    # Settings field names bound to $1, $2, ... in `sql`. Named rather than positional
    # values so a projection stays a static, testable declaration instead of something
    # that has to be built at call time once a query needs a tunable threshold.
    params: tuple[str, ...] = ()
    # A boolean Settings field that switches this projection off when false.
    when: str | None = None


# Uniqueness constraints double as indexes, which is what makes the MERGEs below cheap:
# without them every MERGE is a label scan and the projection degrades quadratically.
CONSTRAINTS = [
    "CREATE CONSTRAINT utt_id    IF NOT EXISTS FOR (u:Utterance) REQUIRE u.utt_id IS UNIQUE",
    "CREATE CONSTRAINT speech_id IF NOT EXISTS FOR (s:Speech)    REQUIRE s.speech_id IS UNIQUE",
    "CREATE CONSTRAINT clip_id   IF NOT EXISTS FOR (c:Clip)      REQUIRE c.clip_id IS UNIQUE",
    "CREATE CONSTRAINT spk_id    IF NOT EXISTS FOR (s:Speaker)   REQUIRE s.profile_id IS UNIQUE",
    "CREATE CONSTRAINT voice_id  IF NOT EXISTS FOR (v:Voice)     REQUIRE v.cluster_id IS UNIQUE",
    "CREATE CONSTRAINT topic_nm  IF NOT EXISTS FOR (t:Topic)     REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT tag_nm    IF NOT EXISTS FOR (t:Tag)       REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT lang_code IF NOT EXISTS FOR (l:Language)  REQUIRE l.code IS UNIQUE",
    "CREATE CONSTRAINT entity_nm IF NOT EXISTS FOR (e:Entity)    REQUIRE e.name IS UNIQUE",
]

# The model, in one place:
#
#   (:Speaker)  an enrolled person              (:Voice)    an unidentified recurring voice
#   (:Clip)     a processed recording           (:Utterance:Speech) one diarized turn
#   (:Topic)    an automatic topic tag          (:Tag)      an operator's own tag
#   (:Language) a detected language             (:Entity)   a person, company or place
#                                                           named in the transcripts
#
#   (Utterance)-[:IN_CLIP]->(Clip)  (Utterance)-[:SPOKEN_BY]->(Speaker)  (Utterance)-[:NEXT]->
#   (Speaker)-[:APPEARS_IN]->(Clip)   (Voice)-[:HEARD_IN]->(Clip)
#   (Speaker)-[:TALKED_WITH]->(Speaker)   (Speaker)-[:DISCUSSED]->(Topic)
#   (Clip)-[:ABOUT]->(Topic)  (Clip|Speaker)-[:TAGGED]->(Tag)  (Clip)-[:IN_LANGUAGE]->(Language)
#   (Utterance|Clip)-[:MENTIONS]->(Entity)   (Speaker)-[:MENTIONED]->(Entity)
#   (Entity)-[:CO_MENTIONED]->(Entity)

NODE_PROJECTIONS = [
    Projection("Speaker", """
        SELECT id AS profile_id, display_name, status, n_enrollments, total_speech_s,
               notes, tags, created_at
        FROM speaker_profiles""", """
        UNWIND $rows AS row
        MERGE (s:Speaker {profile_id: row.profile_id})
        SET s.name = row.display_name, s.status = row.status,
            s.n_enrollments = row.n_enrollments, s.total_speech_s = row.total_speech_s,
            s.notes = row.notes, s.tags = row.tags, s.created_at = row.created_at
    """),
    # Recurring voices nobody has named yet. A promoted cluster is already a Speaker, and
    # one no clip voice points at any more (its members were reassigned or reprocessed) is
    # a stale grouping with nothing to connect to -- it would only float on the map.
    Projection("Voice", """
        SELECT k.id AS cluster_id, k.label, k.n_members, k.total_speech_s
        FROM speaker_clusters k
        WHERE k.promoted_to IS NULL
          AND EXISTS (SELECT 1 FROM clip_speakers cs WHERE cs.cluster_id = k.id)""", """
        UNWIND $rows AS row
        MERGE (v:Voice {cluster_id: row.cluster_id})
        SET v.name = row.label, v.n_members = row.n_members,
            v.total_speech_s = row.total_speech_s
    """),
    # Only COMPLETE clips: anything else has no finished transcript, and a half-processed
    # clip in the graph is worse than an absent one.
    Projection("Clip", """
        SELECT c.id AS clip_id, c.filename, c.duration_s, c.language, c.n_speakers,
               c.needs_review, c.tags, c.created_at, q.grade
        FROM clips c LEFT JOIN quality_metrics q ON q.clip_id = c.id
        WHERE c.status = 'COMPLETE'""", """
        UNWIND $rows AS row
        MERGE (c:Clip {clip_id: row.clip_id})
        SET c.name = row.filename, c.filename = row.filename, c.duration_s = row.duration_s,
            c.language = row.language, c.n_speakers = row.n_speakers,
            c.needs_review = row.needs_review, c.tags = row.tags,
            c.created_at = row.created_at, c.quality = row.grade
    """),
    Projection("Utterance", """
        SELECT u.id AS utt_id, u.idx, u.text, u.start_s, u.end_s, u.mean_word_conf
        FROM utterances u JOIN clips c ON c.id = u.clip_id
        WHERE c.status = 'COMPLETE'""", """
        UNWIND $rows AS row
        MERGE (u:Utterance {utt_id: row.utt_id})
        SET u:Speech, u.speech_id = row.utt_id,
            u.text = row.text, u.idx = row.idx,
            u.start_s = row.start_s, u.end_s = row.end_s, u.asr_conf = row.mean_word_conf
    """, when="graph_utterance_nodes"),
    Projection("Topic", """
        SELECT DISTINCT substr(t, 7) AS name
        FROM clips c, unnest(c.auto_tags) t
        WHERE c.status = 'COMPLETE' AND t LIKE 'topic:%'""", """
        UNWIND $rows AS row MERGE (t:Topic {name: row.name})
    """),
    Projection("Tag", """
        SELECT DISTINCT t AS name FROM (
          SELECT unnest(c.tags) AS t FROM clips c WHERE c.status = 'COMPLETE'
          UNION SELECT unnest(tags) FROM speaker_profiles
        ) x WHERE t <> ''""", """
        UNWIND $rows AS row MERGE (t:Tag {name: row.name})
    """),
    Projection("Language", """
        SELECT DISTINCT language AS code FROM clips c
        WHERE c.status = 'COMPLETE' AND language IS NOT NULL""", """
        UNWIND $rows AS row MERGE (l:Language {code: row.code}) SET l.name = row.code
    """),
]

REL_PROJECTIONS = [
    Projection("IN_CLIP", """
        SELECT u.id AS utt_id, u.clip_id FROM utterances u
        JOIN clips c ON c.id = u.clip_id WHERE c.status = 'COMPLETE'""", """
        UNWIND $rows AS row
        MATCH (u:Utterance {utt_id: row.utt_id}), (c:Clip {clip_id: row.clip_id})
        MERGE (u)-[:IN_CLIP]->(c)
    """, when="graph_utterance_nodes"),
    # Speaker resolution matches the rest of the codebase: clip_speakers is the
    # authoritative identity row and utterances.profile_id can lag it.
    Projection("SPOKEN_BY", """
        SELECT u.id AS utt_id, coalesce(u.profile_id, cs.profile_id) AS profile_id
        FROM utterances u
        JOIN clips c ON c.id = u.clip_id
        LEFT JOIN clip_speakers cs ON cs.clip_id = u.clip_id AND cs.local_label = u.local_label
        WHERE c.status = 'COMPLETE'
          AND coalesce(u.profile_id, cs.profile_id) IS NOT NULL""", """
        UNWIND $rows AS row
        MATCH (u:Utterance {utt_id: row.utt_id}), (s:Speaker {profile_id: row.profile_id})
        MERGE (u)-[:SPOKEN_BY]->(s)
    """, when="graph_utterance_nodes"),
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
    """, when="graph_utterance_nodes"),
    # Who is in which recording, and how much of it they carry.
    Projection("APPEARS_IN", """
        SELECT cs.profile_id, cs.clip_id, cs.speech_s, cs.n_turns, cs.talk_share,
               cs.match_result::text AS match_result, cs.match_score
        FROM clip_speakers cs JOIN clips c ON c.id = cs.clip_id
        WHERE c.status = 'COMPLETE' AND cs.profile_id IS NOT NULL""", """
        UNWIND $rows AS row
        MATCH (s:Speaker {profile_id: row.profile_id}), (c:Clip {clip_id: row.clip_id})
        MERGE (s)-[r:APPEARS_IN]->(c)
        SET r.speech_s = row.speech_s, r.n_turns = row.n_turns, r.talk_share = row.talk_share,
            r.match = row.match_result, r.score = row.match_score
    """),
    Projection("HEARD_IN", """
        SELECT cs.cluster_id, cs.clip_id, cs.speech_s, cs.talk_share
        FROM clip_speakers cs JOIN clips c ON c.id = cs.clip_id
        WHERE c.status = 'COMPLETE' AND cs.profile_id IS NULL
          AND cs.cluster_id IS NOT NULL""", """
        UNWIND $rows AS row
        MATCH (v:Voice {cluster_id: row.cluster_id}), (c:Clip {clip_id: row.clip_id})
        MERGE (v)-[r:HEARD_IN]->(c)
        SET r.speech_s = row.speech_s, r.talk_share = row.talk_share
    """),
    # Who talks to whom, from co-occurrence in the same clip. Undirected in meaning, so
    # only the a<b half is stored — the other direction is the same fact written twice.
    Projection("TALKED_WITH", """
        SELECT a.profile_id AS a, b.profile_id AS b, count(DISTINCT a.clip_id)::int AS n_clips
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
    Projection("ABOUT", """
        SELECT c.id AS clip_id, substr(t, 7) AS topic
        FROM clips c, unnest(c.auto_tags) t
        WHERE c.status = 'COMPLETE' AND t LIKE 'topic:%'""", """
        UNWIND $rows AS row
        MATCH (c:Clip {clip_id: row.clip_id}), (t:Topic {name: row.topic})
        MERGE (c)-[:ABOUT]->(t)
    """),
    # A speaker discussed a topic when they spoke in a recording about it. Weighted by how
    # many such recordings, so a person's recurring subjects draw heavier on the map.
    Projection("DISCUSSED", """
        SELECT cs.profile_id, substr(t, 7) AS topic, count(DISTINCT c.id)::int AS n_clips
        FROM clip_speakers cs
        JOIN clips c ON c.id = cs.clip_id
        CROSS JOIN LATERAL unnest(c.auto_tags) t
        WHERE c.status = 'COMPLETE' AND cs.profile_id IS NOT NULL AND t LIKE 'topic:%'
        GROUP BY cs.profile_id, substr(t, 7)""", """
        UNWIND $rows AS row
        MATCH (s:Speaker {profile_id: row.profile_id}), (t:Topic {name: row.topic})
        MERGE (s)-[r:DISCUSSED]->(t) SET r.n_clips = row.n_clips
    """),
    Projection("CLIP_TAGGED", """
        SELECT c.id AS clip_id, unnest(c.tags) AS tag FROM clips c
        WHERE c.status = 'COMPLETE'""", """
        UNWIND $rows AS row
        MATCH (c:Clip {clip_id: row.clip_id}), (t:Tag {name: row.tag})
        MERGE (c)-[:TAGGED]->(t)
    """),
    Projection("SPEAKER_TAGGED", """
        SELECT id AS profile_id, unnest(tags) AS tag FROM speaker_profiles""", """
        UNWIND $rows AS row
        MATCH (s:Speaker {profile_id: row.profile_id}), (t:Tag {name: row.tag})
        MERGE (s)-[:TAGGED]->(t)
    """),
    Projection("IN_LANGUAGE", """
        SELECT c.id AS clip_id, c.language FROM clips c
        WHERE c.status = 'COMPLETE' AND c.language IS NOT NULL""", """
        UNWIND $rows AS row
        MATCH (c:Clip {clip_id: row.clip_id}), (l:Language {code: row.language})
        MERGE (c)-[:IN_LANGUAGE]->(l)
    """),
]

PROJECTIONS = NODE_PROJECTIONS + REL_PROJECTIONS

# Entities come from the Python extractor in common/entities.py rather than a single query,
# so they are projected separately -- see _entity_rows(). Dict order is write order: the
# Entity nodes go in before the edges that MATCH them.
ENTITY_CYPHER = {
    "Entity": """
        UNWIND $rows AS row
        MERGE (e:Entity {name: row.name}) SET e.kind = row.kind, e.n_mentions = row.n
    """,
    "UTT_MENTIONS": """
        UNWIND $rows AS row
        MATCH (u:Utterance {utt_id: row.speech_id}), (e:Entity {name: row.entity})
        MERGE (u)-[:MENTIONS]->(e)
    """,
    "CLIP_MENTIONS": """
        UNWIND $rows AS row
        MATCH (c:Clip {clip_id: row.clip_id}), (e:Entity {name: row.entity})
        MERGE (c)-[r:MENTIONS]->(e) SET r.n = row.n
    """,
    "SPEAKER_MENTIONED": """
        UNWIND $rows AS row
        MATCH (s:Speaker {profile_id: row.profile_id}), (e:Entity {name: row.entity})
        MERGE (s)-[r:MENTIONED]->(e) SET r.n = row.n
    """,
    # Named in the same recording. Stored a<b, like TALKED_WITH.
    "CO_MENTIONED": """
        UNWIND $rows AS row
        MATCH (a:Entity {name: row.a}), (b:Entity {name: row.b})
        MERGE (a)-[r:CO_MENTIONED]->(b) SET r.n_clips = row.n_clips
    """,
}

# Every unit of transcribed speech with its clip and, where known, its speaker.
SPEECH_TEXT_SQL = """
    SELECT u.id AS speech_id, u.clip_id, coalesce(u.profile_id, cs.profile_id) AS profile_id,
           u.text
    FROM utterances u
    JOIN clips c ON c.id = u.clip_id
    LEFT JOIN clip_speakers cs ON cs.clip_id = u.clip_id AND cs.local_label = u.local_label
    WHERE c.status = 'COMPLETE' AND u.text IS NOT NULL AND u.text <> ''
"""


def _rows(records) -> list[dict]:
    """asyncpg Records -> plain dicts the neo4j driver accepts. UUIDs are the reason this
    exists: neo4j has no UUID type and rejects the objects asyncpg returns, while
    datetimes map to its temporal types natively and pass through untouched."""
    return [{k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in dict(r).items()}
            for r in records]


def entity_rows(speech: list[dict], min_mentions: int) -> dict[str, list[dict]]:
    """Every row each ENTITY_CYPHER statement takes, from the transcripts. Pure, so it is
    testable without either database."""
    found = entities.extract([r["text"] for r in speech], min_mentions)
    known = {name.lower(): name for name in found}

    utt, by_clip, by_speaker = [], Counter(), Counter()
    clip_entities: dict[str, set[str]] = {}
    for r in speech:
        for name in entities.mentions(r["text"], known):
            utt.append({"speech_id": r["speech_id"], "entity": name})
            by_clip[(r["clip_id"], name)] += 1
            clip_entities.setdefault(r["clip_id"], set()).add(name)
            if r.get("profile_id"):
                by_speaker[(r["profile_id"], name)] += 1

    pairs: Counter = Counter()
    for names in clip_entities.values():
        for a, b in combinations(sorted(names), 2):
            pairs[(a, b)] += 1

    return {
        "Entity": [{"name": n, "kind": e["kind"], "n": e["n"]} for n, e in found.items()],
        "UTT_MENTIONS": utt,
        "CLIP_MENTIONS": [{"clip_id": c, "entity": e, "n": n} for (c, e), n in by_clip.items()],
        "SPEAKER_MENTIONED": [{"profile_id": p, "entity": e, "n": n}
                              for (p, e), n in by_speaker.items()],
        "CO_MENTIONED": [{"a": a, "b": b, "n_clips": n} for (a, b), n in pairs.items()],
    }


async def rebuild(cfg=None) -> dict:
    """Wipe and re-project everything. Returns per-step row counts.

    ponytail: full rebuild only, no incremental mode. At a few thousand clips this is
    seconds, and `--since` would need change-tracking timestamps on tables that have none
    (clip_speakers, speaker_clusters). Add it when a full pass stops being fast enough — the
    projection statements themselves would not change, only their WHERE clauses.
    """
    cfg = cfg or await get_effective_settings()

    for c in CONSTRAINTS:
        await graph.run(c)

    # ponytail: single-transaction wipe, fine to ~100k nodes against the 1G heap the
    # compose file allocates. Past that, switch to
    # `MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS`.
    await graph.run("MATCH (n) DETACH DELETE n")

    counts: dict[str, int] = {}
    for p in PROJECTIONS:
        if p.when and not getattr(cfg, p.when):
            continue
        args = [getattr(cfg, name) for name in p.params]
        counts[p.name] = await graph.write(p.cypher, _rows(await db.fetch(p.sql, *args)))

    speech = _rows(await db.fetch(SPEECH_TEXT_SQL))
    for name, rows in entity_rows(speech, cfg.graph_entity_min_mentions).items():
        if name == "UTT_MENTIONS" and not cfg.graph_utterance_nodes:
            continue
        counts[name] = await graph.write(ENTITY_CYPHER[name], rows)

    return counts


SYNC_JOB = "graph_sync_job"


async def schedule(redis, debounce_s: float) -> None:
    """Ask the worker for a rebuild, coalesced with every other request in the same window.

    Every request inside one `debounce_s` window maps to the same arq job id, deferred to the
    end of that window, and arq drops a second enqueue of an id it already holds — so a
    burst of fifty completions costs one rebuild instead of fifty. A request that lands while
    a window's rebuild is already running falls into the *next* window, so nothing written
    after a rebuild started is ever left out of the graph.
    """
    window = max(float(debounce_s), 1.0)
    bucket = int(time.time() // window)
    await redis.enqueue_job(
        SYNC_JOB, _job_id=f"graph-sync-{bucket}",
        _defer_until=datetime.fromtimestamp((bucket + 1) * window, tz=timezone.utc))


async def resync(redis=None) -> None:
    """Best-effort rebuild after a mutation that changes graph identity: a speaker
    rename/merge/delete, or a clip delete. Without this the graph keeps citing a
    name or an id a human just explicitly corrected or removed, and stays wrong until
    the next clip completes or someone hits the admin sync button.

    With an arq `redis`, the rebuild is scheduled on the worker (see schedule()) and the
    request returns at once instead of waiting out a full re-projection.

    Same posture as worker/pipeline.py's _sync_graph: Neo4j is a derived read model
    (GRAPH_RAG_PLAN.md), so a down or slow graph must never fail the request that
    triggered this."""
    cfg = await get_effective_settings()
    if not cfg.graph_enabled:
        return
    if redis is not None:
        try:
            await schedule(redis, cfg.graph_sync_debounce_s)
            return
        except Exception as e:
            log.warning("graph_resync_schedule_failed", error=str(e)[:200])
    try:
        await rebuild(cfg)
    except GraphUnavailable:
        pass
    except Exception as e:
        log.warning("graph_resync_failed", error=str(e)[:200])


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
