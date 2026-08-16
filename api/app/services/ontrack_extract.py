"""Read on-track events out of the corpus and store them.

The taxonomy, the scoring and the abstention rules live in common/ontrack.py, which knows
nothing about databases. This is the pass that joins them to the corpus: embed the
exemplars once, compare every unread piece of speech against them, resolve the other car
where the line names one, and write what survives the thresholds.

Runs on the box. It uses the same MiniLM the search index already uses, so an utterance is
compared against exemplars in the space it is already stored in and no second model is
introduced to disagree with the first.

Re-runnable by design: `POST /v1/admin/ontrack/extract` with force=false only reads speech
that has never been looked at, so an incremental run after an upload is cheap; force=true
re-reads everything, which is what a threshold change or a new event type needs.
"""
import numpy as np

from common import db, entities, ontrack
from common.config import get_effective_settings
from .. import text_embed

# Every exemplar of every type, flattened once so the whole corpus can be scored with one
# matrix multiply rather than a loop over types per utterance.
_EXEMPLAR_CACHE: tuple[np.ndarray, list[str]] | None = None


def _exemplars() -> tuple[np.ndarray, list[str]]:
    """(matrix, type-per-row). Embedded once per process — the model load dominates, and
    the exemplars are a constant of the build."""
    global _EXEMPLAR_CACHE
    if _EXEMPLAR_CACHE is None:
        vectors, owners = [], []
        for event in ontrack.EVENT_TYPES:
            for phrase in event.exemplars:
                vectors.append(text_embed.encode(phrase))
                owners.append(event.name)
        # Ordinary radio scores in the same pass, under a pseudo-type the classifier uses
        # to decline. Scoring it here rather than in a second step keeps every similarity
        # in one matrix and on one scale.
        for phrase in ontrack.NONE_ANCHORS:
            vectors.append(text_embed.encode(phrase))
            owners.append(ontrack.NONE)
        _EXEMPLAR_CACHE = (np.stack(vectors).astype(np.float32), owners)
    return _EXEMPLAR_CACHE


def score_one(embedding, matrix: np.ndarray, owners: list[str]) -> dict[str, float]:
    """Best exemplar similarity per event type.

    Both sides are L2-normalised (SentenceTransformer does it on encode, and the corpus
    embeddings were written the same way), so a dot product is the cosine and there is no
    normalisation to get wrong here.
    """
    sims = matrix @ np.asarray(embedding, dtype=np.float32)
    best: dict[str, float] = {}
    for owner, sim in zip(owners, sims):
        if sim > best.get(owner, -1.0):
            best[owner] = float(sim)
    return best


async def _alias_matcher():
    """The same session-scoped alias table that produces MENTIONS edges. Reused rather than
    re-derived so an event's target and a line's mentions can never disagree about who was
    named."""
    rows = await db.fetch("SELECT alias, entity_type, entity_key FROM f1_aliases")
    table: entities.AliasTable = {}
    for r in rows:
        table.setdefault(r["alias"], []).append((r["entity_type"], r["entity_key"]))
    return entities.compile_aliases(table)


def _target_driver(text: str, compiled, session_key: int | None) -> int | None:
    """The other car, when the line names exactly one driver.

    Two or more named drivers is not a target: "Norris will struggle to pass Antonelli" is
    about a pass between two other cars, and picking one of them would attach the event to
    an arbitrary half of it. Session-scoped, because the same surname is a different car
    number in a different season.
    """
    if session_key is None:
        return None
    drivers = {key for kind, key in entities.find_mentions(text, compiled) if kind == "driver"}
    if len(drivers) != 1:
        return None
    try:
        return int(next(iter(drivers)))
    except (TypeError, ValueError):
        return None


SPEECH_SQL = """
SELECT u.id AS speech_id, 'utterance' AS kind, u.text, u.embedding,
       f.session_key
  FROM utterances u
  JOIN clips c ON c.id = u.clip_id
  LEFT JOIN radio_calls f ON f.clip_id = u.clip_id
 WHERE c.status = 'COMPLETE' AND u.embedding IS NOT NULL
       {only_new}
UNION ALL
SELECT rc.id AS speech_id, 'radio_call' AS kind, rc.text, rc.embedding,
       rc.session_key
  FROM radio_calls rc
 WHERE rc.analyzed_at IS NOT NULL AND rc.error IS NULL AND rc.embedding IS NOT NULL
       {only_new_radio}
"""


async def extract(force: bool = False) -> dict:
    """Read every unread piece of speech, store what clears the thresholds.

    Returns counts rather than rows: this is an admin operation whose result is "what is
    now in the table", and the events themselves are read back through the graph or the
    clip pages.
    """
    cfg = await get_effective_settings()
    matrix, owners = _exemplars()
    compiled = await _alias_matcher()

    sql = SPEECH_SQL.format(
        only_new="" if force else
        "AND NOT EXISTS (SELECT 1 FROM speech_events e WHERE e.utterance_id = u.id)",
        only_new_radio="" if force else
        "AND NOT EXISTS (SELECT 1 FROM speech_events e WHERE e.radio_call_id = rc.id)",
    )
    rows = await db.fetch(sql)

    if force:
        await db.execute("DELETE FROM speech_events")

    by_type: dict[str, int] = {}
    read, declined = 0, 0
    for row in rows:
        read += 1
        if ontrack.too_short(row["text"], cfg.ontrack_min_words):
            declined += 1
            continue
        # asyncpg hands a pgvector column back as its text form; parsing here keeps the
        # pgvector round trip in one place rather than in every caller.
        raw = row["embedding"]
        vector = (np.fromstring(raw.strip("[]"), sep=",", dtype=np.float32)
                  if isinstance(raw, str) else np.asarray(raw, dtype=np.float32))
        if vector.size != matrix.shape[1]:
            declined += 1
            continue

        reading = ontrack.classify(
            score_one(vector, matrix, owners),
            floor=cfg.ontrack_min_similarity, min_margin=cfg.ontrack_min_margin)
        if reading is None:
            declined += 1
            continue

        target = _target_driver(row["text"] or "", compiled, row["session_key"])
        is_utterance = row["kind"] == "utterance"
        await db.execute(
            """INSERT INTO speech_events
                 (utterance_id, radio_call_id, event_type, family, score, margin,
                  target_driver_number, target_session_key, model)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
               ON CONFLICT DO NOTHING""",
            row["speech_id"] if is_utterance else None,
            None if is_utterance else row["speech_id"],
            reading.event_type, reading.family, reading.score, reading.margin,
            target, row["session_key"] if target is not None else None,
            cfg.text_embed_model)
        by_type[reading.event_type] = by_type.get(reading.event_type, 0) + 1

    return {
        "read": read, "events": sum(by_type.values()), "declined": declined,
        "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1])),
        "floor": cfg.ontrack_min_similarity, "margin": cfg.ontrack_min_margin,
    }


async def backfill_radio_embeddings(limit: int = 500) -> int:
    """Give analyzed radio calls the embedding the lightweight path never computed.

    Utterances get theirs from the INDEXING stage; radio calls skip it, which left most of
    the F1 corpus outside anything that compares vectors. Encoded here with the same model
    so both halves of the corpus stay in one space.
    """
    rows = await db.fetch(
        """SELECT id, text FROM radio_calls
            WHERE analyzed_at IS NOT NULL AND error IS NULL
              AND embedding IS NULL AND text IS NOT NULL AND length(text) > 0
            LIMIT $1""", limit)
    for row in rows:
        vector = text_embed.encode(row["text"])
        await db.execute("UPDATE radio_calls SET embedding=$2 WHERE id=$1",
                          row["id"], list(map(float, vector)))
    return len(rows)
