"""Retrieval over transcripts.

Lifted out of routers/search.py so the graph-context endpoint can anchor on the same
ranking the search page uses, instead of growing a second, subtly different one. The
router is now a thin wrapper over `hybrid_search`.

Three entry points, deliberately distinct:

- `hybrid_search` — what `POST /v1/search` has always returned: utterances, ranked by
  FTS, vector similarity, or reciprocal-rank fusion of both.
- `anchor_speech` — the same thing widened to cover radio calls, for callers that want
  *any* unit of speech. Radio calls are the bulk of the F1 corpus and never reach the
  `utterances` table unless they are put through the full pipeline, so a graph query
  anchored on utterances alone would miss almost all of it.
- `speech_by_mood` — speech selected by its tone reading with no topic at all. Tone is a
  column, not words: "which calls sound stressed" matches no text and is unanswerable by
  either of the above, which is exactly how it failed before this existed.

Every filter is optional and composes with every other, because the agent reaches for
these in combinations nobody enumerated in advance.
"""
from itertools import zip_longest

from common import db
from .. import text_embed

# Reciprocal-rank-fusion constant. Same 60 the hybrid SQL in hybrid_search() uses — the
# fusion here combines *its* output with the radio ranking, so a different damping between
# the two levels would weight one list against the other for no stated reason.
RRF_K = 60

# What worker/worker/audio/tone.py can emit. A filter value outside this set matches
# nothing, and "nothing found" reads as a fact about the corpus rather than a bad argument,
# so callers validate against this list instead of passing the string straight through.
MOODS = ("calm", "stressed", "tired")


def _utterance_filters(first_param: int, speaker_id: str | None,
                       mood: str | None) -> tuple[str, list]:
    """SQL fragment and args for the optional utterance filters.

    Numbered from `first_param` because each search mode has a different number of fixed
    parameters ahead of them — the placeholder indices used to be written by hand at each
    call site, which is fine for one optional filter and a trap for two.
    """
    conds, args = [], []
    for column, value in (("profile_id", speaker_id), ("mood", mood)):
        if value:
            args.append(value)
            conds.append(f"AND u.{column} = ${first_param + len(args) - 1}")
    return " ".join(conds), args


async def hybrid_search(q: str, mode: str = "hybrid", speaker_id: str | None = None,
                        limit: int = 25, mood: str | None = None) -> list[dict]:
    """Utterance search. `mode` is fts | semantic | hybrid (spec §9.4).

    `mood` narrows to one tone reading, and composes with `speaker_id` — "what did this
    person say when their voice read stressed" is one query, not a client-side join.
    """
    if mode == "fts":
        spk_cond, args_tail = _utterance_filters(2, speaker_id, mood)
        rows = await db.fetch(
            f"""SELECT u.id AS utterance_id, u.clip_id, u.start_s, u.end_s, u.text, u.mood,
                       ts_rank_cd(u.tsv, q) AS score
                FROM utterances u, websearch_to_tsquery('simple', $1) q
                WHERE u.tsv @@ q {spk_cond}
                ORDER BY score DESC LIMIT {limit}""",
            q, *args_tail)
        return [dict(r) for r in rows]

    qvec = text_embed.encode(q)
    if mode == "semantic":
        spk_cond, args_tail = _utterance_filters(2, speaker_id, mood)
        rows = await db.fetch(
            f"""SELECT u.id AS utterance_id, u.clip_id, u.start_s, u.end_s, u.text, u.mood,
                       1 - (u.embedding <=> $1::vector) AS score
                FROM utterances u WHERE u.embedding IS NOT NULL {spk_cond}
                ORDER BY u.embedding <=> $1::vector LIMIT {limit}""",
            list(map(float, qvec)), *args_tail)
        return [dict(r) for r in rows]

    # hybrid: reciprocal rank fusion over FTS and vector rank (spec §9.4)
    # params here are [q, qvec, then whichever filters are set]
    spk_cond, args_tail = _utterance_filters(3, speaker_id, mood)
    rows = await db.fetch(
        f"""WITH fts AS (
              SELECT u.id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(u.tsv, q) DESC) rk
              FROM utterances u, websearch_to_tsquery('simple', $1) q
              WHERE u.tsv @@ q {spk_cond} LIMIT 200
            ),
            vec AS (
              SELECT u.id, ROW_NUMBER() OVER (ORDER BY u.embedding <=> $2::vector) rk
              FROM utterances u WHERE u.embedding IS NOT NULL {spk_cond} LIMIT 200
            )
            SELECT COALESCE(f.id, v.id) AS id,
                   COALESCE(1.0/({RRF_K}+f.rk), 0) + COALESCE(1.0/({RRF_K}+v.rk), 0) AS score
            FROM fts f FULL OUTER JOIN vec v ON f.id = v.id
            ORDER BY score DESC LIMIT {limit}""",
        q, list(map(float, qvec)), *args_tail)
    ids = [r["id"] for r in rows]
    if not ids:
        return []
    # Aliased to utterance_id so all three modes return the same key. They did not before
    # this was extracted: fts and semantic returned `utterance_id` while hybrid returned
    # `id`, so any caller that switched modes silently got a KeyError or a missing field.
    # Safe to normalise — no UI consumes /v1/search yet.
    u_rows = await db.fetch(
        "SELECT id AS utterance_id, clip_id, start_s, end_s, text, mood "
        "FROM utterances WHERE id = ANY($1)", ids)
    by_id = {str(u["utterance_id"]): dict(u) for u in u_rows}
    return [{**by_id[str(r["id"])], "score": r["score"]}
            for r in rows if str(r["id"]) in by_id]


async def _radio_search(q: str, limit: int, mood: str | None = None) -> list[dict]:
    """Keyword search over analyzed radio calls.

    FTS only, no vector leg: radio calls have no embedding — the lightweight transcribe+
    tone path skips the INDEXING stage that produces one. Putting radio through the full
    pipeline (radio_calls.clip_id is the seam) would make them utterances and this function
    unnecessary."""
    mood_cond = "AND rc.mood = $3" if mood else ""
    rows = await db.fetch(
        f"""SELECT rc.id AS speech_id, rc.text, rc.mood, rc.recorded_at,
                   ts_rank_cd(rc.tsv, q) AS score
            FROM radio_calls rc, websearch_to_tsquery('simple', $1) q
            WHERE rc.tsv @@ q AND rc.analyzed_at IS NOT NULL AND rc.error IS NULL {mood_cond}
            ORDER BY score DESC LIMIT $2""",
        q, limit, *([mood] if mood else []))
    return [dict(r) for r in rows]


async def speech_by_mood(mood: str, limit: int = 10,
                         speaker_id: str | None = None) -> list[dict]:
    """Speech carrying one tone reading, with no topic to rank against.

    Tone lives in a column, so "which calls sound stressed" matches no text and cannot be
    served by ranking: there is nothing to rank. Ordering is therefore recency, newest
    first, and the caller is expected to say so rather than imply these are the *most*
    stressed — the classifier emits a label, not a magnitude.

    Utterances and radio calls are interleaved rather than concatenated. Taking the top
    `limit` of a concatenation would return one source only whenever that source has
    enough rows, which silently hides the other half of the corpus.
    """
    utterances = await db.fetch(
        f"""SELECT u.id AS utterance_id, u.clip_id, u.start_s, u.end_s, u.text, u.mood
              FROM utterances u JOIN clips c ON c.id = u.clip_id
             WHERE u.mood = $1 AND c.status = 'COMPLETE'
                   {"AND u.profile_id = $3" if speaker_id else ""}
             ORDER BY c.created_at DESC, u.idx LIMIT $2""",
        mood, limit, *([speaker_id] if speaker_id else []))
    # Radio calls carry a driver, not a speaker profile, so a speaker filter excludes them
    # rather than matching none of them by accident — same rule as anchor_speech.
    radio = [] if speaker_id else await db.fetch(
        """SELECT rc.id AS speech_id, rc.text, rc.mood
             FROM radio_calls rc
            WHERE rc.mood = $1 AND rc.analyzed_at IS NOT NULL AND rc.error IS NULL
            ORDER BY rc.recorded_at DESC NULLS LAST LIMIT $2""",
        mood, limit)

    out: list[dict] = []
    for u_row, r_row in zip_longest([dict(r) for r in utterances], [dict(r) for r in radio]):
        for row, kind, id_key in ((u_row, "utterance", "utterance_id"),
                                  (r_row, "radio_call", "speech_id")):
            if row:
                out.append({"speech_id": str(row[id_key]), "kind": kind,
                            "text": row.get("text"), "mood": row.get("mood"), "score": 0.0})
    return out[:limit]


async def anchor_speech(q: str, limit: int = 10, mode: str = "hybrid",
                        speaker_id: str | None = None,
                        mood: str | None = None) -> list[dict]:
    """Top speech ids for a query, across both utterances and radio calls.

    The two rankings are not comparable — `ts_rank_cd` and a fused rank score live on
    different scales — so they are combined by *rank* rather than by score, the same
    reciprocal-rank fusion the hybrid SQL already uses to combine FTS with vectors. Fusing
    on raw scores would silently let whichever list happens to produce bigger numbers win.
    """
    utterances = await hybrid_search(q, mode=mode, speaker_id=speaker_id, limit=limit, mood=mood)
    # A speaker filter is a filter on enrolled utterances; radio calls carry a driver, not
    # a speaker profile, so scoping by speaker means excluding them rather than matching
    # none of them by accident.
    radio = [] if speaker_id else await _radio_search(q, limit, mood=mood)

    fused: dict[str, dict] = {}
    for ranked, kind, id_key in ((utterances, "utterance", "utterance_id"), (radio, "radio_call", "speech_id")):
        for rank, row in enumerate(ranked, start=1):
            sid = str(row[id_key])
            entry = fused.setdefault(sid, {"speech_id": sid, "kind": kind,
                                            "text": row.get("text"),
                                            "mood": row.get("mood"), "score": 0.0})
            entry["score"] += 1.0 / (RRF_K + rank)

    return sorted(fused.values(), key=lambda r: r["score"], reverse=True)[:limit]
