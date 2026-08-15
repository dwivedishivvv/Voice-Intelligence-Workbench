"""Retrieval over transcripts.

Lifted out of routers/search.py so the graph-context endpoint can anchor on the same
ranking the search page uses, instead of growing a second, subtly different one. The
router is now a thin wrapper over `hybrid_search`.

Two entry points, deliberately distinct:

- `hybrid_search` — what `POST /v1/search` has always returned: utterances, ranked by
  FTS, vector similarity, or reciprocal-rank fusion of both.
- `anchor_speech` — the same thing widened to cover radio calls, for callers that want
  *any* unit of speech. Radio calls are the bulk of the F1 corpus and never reach the
  `utterances` table unless they are put through the full pipeline, so a graph query
  anchored on utterances alone would miss almost all of it.
"""
from common import db
from .. import text_embed

# Reciprocal-rank-fusion constant. Same 60 the hybrid SQL in hybrid_search() uses — the
# fusion here combines *its* output with the radio ranking, so a different damping between
# the two levels would weight one list against the other for no stated reason.
RRF_K = 60


async def hybrid_search(q: str, mode: str = "hybrid", speaker_id: str | None = None,
                        limit: int = 25) -> list[dict]:
    """Utterance search. `mode` is fts | semantic | hybrid (spec §9.4)."""
    args_tail = [speaker_id] if speaker_id else []

    if mode == "fts":
        spk_cond = "AND u.profile_id = $2" if speaker_id else ""
        rows = await db.fetch(
            f"""SELECT u.id AS utterance_id, u.clip_id, u.start_s, u.end_s, u.text,
                       ts_rank_cd(u.tsv, q) AS score
                FROM utterances u, websearch_to_tsquery('simple', $1) q
                WHERE u.tsv @@ q {spk_cond}
                ORDER BY score DESC LIMIT {limit}""",
            q, *args_tail)
        return [dict(r) for r in rows]

    qvec = text_embed.encode(q)
    if mode == "semantic":
        spk_cond = "AND u.profile_id = $2" if speaker_id else ""
        rows = await db.fetch(
            f"""SELECT u.id AS utterance_id, u.clip_id, u.start_s, u.end_s, u.text,
                       1 - (u.embedding <=> $1::vector) AS score
                FROM utterances u WHERE u.embedding IS NOT NULL {spk_cond}
                ORDER BY u.embedding <=> $1::vector LIMIT {limit}""",
            list(map(float, qvec)), *args_tail)
        return [dict(r) for r in rows]

    # hybrid: reciprocal rank fusion over FTS and vector rank (spec §9.4)
    # params here are [q, qvec, speaker_id?] — speaker filter is $3, not $2
    spk_cond = "AND u.profile_id = $3" if speaker_id else ""
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
        "SELECT id AS utterance_id, clip_id, start_s, end_s, text "
        "FROM utterances WHERE id = ANY($1)", ids)
    by_id = {str(u["utterance_id"]): dict(u) for u in u_rows}
    return [{**by_id[str(r["id"])], "score": r["score"]}
            for r in rows if str(r["id"]) in by_id]


async def _radio_search(q: str, limit: int) -> list[dict]:
    """Keyword search over analyzed radio calls.

    FTS only, no vector leg: radio calls have no embedding — the lightweight transcribe+
    tone path skips the INDEXING stage that produces one. Putting radio through the full
    pipeline (radio_calls.clip_id is the seam) would make them utterances and this function
    unnecessary."""
    rows = await db.fetch(
        """SELECT rc.id AS speech_id, rc.text, rc.mood, rc.recorded_at,
                  ts_rank_cd(rc.tsv, q) AS score
           FROM radio_calls rc, websearch_to_tsquery('simple', $1) q
           WHERE rc.tsv @@ q AND rc.analyzed_at IS NOT NULL AND rc.error IS NULL
           ORDER BY score DESC LIMIT $2""",
        q, limit)
    return [dict(r) for r in rows]


async def anchor_speech(q: str, limit: int = 10, mode: str = "hybrid",
                        speaker_id: str | None = None) -> list[dict]:
    """Top speech ids for a query, across both utterances and radio calls.

    The two rankings are not comparable — `ts_rank_cd` and a fused rank score live on
    different scales — so they are combined by *rank* rather than by score, the same
    reciprocal-rank fusion the hybrid SQL already uses to combine FTS with vectors. Fusing
    on raw scores would silently let whichever list happens to produce bigger numbers win.
    """
    utterances = await hybrid_search(q, mode=mode, speaker_id=speaker_id, limit=limit)
    # A speaker filter is a filter on enrolled utterances; radio calls carry a driver, not
    # a speaker profile, so scoping by speaker means excluding them rather than matching
    # none of them by accident.
    radio = [] if speaker_id else await _radio_search(q, limit)

    fused: dict[str, dict] = {}
    for ranked, kind, id_key in ((utterances, "utterance", "utterance_id"), (radio, "radio_call", "speech_id")):
        for rank, row in enumerate(ranked, start=1):
            sid = str(row[id_key])
            entry = fused.setdefault(sid, {"speech_id": sid, "kind": kind,
                                            "text": row.get("text"), "score": 0.0})
            entry["score"] += 1.0 / (RRF_K + rank)

    return sorted(fused.values(), key=lambda r: r["score"], reverse=True)[:limit]
