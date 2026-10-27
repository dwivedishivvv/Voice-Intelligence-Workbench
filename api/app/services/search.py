"""Retrieval over transcripts.

Lifted out of routers/search.py so the graph-context endpoint and the agent can anchor on
the same ranking the search page uses, instead of growing a second, subtly different one.
The router is now a thin wrapper over `hybrid_search`.

Two entry points:

- `hybrid_search` — what `POST /v1/search` returns: utterances ranked by FTS, vector
  similarity, or reciprocal-rank fusion of both.
- `anchor_speech` — the same ranking keyed by `speech_id`, the id every graph and agent
  caller cites.

Every filter is optional and composes with every other, because the agent reaches for
these in combinations nobody enumerated in advance.
"""
from common import db
from .. import text_embed

# Reciprocal-rank-fusion constant.
RRF_K = 60


def _utterance_filters(first_param: int, speaker_id: str | None,
                       tags: list[str] | None = None,
                       clip_id: str | None = None) -> tuple[str, list]:
    """SQL fragment and args for the optional utterance filters.

    Numbered from `first_param` because each search mode has a different number of fixed
    parameters ahead of them — the placeholder indices used to be written by hand at each
    call site, which is fine for one optional filter and a trap for two.

    Each of `tags` must be on the utterance's clip, as an operator tag or an auto tag.
    """
    conds, args = [], []
    for column, value in (("profile_id", speaker_id), ("clip_id", clip_id)):
        if value:
            args.append(value)
            conds.append(f"AND u.{column} = ${first_param + len(args) - 1}")
    for tag in tags or ():
        args.append(tag)
        p = f"${first_param + len(args) - 1}"
        conds.append(f"AND u.clip_id IN (SELECT id FROM clips WHERE {p} = ANY(tags) "
                     f"OR {p} = ANY(auto_tags))")
    return " ".join(conds), args


async def hybrid_search(q: str, mode: str = "hybrid", speaker_id: str | None = None,
                        limit: int = 25, tags: list[str] | None = None,
                        clip_id: str | None = None) -> list[dict]:
    """Utterance search. `mode` is fts | semantic | hybrid (spec §9.4).

    `speaker_id` narrows to one enrolled speaker, `clip_id` to one recording, and `tags` to
    clips carrying every one of them, operator-set or automatic.
    """
    if mode == "fts":
        cond, args_tail = _utterance_filters(2, speaker_id, tags, clip_id)
        rows = await db.fetch(
            f"""SELECT u.id AS utterance_id, u.clip_id, u.start_s, u.end_s, u.text,
                       ts_rank_cd(u.tsv, q) AS score
                FROM utterances u, websearch_to_tsquery('simple', $1) q
                WHERE u.tsv @@ q {cond}
                ORDER BY score DESC LIMIT {limit}""",
            q, *args_tail)
        return [dict(r) for r in rows]

    qvec = text_embed.encode(q)
    if mode == "semantic":
        cond, args_tail = _utterance_filters(2, speaker_id, tags, clip_id)
        rows = await db.fetch(
            f"""SELECT u.id AS utterance_id, u.clip_id, u.start_s, u.end_s, u.text,
                       1 - (u.embedding <=> $1::vector) AS score
                FROM utterances u WHERE u.embedding IS NOT NULL {cond}
                ORDER BY u.embedding <=> $1::vector LIMIT {limit}""",
            list(map(float, qvec)), *args_tail)
        return [dict(r) for r in rows]

    # hybrid: reciprocal rank fusion over FTS and vector rank (spec §9.4)
    # params here are [q, qvec, then whichever filters are set]
    cond, args_tail = _utterance_filters(3, speaker_id, tags, clip_id)
    rows = await db.fetch(
        f"""WITH fts AS (
              SELECT u.id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(u.tsv, q) DESC) rk
              FROM utterances u, websearch_to_tsquery('simple', $1) q
              WHERE u.tsv @@ q {cond} LIMIT 200
            ),
            vec AS (
              SELECT u.id, ROW_NUMBER() OVER (ORDER BY u.embedding <=> $2::vector) rk
              FROM utterances u WHERE u.embedding IS NOT NULL {cond} LIMIT 200
            )
            SELECT COALESCE(f.id, v.id) AS id,
                   COALESCE(1.0/({RRF_K}+f.rk), 0) + COALESCE(1.0/({RRF_K}+v.rk), 0) AS score
            FROM fts f FULL OUTER JOIN vec v ON f.id = v.id
            ORDER BY score DESC LIMIT {limit}""",
        q, list(map(float, qvec)), *args_tail)
    ids = [r["id"] for r in rows]
    if not ids:
        return []
    # Aliased to utterance_id so all three modes return the same key.
    u_rows = await db.fetch(
        "SELECT id AS utterance_id, clip_id, start_s, end_s, text "
        "FROM utterances WHERE id = ANY($1)", ids)
    by_id = {str(u["utterance_id"]): dict(u) for u in u_rows}
    return [{**by_id[str(r["id"])], "score": r["score"]}
            for r in rows if str(r["id"]) in by_id]


async def anchor_speech(q: str, limit: int = 10, mode: str = "hybrid",
                        speaker_id: str | None = None,
                        clip_id: str | None = None) -> list[dict]:
    """Top speech ids for a query, in the shape graph and agent callers expect."""
    rows = await hybrid_search(q, mode=mode, speaker_id=speaker_id, limit=limit,
                               clip_id=clip_id)
    return [{"speech_id": str(r["utterance_id"]), "kind": "utterance",
             "text": r.get("text"), "score": float(r["score"] or 0)} for r in rows]
