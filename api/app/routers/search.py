from fastapi import APIRouter, Depends
from pydantic import BaseModel

from common import db
from ..auth import get_current_user
from .. import text_embed

router = APIRouter(prefix="/v1/search", tags=["search"])


class SearchBody(BaseModel):
    q: str
    mode: str = "hybrid"  # fts | semantic | hybrid
    speaker_id: str | None = None
    limit: int = 25


@router.post("")
async def search(body: SearchBody, user=Depends(get_current_user)):
    args_tail = [body.speaker_id] if body.speaker_id else []

    if body.mode == "fts":
        spk_cond = "AND u.profile_id = $2" if body.speaker_id else ""
        rows = await db.fetch(
            f"""SELECT u.id AS utterance_id, u.clip_id, u.start_s, u.end_s, u.text,
                       ts_rank_cd(u.tsv, q) AS score
                FROM utterances u, websearch_to_tsquery('simple', $1) q
                WHERE u.tsv @@ q {spk_cond}
                ORDER BY score DESC LIMIT {body.limit}""",
            body.q, *args_tail)
        return {"items": [dict(r) for r in rows]}

    qvec = text_embed.encode(body.q)
    if body.mode == "semantic":
        spk_cond = "AND u.profile_id = $2" if body.speaker_id else ""
        rows = await db.fetch(
            f"""SELECT u.id AS utterance_id, u.clip_id, u.start_s, u.end_s, u.text,
                       1 - (u.embedding <=> $1::vector) AS score
                FROM utterances u WHERE u.embedding IS NOT NULL {spk_cond}
                ORDER BY u.embedding <=> $1::vector LIMIT {body.limit}""",
            list(map(float, qvec)), *args_tail)
        return {"items": [dict(r) for r in rows]}

    # hybrid: reciprocal rank fusion over FTS and vector rank (spec §9.4)
    # params here are [q, qvec, speaker_id?] — speaker filter is $3, not $2
    spk_cond = "AND u.profile_id = $3" if body.speaker_id else ""
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
                   COALESCE(1.0/(60+f.rk), 0) + COALESCE(1.0/(60+v.rk), 0) AS score
            FROM fts f FULL OUTER JOIN vec v ON f.id = v.id
            ORDER BY score DESC LIMIT {body.limit}""",
        body.q, list(map(float, qvec)), *args_tail)
    ids = [r["id"] for r in rows]
    if not ids:
        return {"items": []}
    u_rows = await db.fetch(
        "SELECT id, clip_id, start_s, end_s, text FROM utterances WHERE id = ANY($1)", ids)
    by_id = {str(u["id"]): dict(u) for u in u_rows}
    return {"items": [{**by_id[str(r["id"])], "score": r["score"]} for r in rows if str(r["id"]) in by_id]}
