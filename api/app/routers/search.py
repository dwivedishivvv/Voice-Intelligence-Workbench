import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from common import db, speaker as sc
from common.config import get_effective_settings
from ..auth import get_current_user
from ..services import queue, search as search_svc

router = APIRouter(prefix="/v1/search", tags=["search"])


class SearchBody(BaseModel):
    q: str
    mode: str = "hybrid"  # fts | semantic | hybrid
    speaker_id: str | None = None
    clip_id: str | None = None
    tags: list[str] = []
    limit: int = 25


@router.post("")
async def search(body: SearchBody, user=Depends(get_current_user)):
    # Ranking lives in services/search.py so the graph-context endpoint anchors on exactly
    # this, rather than growing a second implementation that drifts from it.
    items = await search_svc.hybrid_search(body.q, body.mode, body.speaker_id,
                                           min(max(body.limit, 1), 100),
                                           tags=body.tags, clip_id=body.clip_id)
    return {"items": await _with_filenames(items)}


async def _with_filenames(items: list[dict]) -> list[dict]:
    # The clip each hit belongs to, so a result list can say where it came from without a
    # request per row.
    clip_ids = list({str(i["clip_id"]) for i in items})
    clips = {str(r["id"]): r for r in await db.fetch(
        "SELECT id, filename FROM clips WHERE id = ANY($1::uuid[])", clip_ids)} if clip_ids else {}
    for i in items:
        i["filename"] = clips[str(i["clip_id"])]["filename"] if str(i["clip_id"]) in clips else None
    return items


VOICE_QUERY_MAX_MB = 20


async def voices_like(embedding: list[float], cfg, limit: int) -> list[dict]:
    """Clip voices nearest a query voiceprint, each with where in the clip that voice first
    speaks, and the same bands identification uses: "likely" at or above id_threshold,
    "possible" within the suggestion band below it. Under that, a result is only the
    nearest of a field of strangers and is labelled as such rather than hidden — a search
    shows what is closest even when nothing is close."""
    rows = await db.fetch(
        """SELECT cs.clip_id, cs.local_label, cs.speech_s, cs.reliability, cs.match_result,
                  p.display_name, c.filename,
                  1 - (cs.embedding <=> $1::vector) AS sim,
                  (SELECT min(t.start_s) FROM speaker_turns t
                    WHERE t.clip_id = cs.clip_id AND t.local_label = cs.local_label) AS first_s
             FROM clip_speakers cs
             JOIN clips c ON c.id = cs.clip_id
             LEFT JOIN speaker_profiles p ON p.id = cs.profile_id
            WHERE cs.embedding IS NOT NULL AND c.status = 'COMPLETE'
            ORDER BY cs.embedding <=> $1::vector LIMIT $2""", embedding, limit)

    def band(sim: float) -> str:
        if sim >= cfg.id_threshold:
            return "likely"
        return "possible" if sim >= cfg.id_threshold - cfg.id_suggest_delta else "unlikely"

    return [{**dict(r), "clip_id": str(r["clip_id"]), "band": band(float(r["sim"]))} for r in rows]


@router.post("/voice")
async def voice_search(file: UploadFile = File(...), mode: str = Form("both"),
                       limit: int = Form(20), user=Depends(get_current_user)):
    """Search by speaking: `speech` finds where the same words were said, `speaker` finds
    the same voice across recordings and says who it is if anyone enrolled matches, `both`
    does both from the one recording.

    The audio goes to the worker (the api holds no models), which transcribes it and takes
    a voiceprint from its speech regions; the searches themselves run here, over the same
    indexes the text search and identification use.
    """
    if mode not in ("both", "speech", "speaker"):
        raise HTTPException(400, "mode must be both, speech or speaker")
    cfg = await get_effective_settings()
    rel = f"tmp/voice/{uuid.uuid4().hex}{Path(file.filename or 'q.webm').suffix or '.webm'}"
    dst = Path(cfg.data_dir) / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = await file.read()
    if len(data) > VOICE_QUERY_MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"a voice query is capped at {VOICE_QUERY_MAX_MB}MB")
    dst.write_bytes(data)
    try:
        q = await queue.voice_query(rel)
    except TimeoutError as e:
        raise HTTPException(504, "the worker did not get to this query in time — it may be "
                                 "busy with a long clip; try again shortly") from e

    out: dict = {"heard": q["text"], "language": q["language"],
                 "duration_s": q["duration_s"], "speech_s": q["speech_s"],
                 "reliability": q["reliability"]}
    limit = min(max(limit, 1), 50)

    if mode in ("both", "speech"):
        out["speech"] = (await _with_filenames(
            await search_svc.hybrid_search(q["text"], "hybrid", limit=limit))) if q["text"] else []

    if mode in ("both", "speaker"):
        if q["embedding"] is None:
            # too little speech to fingerprint: say so instead of ranking noise
            out["speaker"] = {"matches": [], "who": None,
                              "note": "not enough speech in the query to take a voiceprint"}
        else:
            m = await sc.identify(np.asarray(q["embedding"], dtype=np.float32), q["reliability"], cfg)
            who = None
            if m.result in ("confident", "suggested") and m.profile_id:
                name = await db.fetchval(
                    "SELECT display_name FROM speaker_profiles WHERE id=$1", m.profile_id)
                who = {"profile_id": str(m.profile_id), "name": name, "result": m.result,
                       "score": m.score, "margin": m.margin}
            out["speaker"] = {"who": who, "result": m.result,
                              "matches": await voices_like(q["embedding"], cfg, limit)}
    return out
