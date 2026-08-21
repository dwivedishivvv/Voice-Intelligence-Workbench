import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel

from common.config import get_effective_settings
from common import db, speaker as sc, audit as audit_mod
from ..auth import get_current_user
from ..services import queue

router = APIRouter(prefix="/v1/live", tags=["live"])


@router.post("/{session_id}/chunk")
async def upload_chunk(session_id: str, seq: int, file: UploadFile = File(...),
                        user=Depends(get_current_user)):
    cfg = await get_effective_settings()
    rel_path = f"tmp/live/{session_id}/{seq}-{uuid.uuid4().hex[:8]}.webm"
    abs_path = Path(cfg.data_dir) / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(abs_path, "wb") as out:
        out.write(await file.read())

    await queue.enqueue_live_chunk(session_id, rel_path, seq)
    return {"queued": True, "ws_url": f"/v1/ws/jobs/{session_id}"}


class TagBody(BaseModel):
    display_name: str
    embedding: list[float]
    duration_s: float
    reliability: float


@router.post("/tag")
async def tag_speaker(body: TagBody, user=Depends(get_current_user)):
    """Name an unmatched (or only-suggested) live turn. The audio behind a live chunk is
    deleted right after it's processed (see worker/main.py), so there's no clip to enroll
    from later the way the clip-detail page does it — the embedding the worker already
    computed for this turn is enrolled directly, with no clip_id, which speaker_enrollments
    allows for exactly this reason. Once enrolled, every later chunk's identify() call sees
    this voice and can match it, live-mic or F1-radio session alike.
    """
    if len(body.embedding) != 192:
        raise HTTPException(400, "embedding must be 192-dimensional")
    name = body.display_name.strip()
    if not name:
        raise HTTPException(400, "display_name is required")

    row = await db.fetchrow(
        "SELECT id FROM speaker_profiles WHERE lower(display_name) = lower($1)", name)
    profile_id = str(row["id"]) if row else await db.insert(
        "speaker_profiles", {"display_name": name, "status": "provisional"})

    await sc.add_enrollment(
        profile_id,
        {"embedding": body.embedding, "reliability": body.reliability,
         "speech_s": body.duration_s, "local_label": None},
        clip_id=None, source="manual", actor=user)
    await audit_mod.audit("speaker.live_tag", "speaker_profile", profile_id,
                           after={"display_name": name}, actor=user)
    return {"profile_id": profile_id, "display_name": name}
