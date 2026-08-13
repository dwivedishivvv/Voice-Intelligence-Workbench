import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File

from common.config import get_effective_settings
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
