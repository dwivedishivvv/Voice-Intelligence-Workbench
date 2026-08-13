import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from common.config import get_effective_settings
from ..auth import get_current_user
from ..services import queue

router = APIRouter(prefix="/v1/f1", tags=["f1"])

OPENF1 = "https://api.openf1.org/v1"


async def _proxy(path: str, params: dict):
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{OPENF1}/{path}", params={k: v for k, v in params.items() if v is not None})
        r.raise_for_status()
        return r.json()


@router.get("/sessions")
async def sessions(year: int | None = None, session_type: str = "Race", user=Depends(get_current_user)):
    return await _proxy("sessions", {"year": year, "session_type": session_type})


@router.get("/drivers")
async def drivers(session_key: int, user=Depends(get_current_user)):
    return await _proxy("drivers", {"session_key": session_key})


@router.get("/laps")
async def laps(session_key: int, driver_number: int, user=Depends(get_current_user)):
    return await _proxy("laps", {"session_key": session_key, "driver_number": driver_number})


@router.get("/team_radio")
async def team_radio(session_key: int, driver_number: int | None = None, user=Depends(get_current_user)):
    return await _proxy("team_radio", {"session_key": session_key, "driver_number": driver_number})


class IngestBody(BaseModel):
    recording_url: str
    # optional — when given, tone analysis calibrates against this driver's other radio
    # calls in the same session instead of a fixed global pitch/rate threshold
    session_key: int | None = None
    driver_number: int | None = None


@router.post("/ingest")
async def ingest(body: IngestBody, user=Depends(get_current_user)):
    cfg = await get_effective_settings()
    if not body.recording_url.startswith("https://livetiming.formula1.com/"):
        raise HTTPException(400, "recording_url must be an official F1 livetiming asset")

    job_id = uuid.uuid4().hex
    rel_path = f"tmp/f1/{job_id}.mp3"
    abs_path = Path(cfg.data_dir) / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(body.recording_url)
        r.raise_for_status()
        abs_path.write_bytes(r.content)

    await queue.enqueue_f1_radio(job_id, rel_path, body.session_key, body.driver_number)
    return {"job_id": job_id, "ws_url": f"/v1/ws/jobs/{job_id}"}
