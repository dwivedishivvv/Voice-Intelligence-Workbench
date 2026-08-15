import asyncio

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from common import db, audit as audit_mod, speaker as sc
from common.config import get_settings, get_effective_settings
from ..auth import get_current_user
from ..services import montage as montage_svc

router = APIRouter(prefix="/v1/clusters", tags=["clusters"])


@router.get("")
async def list_clusters(user=Depends(get_current_user)):
    rows = await db.fetch(
        """SELECT id, label, n_members, total_speech_s, intra_cohesion, created_at
           FROM speaker_clusters WHERE promoted_to IS NULL ORDER BY total_speech_s DESC""")
    return {"items": [dict(r) for r in rows]}


@router.get("/{cluster_id}")
async def get_cluster(cluster_id: str, user=Depends(get_current_user)):
    row = await db.fetchrow("SELECT * FROM speaker_clusters WHERE id=$1", cluster_id)
    if not row:
        raise HTTPException(404, "cluster not found")
    members = await db.fetch(
        """SELECT clip_id, local_label, speech_s, reliability FROM clip_speakers
           WHERE cluster_id=$1 ORDER BY speech_s DESC""", cluster_id)
    d = dict(row)
    d.pop("centroid", None)
    segs = await montage_svc.select_segments(cluster_id)
    return {**d, "members": [dict(m) for m in members],
            "montage": montage_svc.summary(segs)}


@router.get("/{cluster_id}/montage")
async def cluster_montage(cluster_id: str, user=Depends(get_current_user)):
    """This cluster's own speech, stitched into one stream.

    Naming an unclaimed voice means recognising it, and recognising it means hearing it —
    otherwise the only way is opening every clip and hunting for the right timestamps.
    """
    cfg = get_settings()
    if not await db.fetchval("SELECT 1 FROM speaker_clusters WHERE id=$1", cluster_id):
        raise HTTPException(404, "cluster not found")

    segs = await montage_svc.select_segments(cluster_id)
    if not segs:
        raise HTTPException(404, "no usable speech for this cluster — its turns are all "
                                 "overlapped or too short to be worth listening to")
    try:
        path = await asyncio.to_thread(montage_svc.build, cfg, cluster_id, segs)
    except RuntimeError as e:
        raise HTTPException(500, f"could not build montage: {e}")

    s = montage_svc.summary(segs)
    # counts as headers so the player can label itself without a second round trip
    return FileResponse(path, media_type="audio/wav", headers={
        "X-Montage-Segments": str(s["segments"]),
        "X-Montage-Seconds": str(s["seconds"]),
        "X-Montage-Clips": str(len(s["clips"])),
        "Cache-Control": "private, max-age=300",
    })


class PromoteBody(BaseModel):
    display_name: str


@router.post("/{cluster_id}/promote")
async def promote(cluster_id: str, body: PromoteBody, user=Depends(get_current_user)):
    pid = await sc.promote_cluster(cluster_id, body.display_name, await get_effective_settings())
    await audit_mod.audit("cluster.promote", "speaker_cluster", cluster_id,
                           after={"profile_id": pid, "name": body.display_name}, actor=user)
    return {"profile_id": pid}
