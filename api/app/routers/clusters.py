from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from common import db, audit as audit_mod, speaker as sc
from common.config import get_effective_settings
from ..auth import get_current_user

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
    return {**d, "members": [dict(m) for m in members]}


class PromoteBody(BaseModel):
    display_name: str


@router.post("/{cluster_id}/promote")
async def promote(cluster_id: str, body: PromoteBody, user=Depends(get_current_user)):
    pid = await sc.promote_cluster(cluster_id, body.display_name, await get_effective_settings())
    await audit_mod.audit("cluster.promote", "speaker_cluster", cluster_id,
                           after={"profile_id": pid, "name": body.display_name}, actor=user)
    return {"profile_id": pid}
