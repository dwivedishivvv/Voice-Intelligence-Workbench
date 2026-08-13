from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from common import db, audit as audit_mod, speaker as sc
from ..auth import get_current_user

router = APIRouter(prefix="/v1/speakers", tags=["speakers"])


@router.get("")
async def list_speakers(q: str | None = None, user=Depends(get_current_user)):
    where = "WHERE display_name ILIKE $1" if q else ""
    args = [f"%{q}%"] if q else []
    rows = await db.fetch(
        f"""SELECT id, display_name, status, n_enrollments, total_speech_s,
                   mean_reliability, intra_cohesion, created_at FROM speaker_profiles
            {where} ORDER BY display_name""", *args)
    return {"items": [dict(r) for r in rows]}


class CreateProfile(BaseModel):
    display_name: str
    notes: str | None = None


@router.post("")
async def create_profile(body: CreateProfile, user=Depends(get_current_user)):
    pid = await db.insert("speaker_profiles", {"display_name": body.display_name,
                                                "notes": body.notes, "status": "provisional"})
    await audit_mod.audit("speaker.create", "speaker_profile", pid, after=body.model_dump(), actor=user)
    return {"id": pid}


@router.get("/{profile_id}")
async def get_profile(profile_id: str, user=Depends(get_current_user)):
    row = await db.fetchrow("SELECT * FROM speaker_profiles WHERE id=$1", profile_id)
    if not row:
        raise HTTPException(404, "profile not found")
    enrollments = await db.fetch(
        """SELECT id, clip_id, duration_s, reliability, quality, source, is_outlier,
                  cos_to_centroid, created_at FROM speaker_enrollments
           WHERE profile_id=$1 ORDER BY created_at DESC""", profile_id)
    d = dict(row)
    d.pop("centroid", None)
    return {**d, "enrollments": [dict(e) for e in enrollments]}


class PatchProfile(BaseModel):
    display_name: str | None = None
    notes: str | None = None
    status: str | None = None


@router.patch("/{profile_id}")
async def patch_profile(profile_id: str, body: PatchProfile, user=Depends(get_current_user)):
    sets, params = [], []
    for field, val in body.model_dump(exclude_none=True).items():
        params.append(val)
        sets.append(f"{field}=${len(params)}")
    if sets:
        params.append(profile_id)
        await db.execute(f"UPDATE speaker_profiles SET {', '.join(sets)}, updated_at=now() WHERE id=${len(params)}", *params)
        await audit_mod.audit("speaker.rename", "speaker_profile", profile_id,
                               after=body.model_dump(exclude_none=True), actor=user)
    return {"ok": True}


@router.delete("/{profile_id}")
async def delete_profile(profile_id: str, reassign_to: str | None = None, user=Depends(get_current_user)):
    if reassign_to:
        await sc.merge_profiles(profile_id, reassign_to)
    else:
        await db.execute("UPDATE clip_speakers SET profile_id=NULL WHERE profile_id=$1", profile_id)
        await db.execute("UPDATE utterances SET profile_id=NULL WHERE profile_id=$1", profile_id)
        await db.execute("DELETE FROM speaker_profiles WHERE id=$1", profile_id)
    await audit_mod.audit("speaker.delete", "speaker_profile", profile_id, actor=user)
    return {"ok": True}


@router.get("/{profile_id}/clips")
async def profile_clips(profile_id: str, user=Depends(get_current_user)):
    rows = await db.fetch(
        """SELECT c.id, c.filename, c.duration_s, c.created_at, cs.talk_share, cs.match_result
           FROM clip_speakers cs JOIN clips c ON c.id = cs.clip_id
           WHERE cs.profile_id=$1 ORDER BY c.created_at DESC""", profile_id)
    return {"items": [dict(r) for r in rows]}


class MergeBody(BaseModel):
    source_id: str


@router.post("/{profile_id}/merge")
async def merge(profile_id: str, body: MergeBody, user=Depends(get_current_user)):
    await sc.merge_profiles(body.source_id, profile_id)
    await audit_mod.audit("speaker.merge", "speaker_profile", profile_id,
                           after={"merged_from": body.source_id}, actor=user)
    return {"ok": True}


@router.delete("/{profile_id}/enrollments/{eid}")
async def delete_enrollment(profile_id: str, eid: str, user=Depends(get_current_user)):
    await db.execute("DELETE FROM speaker_enrollments WHERE id=$1 AND profile_id=$2", eid, profile_id)
    await sc.recompute_centroid(profile_id)
    await audit_mod.audit("enrollment.remove", "speaker_profile", profile_id, after={"enrollment_id": eid}, actor=user)
    return {"ok": True}
