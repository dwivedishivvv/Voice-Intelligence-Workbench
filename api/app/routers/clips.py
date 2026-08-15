import json
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from common.config import get_settings, get_effective_settings
from common import db, storage, audit as audit_mod
from ..auth import get_current_user
from ..services import queue, ingest

router = APIRouter(prefix="/v1/clips", tags=["clips"])


@router.post("")
async def upload_clip(file: UploadFile = File(...), tags: str = "", notes: str = "",
                       user=Depends(get_current_user)):
    cfg = await get_effective_settings()
    return await ingest.ingest_clip(file, cfg, user,
                                     tags=tags.split(",") if tags else [], notes=notes)


@router.get("")
async def list_clips(q: str | None = None, status: str | None = None,
                      needs_review: bool | None = None, limit: int = 25, offset: int = 0,
                      user=Depends(get_current_user)):
    conds, params = [], []
    if status:
        params.append(status); conds.append(f"status=${len(params)}")
    if needs_review is not None:
        params.append(needs_review); conds.append(f"needs_review=${len(params)}")
    if q:
        params.append(f"%{q}%"); conds.append(f"filename ILIKE ${len(params)}")
    where = f"WHERE {' AND '.join(conds)}" if conds else ""
    params += [limit, offset]
    rows = await db.fetch(
        f"""SELECT id, filename, duration_s, status, language, n_speakers, needs_review,
                   created_at, error_code
            FROM clips {where} ORDER BY created_at DESC LIMIT ${len(params)-1} OFFSET ${len(params)}""",
        *params)
    total = await db.fetchval(f"SELECT count(*) FROM clips {where}", *params[:-2])
    return {"items": [dict(r) for r in rows], "total": total}


@router.get("/{clip_id}")
async def get_clip(clip_id: str, user=Depends(get_current_user)):
    row = await db.fetchrow("SELECT * FROM clips WHERE id=$1", clip_id)
    if not row:
        raise HTTPException(404, "clip not found")
    return dict(row)


@router.get("/{clip_id}/result")
async def get_result(clip_id: str, user=Depends(get_current_user)):
    clip = await db.fetchrow("SELECT * FROM clips WHERE id=$1", clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    quality = await db.fetchrow("SELECT * FROM quality_metrics WHERE clip_id=$1", clip_id)
    vad = await db.fetch("SELECT start_s, end_s FROM vad_regions WHERE clip_id=$1 ORDER BY idx", clip_id)
    turns = await db.fetch("SELECT * FROM speaker_turns WHERE clip_id=$1 ORDER BY idx", clip_id)
    transcript = await db.fetchrow(
        "SELECT * FROM transcripts WHERE clip_id=$1 ORDER BY created_at DESC LIMIT 1", clip_id)
    utterances, words = [], []
    if transcript:
        utterances = [dict(u) for u in await db.fetch(
            "SELECT * FROM utterances WHERE transcript_id=$1 ORDER BY idx", transcript["id"])]
        words = [dict(w) for w in await db.fetch(
            "SELECT * FROM words WHERE clip_id=$1 ORDER BY idx", clip_id)]
    speakers = await db.fetch(
        """SELECT cs.*, sp.display_name FROM clip_speakers cs
           LEFT JOIN speaker_profiles sp ON sp.id = cs.profile_id
           WHERE cs.clip_id=$1""", clip_id)
    run = await db.fetchrow(
        "SELECT * FROM processing_runs WHERE clip_id=$1 ORDER BY started_at DESC LIMIT 1", clip_id)

    def clean(r):
        d = dict(r)
        d.pop("embedding", None)
        return d

    return {
        "clip": dict(clip), "quality": dict(quality) if quality else None,
        "vad": [dict(v) for v in vad], "turns": [dict(t) for t in turns],
        "transcript": dict(transcript) if transcript else None,
        "utterances": [clean(u) for u in utterances], "words": words,
        "speakers": [clean(s) for s in speakers],
        "run": {
            **dict(run),
            "config_snapshot": json.loads(run["config_snapshot"]) if run["config_snapshot"] else None,
            "warnings": json.loads(run["warnings"]) if run["warnings"] else [],
        } if run else None,
    }


@router.get("/{clip_id}/audio")
async def get_audio(clip_id: str, variant: str = "original", user=Depends(get_current_user)):
    cfg = get_settings()
    clip = await db.fetchrow("SELECT raw_path, work_path FROM clips WHERE id=$1", clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    rel_path = clip["work_path"] if variant == "work" and clip["work_path"] else clip["raw_path"]
    if not rel_path:
        raise HTTPException(404, "audio not available")
    path = storage.resolve(cfg, rel_path)
    if not os.path.exists(path):
        raise HTTPException(404, "audio not available")
    return FileResponse(path)


EXPORT_FILENAMES = {"srt": "transcript.srt", "vtt": "transcript.vtt", "txt": "transcript.txt",
                     "rttm": "diarization.rttm", "json": "full.json"}


@router.get("/{clip_id}/export/{fmt}")
async def export_clip(clip_id: str, fmt: str, user=Depends(get_current_user)):
    cfg = get_settings()
    if fmt not in EXPORT_FILENAMES:
        raise HTTPException(404, f"unknown export format: {fmt}")
    path = storage.export_path(cfg, clip_id, EXPORT_FILENAMES[fmt])
    if not os.path.exists(path):
        raise HTTPException(404, f"{fmt} export not generated yet")
    return FileResponse(path)


@router.post("/{clip_id}/reprocess")
async def reprocess_clip(clip_id: str, user=Depends(get_current_user)):
    row = await db.fetchrow("SELECT id FROM clips WHERE id=$1", clip_id)
    if not row:
        raise HTTPException(404, "clip not found")
    await db.execute("UPDATE clips SET status='QUEUED', error_code=NULL, error_detail=NULL WHERE id=$1", clip_id)
    job_id = await queue.enqueue_clip(clip_id)
    await audit_mod.audit("clip.reprocess", "clip", clip_id, actor=user)
    return {"job_id": job_id, "status": "QUEUED"}


class PatchClip(BaseModel):
    tags: list[str] | None = None
    notes: str | None = None
    needs_review: bool | None = None


@router.patch("/{clip_id}")
async def patch_clip(clip_id: str, body: PatchClip, user=Depends(get_current_user)):
    sets, params = [], []
    for field, val in body.model_dump(exclude_none=True).items():
        params.append(val)
        sets.append(f"{field}=${len(params)}")
    if not sets:
        return {"ok": True}
    params.append(clip_id)
    await db.execute(f"UPDATE clips SET {', '.join(sets)}, updated_at=now() WHERE id=${len(params)}", *params)
    await audit_mod.audit("clip.tag", "clip", clip_id, after=body.model_dump(exclude_none=True), actor=user)
    return {"ok": True}


@router.delete("/{clip_id}")
async def delete_clip(clip_id: str, user=Depends(get_current_user)):
    cfg = get_settings()
    row = await db.fetchrow("SELECT sha256 FROM clips WHERE id=$1", clip_id)
    if not row:
        raise HTTPException(404, "clip not found")
    storage.delete_clip_files(cfg, clip_id)
    await db.execute("DELETE FROM clips WHERE id=$1", clip_id)
    await db.insert("deletion_receipts", {"object_type": "clip", "object_id": clip_id,
                                           "sha256": row["sha256"], "reason": "user_requested"})
    await audit_mod.audit("clip.delete", "clip", clip_id, actor=user)
    return {"ok": True}


class AssignBody(BaseModel):
    profile_id: str | None = None
    create_profile: dict | None = None
    mark_unknown: bool | None = None
    enroll: bool | None = None


@router.post("/{clip_id}/speakers/{local_label}/assign")
async def assign_speaker(clip_id: str, local_label: str, body: AssignBody, user=Depends(get_current_user)):
    from common import speaker as sc  # shared with worker — same enrollment/centroid logic

    cs = await db.fetchrow("SELECT * FROM clip_speakers WHERE clip_id=$1 AND local_label=$2",
                            clip_id, local_label)
    if not cs:
        raise HTTPException(404, "speaker not found on this clip")
    before = {"profile_id": str(cs["profile_id"]) if cs["profile_id"] else None,
              "match_result": cs["match_result"], "score": cs["match_score"]}

    profile_id = body.profile_id
    if body.create_profile:
        name = body.create_profile["display_name"]
        # correcting the same name across clips should accumulate enrollments into one
        # profile, not fragment into a new empty profile every time — a profile with
        # several averaged enrollments identifies far more reliably than several
        # profiles with one (or zero) each
        existing = await db.fetchrow(
            "SELECT id FROM speaker_profiles WHERE lower(display_name)=lower($1)", name)
        profile_id = existing["id"] if existing else await db.insert(
            "speaker_profiles", {"display_name": name, "status": "provisional"})
    if body.mark_unknown:
        profile_id = None

    await db.execute(
        """UPDATE clip_speakers SET profile_id=$1, is_manual=true, match_result='confident'
           WHERE clip_id=$2 AND local_label=$3""", profile_id, clip_id, local_label)
    await db.execute(
        """UPDATE utterances SET profile_id=$1 WHERE clip_id=$2 AND local_label=$3""",
        profile_id, clip_id, local_label)

    enrolled, skip_reason = False, None
    if body.enroll and profile_id:
        if cs["embedding"] is None:
            skip_reason = "not enough clean (non-overlapping) speech from this speaker to build a voiceprint"
        elif (cs["reliability"] or 0) < (await get_effective_settings()).reliability_fair:
            skip_reason = f"voiceprint reliability too low ({cs['reliability'] or 0:.2f})"
        else:
            await sc.add_enrollment(profile_id, cs, clip_id, source="manual", actor=user)
            enrolled = True

    await audit_mod.audit("identification.correct", "clip_speaker", f"{clip_id}:{local_label}",
                           before=before, after={"profile_id": str(profile_id) if profile_id else None}, actor=user)
    return {"ok": True, "profile_id": profile_id, "enrolled": enrolled,
            "enroll_skip_reason": skip_reason, "was": before}
