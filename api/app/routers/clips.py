import json
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from common.config import get_settings, get_effective_settings
from common import db, storage, audit as audit_mod, graph_sync
from ..auth import get_current_user
from ..services import queue, ingest, speaker_ops

router = APIRouter(prefix="/v1/clips", tags=["clips"])


@router.post("")
async def upload_clip(file: UploadFile = File(...), tags: str = "", notes: str = "",
                       user=Depends(get_current_user)):
    cfg = await get_effective_settings()
    return await ingest.ingest_clip(file, cfg, user,
                                     tags=tags.split(",") if tags else [], notes=notes)


@router.get("")
async def list_clips(q: str | None = None, status: str | None = None,
                      needs_review: bool | None = None, tag: list[str] = Query(default=[]),
                      limit: int = 25, offset: int = 0, user=Depends(get_current_user)):
    conds, params = [], []
    if status:
        params.append(status); conds.append(f"status=${len(params)}")
    if needs_review is not None:
        params.append(needs_review); conds.append(f"needs_review=${len(params)}")
    if q:
        params.append(f"%{q}%"); conds.append(f"filename ILIKE ${len(params)}")
    # Every tag given must be present, from either column: the operator's or the
    # pipeline's. ?tag=voices:dialogue&tag=lang:en narrows, it does not widen.
    for t in tag:
        params.append(t); conds.append(f"(${len(params)} = ANY(tags) OR ${len(params)} = ANY(auto_tags))")
    where = f"WHERE {' AND '.join(conds)}" if conds else ""
    params += [limit, offset]
    rows = await db.fetch(
        f"""SELECT id, filename, duration_s, status, language, n_speakers, needs_review,
                   created_at, error_code, tags, auto_tags, notes
            FROM clips {where} ORDER BY created_at DESC LIMIT ${len(params)-1} OFFSET ${len(params)}""",
        *params)
    total = await db.fetchval(f"SELECT count(*) FROM clips {where}", *params[:-2])
    return {"items": [dict(r) for r in rows], "total": total}


@router.get("/tags")
async def tag_counts(user=Depends(get_current_user)):
    """Every tag in use with how many clips carry it, manual and automatic apart — the
    Library's filter menu. Declared before /{clip_id} so "tags" is not read as an id."""
    rows = await db.fetch(
        """SELECT tag, source, count(*)::int AS n FROM (
             SELECT unnest(tags) AS tag, 'manual' AS source FROM clips
             UNION ALL
             SELECT unnest(auto_tags), 'auto' FROM clips WHERE auto_tags IS NOT NULL
           ) t
           WHERE tag <> ''
           GROUP BY tag, source ORDER BY n DESC, tag""")
    return {"items": [dict(r) for r in rows]}


@router.post("/{clip_id}/retag")
async def retag_clip(clip_id: str, user=Depends(get_current_user)):
    row = await db.fetchrow("SELECT status FROM clips WHERE id=$1", clip_id)
    if not row:
        raise HTTPException(404, "clip not found")
    if row["status"] != "COMPLETE":
        raise HTTPException(409, "only a COMPLETE clip has the analysis tags are read from")
    job_id = await queue.enqueue_retag([clip_id])
    return {"job_id": job_id}


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
    filename: str | None = None
    tags: list[str] | None = None
    notes: str | None = None
    needs_review: bool | None = None


@router.patch("/{clip_id}")
async def patch_clip(clip_id: str, body: PatchClip, user=Depends(get_current_user)):
    if body.filename is not None:
        body.filename = body.filename.strip()
        if not body.filename:
            raise HTTPException(400, "filename cannot be empty")
    if body.tags is not None:
        # trimmed and de-duplicated, order kept: "call, call , " is one tag, not three
        body.tags = list(dict.fromkeys(t.strip() for t in body.tags if t.strip()))
    sets, params = [], []
    for field, val in body.model_dump(exclude_none=True).items():
        params.append(val)
        sets.append(f"{field}=${len(params)}")
    if not sets:
        return {"ok": True}
    params.append(clip_id)
    await db.execute(f"UPDATE clips SET {', '.join(sets)}, updated_at=now() WHERE id=${len(params)}", *params)
    await audit_mod.audit("clip.edit", "clip", clip_id, after=body.model_dump(exclude_none=True), actor=user)
    # The name and tags are graph properties and Tag edges.
    if body.filename is not None or body.tags is not None:
        await graph_sync.resync(queue.redis())
    return {"ok": True}


async def _delete_one(clip_id: str, user: str) -> bool:
    cfg = get_settings()
    row = await db.fetchrow("SELECT sha256 FROM clips WHERE id=$1", clip_id)
    if not row:
        return False
    storage.delete_clip_files(cfg, clip_id)
    await db.execute("DELETE FROM clips WHERE id=$1", clip_id)
    await db.insert("deletion_receipts", {"object_type": "clip", "object_id": clip_id,
                                           "sha256": row["sha256"], "reason": "user_requested"})
    await audit_mod.audit("clip.delete", "clip", clip_id, actor=user)
    return True


class BulkBody(BaseModel):
    ids: list[str]
    # delete | reprocess | add_tag | remove_tag | mark_reviewed | flag_review
    action: str
    tag: str | None = None


BULK_ACTIONS = {"delete", "reprocess", "add_tag", "remove_tag", "mark_reviewed", "flag_review"}


@router.post("/bulk")
async def bulk(body: BulkBody, user=Depends(get_current_user)):
    """One action over many clips — the Library's selection bar. Each clip goes through
    the same path its single-clip endpoint uses, so audit and cleanup are identical."""
    if body.action not in BULK_ACTIONS:
        raise HTTPException(400, f"action must be one of {', '.join(sorted(BULK_ACTIONS))}")
    tag = (body.tag or "").strip()
    if body.action in ("add_tag", "remove_tag") and not tag:
        raise HTTPException(400, "tag is required")
    ids = list(dict.fromkeys(body.ids))[:500]
    done = 0
    for clip_id in ids:
        if body.action == "delete":
            done += await _delete_one(clip_id, user)
        elif body.action == "reprocess":
            if await db.fetchval("SELECT 1 FROM clips WHERE id=$1", clip_id):
                await db.execute("UPDATE clips SET status='QUEUED', error_code=NULL, "
                                 "error_detail=NULL WHERE id=$1", clip_id)
                await queue.enqueue_clip(clip_id)
                done += 1
        elif body.action == "add_tag":
            done += int((await db.execute(
                "UPDATE clips SET tags = array_append(tags, $2), updated_at=now() "
                "WHERE id=$1 AND NOT ($2 = ANY(tags))", clip_id, tag)).split()[-1])
        elif body.action == "remove_tag":
            done += int((await db.execute(
                "UPDATE clips SET tags = array_remove(tags, $2), updated_at=now() "
                "WHERE id=$1 AND $2 = ANY(tags)", clip_id, tag)).split()[-1])
        else:
            done += int((await db.execute(
                "UPDATE clips SET needs_review=$2, updated_at=now() WHERE id=$1",
                clip_id, body.action == "flag_review")).split()[-1])
    await audit_mod.audit(f"clip.bulk.{body.action}", "clip", ",".join(ids)[:200],
                          after={"n": done, "tag": tag or None}, actor=user)
    if body.action != "reprocess":
        await graph_sync.resync(queue.redis())
    return {"ok": True, "n": done}


class PatchUtterance(BaseModel):
    text: str


@router.patch("/{clip_id}/utterances/{utterance_id}")
async def patch_utterance(clip_id: str, utterance_id: str, body: PatchUtterance,
                          user=Depends(get_current_user)):
    """Correct one line of a transcript. The full-text index follows by trigger; the
    vector is re-encoded here so semantic search finds the corrected words, not the
    misheard ones."""
    text = " ".join(body.text.split())
    if not text:
        raise HTTPException(400, "text cannot be empty")
    before = await db.fetchval(
        "SELECT text FROM utterances WHERE id=$1 AND clip_id=$2", utterance_id, clip_id)
    if before is None:
        raise HTTPException(404, "utterance not found on this clip")
    try:
        from .. import text_embed
        vec = list(map(float, text_embed.encode(text)))
    except Exception:
        vec = None  # no encoder on this box: keep the old vector rather than fail the edit
    if vec is not None:
        await db.execute("UPDATE utterances SET text=$2, embedding=$3::vector WHERE id=$1",
                         utterance_id, text, vec)
    else:
        await db.execute("UPDATE utterances SET text=$2 WHERE id=$1", utterance_id, text)
    await audit_mod.audit("utterance.edit", "utterance", utterance_id,
                          before={"text": before}, after={"text": text}, actor=user)
    await graph_sync.resync(queue.redis())
    return {"ok": True, "text": text}


@router.delete("/{clip_id}")
async def delete_clip(clip_id: str, user=Depends(get_current_user)):
    if not await _delete_one(clip_id, user):
        raise HTTPException(404, "clip not found")
    await graph_sync.resync(queue.redis())
    return {"ok": True}


class AssignBody(BaseModel):
    profile_id: str | None = None
    create_profile: dict | None = None
    mark_unknown: bool | None = None
    enroll: bool | None = None


@router.post("/{clip_id}/speakers/{local_label}/assign")
async def assign_speaker(clip_id: str, local_label: str, body: AssignBody, user=Depends(get_current_user)):
    if not await db.fetchval("SELECT 1 FROM clip_speakers WHERE clip_id=$1 AND local_label=$2",
                             clip_id, local_label):
        raise HTTPException(404, "speaker not found on this clip")

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

    result = await speaker_ops.assign(clip_id, local_label, profile_id, enroll=bool(body.enroll),
                                      cfg=await get_effective_settings(), actor=user)
    # A new enrolment can match this voice in any other clip, so identification re-runs
    # over the corpus; that pass also schedules the graph rebuild for whatever it changes.
    # The graph is scheduled here too for the correction itself (SPOKEN_BY edges).
    await queue.enqueue_reidentify()
    await graph_sync.resync(queue.redis())
    return result
