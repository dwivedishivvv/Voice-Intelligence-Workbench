"""Shared clip-ingest path: stream to disk, dedupe by content hash, register, enqueue.

Extracted so /v1/clips and /v1/races/{id}/clips are literally the same ingest — a race
recording is an ordinary clip that happens to carry a race_id, not a parallel pipeline.
Anything added here (size limits, dedupe, audit) applies to both by construction.
"""
import asyncio
import hashlib
import os
from pathlib import Path

from fastapi import HTTPException, UploadFile

from common import db, storage, audit as audit_mod
from . import queue


async def ingest_clip(file: UploadFile, cfg, user: str, *, tags: list[str] | None = None,
                      notes: str | None = None, race_id: str | None = None) -> dict:
    ext = Path(file.filename or "upload.wav").suffix or ".bin"
    tmp_path = Path(cfg.data_dir) / "tmp" / f"{os.urandom(8).hex()}{ext}"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    h = hashlib.sha256()
    size = 0
    max_bytes = cfg.max_upload_mb * 1024 * 1024
    with open(tmp_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(413, f"file exceeds {cfg.max_upload_mb}MB limit")
            h.update(chunk)
            await asyncio.to_thread(out.write, chunk)
    sha = h.hexdigest()

    existing = await db.fetchrow("SELECT id, status, race_id FROM clips WHERE sha256=$1", sha)
    if existing:
        tmp_path.unlink(missing_ok=True)
        # re-uploading the same audio into a race is a *move*, not a no-op: the operator
        # is telling us where it belongs. Without this, dragging a folder that overlaps a
        # previous batch silently leaves those files out of the race they were dropped on.
        if race_id and str(existing["race_id"] or "") != race_id:
            await db.execute("UPDATE clips SET race_id=$2 WHERE id=$1", existing["id"], race_id)
        return {"clip_id": str(existing["id"]), "duplicate": True, "status": existing["status"]}

    clip_id = await db.insert("clips", {
        "sha256": sha, "filename": file.filename or "upload", "mime": file.content_type,
        "size_bytes": size, "raw_path": "", "tags": tags or [],
        "notes": notes or None, "race_id": race_id,
    })
    rel_path = storage.raw_path(cfg, clip_id, ext)
    tmp_path.rename(storage.resolve(cfg, rel_path))
    await db.execute("UPDATE clips SET raw_path=$1 WHERE id=$2", rel_path, clip_id)
    await audit_mod.audit("clip.upload", "clip", clip_id,
                          after={"filename": file.filename, "race_id": race_id}, actor=user)

    job_id = await queue.enqueue_clip(clip_id)
    return {"clip_id": clip_id, "job_id": job_id, "status": "QUEUED", "duplicate": False,
            "ws_url": f"/v1/ws/jobs/{job_id}"}
