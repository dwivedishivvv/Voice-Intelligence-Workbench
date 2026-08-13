"""Live model catalog: browse alternatives, pull one on demand, activate a downloaded one.
Downloads run in a thread (snapshot_download is a blocking call) tracked via the
downloaded_models table so status survives across requests and page reloads. Activating a
model just writes a settings_overrides row for the category's *_model field — ModelPool only
reads that at worker startup, so it's inert until the worker (and, for text_embed_model, the
api) restarts. That's intentional: hot-swapping a loaded torch model is a much bigger change
than this feature warrants."""
import asyncio
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from common import db
from common.config import get_settings
from common.models_catalog import CATALOG, find_option
from ..auth import get_current_user

router = APIRouter(prefix="/v1/admin/models", tags=["models"])

_inflight: set[tuple[str, str]] = set()
# HF_HUB_OFFLINE=1 is correct for every other code path in this process (guarantees the
# pipeline never silently reaches out to the network) but has to be lifted for the duration
# of a deliberate live pull. os.environ is process-global, not thread-local, so this lock
# fully serializes downloads — one at a time, never overlapping with itself — rather than
# risk another concurrent request seeing offline mode flipped off underneath it.
_download_lock = asyncio.Lock()


# A gated repo that 401s still leaves a directory behind holding whatever was public
# (pyannote's leaves README.md + reproducible_research/), so "directory is non-empty" reports
# a model as ready that the worker then can't load. Require an actual weights/pipeline file.
_WEIGHT_SUFFIXES = {".bin", ".safetensors", ".ckpt", ".pt", ".pth", ".onnx"}


def _has_weights(d: Path) -> bool:
    return any(f.suffix in _WEIGHT_SUFFIXES or f.name == "config.yaml"
               for f in d.rglob("*") if f.is_file())


def _model_path(category: str, dir_name: str) -> Path:
    cfg = get_settings()
    sub = ("diar" if category == "diarization" else "asr" if category == "asr"
           else "embed" if category == "embedding" else "text")
    return Path(cfg.model_dir) / sub / dir_name


@router.get("")
async def list_models(user=Depends(get_current_user)):
    cfg = get_settings()
    rows = await db.fetch("SELECT category, repo_id, dir_name, status, error FROM downloaded_models")
    downloaded = {(r["category"], r["repo_id"]): dict(r) for r in rows}

    out = {}
    for category, spec in CATALOG.items():
        active_dir = getattr(cfg, spec["settings_field"])
        options = []
        for opt in spec["options"]:
            state = downloaded.get((category, opt["repo_id"]))
            # No DB row doesn't mean absent: models baked in at image build (or fetched by
            # models/prefetch.py) never went through /pull, so disk is the other source of
            # truth. Without this the page offers to re-download what's already there.
            if state:
                status = state["status"]
            else:
                d = _model_path(category, opt["dir_name"])
                status = "downloaded" if d.is_dir() and _has_weights(d) else "not_downloaded"
            options.append({
                **opt,
                "active": opt["dir_name"] == active_dir,
                "status": "downloading" if (category, opt["repo_id"]) in _inflight else status,
                "error": state["error"] if state else None,
            })
        out[category] = {"gated": spec["gated"], "options": options}
    return out


class PullBody(BaseModel):
    category: str
    repo_id: str


def _hf_token() -> str | None:
    for p in (Path(".hf_token"), Path("/run/secrets/hf_token")):
        if p.exists():
            return p.read_text().strip() or None
    return None


def _reset_symlink_probe():
    """huggingface_hub probes symlink support once per cache subdir and memoizes it, but it
    seeds the entry True *before* the probe and only flips it to False inside the probe's
    own except — so if the probe itself dies (transient lock on the temp dir it creates),
    the dir is remembered as symlink-capable for the life of the process. Every later write
    there then hits WinError 1314, which no amount of retrying from the UI can clear.
    Dropping the memo makes Retry re-probe instead of needing an api restart."""
    try:
        from huggingface_hub.file_download import _are_symlinks_supported_in_dir
        _are_symlinks_supported_in_dir.clear()
    except ImportError:  # private, may move between hub versions — retry is still worth trying
        pass


def _do_download(category: str, opt: dict, gated: bool):
    from huggingface_hub import snapshot_download

    _reset_symlink_probe()
    dest = _model_path(category, opt["dir_name"])
    token = _hf_token() if gated else None
    snapshot_download(repo_id=opt["repo_id"], local_dir=str(dest), token=token, max_workers=4)
    # Same reason prefetch.py fetches these: nested repo-id refs inside a pipeline's config
    # (pyannote's segmentation submodel, speechbrain's pretrained_path sub-checkpoints) are
    # resolved at runtime by repo id through the HF *cache*, not this flat local_dir copy.
    # Without them the model shows "downloaded" here but the worker fails to load it offline.
    for dep_id in opt.get("depends_on", []):
        snapshot_download(repo_id=dep_id, token=token, max_workers=4)


@router.post("/pull")
async def pull_model(body: PullBody, user=Depends(get_current_user)):
    spec = CATALOG.get(body.category)
    if not spec:
        raise HTTPException(404, f"unknown category: {body.category}")
    opt = find_option(body.category, body.repo_id)
    if not opt:
        raise HTTPException(404, f"unknown model for {body.category}: {body.repo_id}")
    key = (body.category, body.repo_id)
    if key in _inflight:
        return {"ok": True, "status": "downloading"}

    await db.execute(
        """INSERT INTO downloaded_models (category, repo_id, dir_name, status)
           VALUES ($1, $2, $3, 'downloading')
           ON CONFLICT (category, repo_id) DO UPDATE SET status='downloading', error=NULL""",
        body.category, body.repo_id, opt["dir_name"])

    async def run():
        _inflight.add(key)
        try:
            async with _download_lock:
                prev = os.environ.get("HF_HUB_OFFLINE")
                os.environ["HF_HUB_OFFLINE"] = "0"
                try:
                    await asyncio.to_thread(_do_download, body.category, opt, spec["gated"])
                finally:
                    if prev is None:
                        os.environ.pop("HF_HUB_OFFLINE", None)
                    else:
                        os.environ["HF_HUB_OFFLINE"] = prev
            await db.execute(
                "UPDATE downloaded_models SET status='downloaded', downloaded_at=now() WHERE category=$1 AND repo_id=$2",
                body.category, body.repo_id)
        except Exception as e:
            await db.execute(
                "UPDATE downloaded_models SET status='error', error=$3 WHERE category=$1 AND repo_id=$2",
                body.category, body.repo_id, str(e)[:500])
        finally:
            _inflight.discard(key)

    asyncio.create_task(run())
    return {"ok": True, "status": "downloading"}


class ActivateBody(BaseModel):
    category: str
    repo_id: str


@router.post("/activate")
async def activate_model(body: ActivateBody, user=Depends(get_current_user)):
    spec = CATALOG.get(body.category)
    if not spec:
        raise HTTPException(404, f"unknown category: {body.category}")
    opt = find_option(body.category, body.repo_id)
    if not opt:
        raise HTTPException(404, f"unknown model for {body.category}: {body.repo_id}")
    row = await db.fetchrow(
        "SELECT status FROM downloaded_models WHERE category=$1 AND repo_id=$2", body.category, body.repo_id)
    if not row or row["status"] != "downloaded":
        raise HTTPException(400, "model isn't downloaded yet")

    await db.execute(
        """INSERT INTO settings_overrides (key, value, updated_at, updated_by)
           VALUES ($1, $2, now(), $3)
           ON CONFLICT (key) DO UPDATE SET value=$2, updated_at=now(), updated_by=$3""",
        spec["settings_field"], opt["dir_name"], user)
    return {"ok": True, "note": "Takes effect after the worker restarts."}
