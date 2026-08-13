"""Object storage → local disk. Single-node offline deployment has no use for an
S3-compatible service; swap for MinIO/S3 if this ever goes multi-node.

raw_path/work_path are persisted in the clips table and read back by a *different*
process than the one that wrote them (api uploads, worker preprocesses, api serves
audio/exports) — processes that can have a different DATA_DIR for the same underlying
directory (e.g. a Dockerized api at /data vs. a natively-run worker at C:\\data, both
bind-mounted to the same folder). An absolute path baked in by whichever process wrote
it is meaningless to the other, so raw_path/work_path are stored *relative* to DATA_DIR
and resolved against the reading process's own cfg.data_dir via resolve()."""
from pathlib import Path


def _base(cfg) -> Path:
    return Path(cfg.data_dir)


def resolve(cfg, rel_path: str) -> str:
    """Turn a relative path (as stored in the DB) into an absolute one for this process."""
    return str(_base(cfg) / rel_path)


def raw_path(cfg, clip_id: str, ext: str) -> str:
    (_base(cfg) / "raw").mkdir(parents=True, exist_ok=True)
    return f"raw/{clip_id}{ext}"


def work_path(cfg, clip_id: str, name: str) -> str:
    (_base(cfg) / "work" / clip_id).mkdir(parents=True, exist_ok=True)
    return f"work/{clip_id}/{name}"


def export_path(cfg, clip_id: str, name: str) -> str:
    """Unlike raw_path/work_path, never persisted in the DB — always recomputed by
    clip_id + a fixed filename, so it's returned absolute for the caller's immediate use."""
    p = _base(cfg) / "exports" / clip_id / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def delete_clip_files(cfg, clip_id: str):
    import shutil
    for sub in ("raw", "work", "exports"):
        d = _base(cfg) / sub
        for f in d.glob(f"{clip_id}*"):
            f.unlink(missing_ok=True)
        cd = d / clip_id
        if cd.is_dir():
            shutil.rmtree(cd, ignore_errors=True)
