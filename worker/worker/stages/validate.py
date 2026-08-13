import asyncio

from common import db, storage
from ..audio.decode import ffprobe
from ..errors import RejectError


async def run(ctx):
    row = await db.fetchrow("SELECT raw_path FROM clips WHERE id=$1", ctx.clip_id)
    # raw_path is stored relative to DATA_DIR (see common/storage.py) — resolve once here
    # so every later use of ctx.raw_path in this process gets a correct absolute path.
    ctx.raw_path = storage.resolve(ctx.cfg, row["raw_path"])

    # ffprobe shells out and blocks; off the event loop so other clips' jobs keep moving
    probe = await asyncio.to_thread(ffprobe, ctx.raw_path)
    if not probe.get("audio_streams"):
        raise RejectError("NO_AUDIO_STREAM", "file contains no audio track")

    dur = float(probe["duration"])
    if dur > ctx.cfg.max_duration_s:
        raise RejectError("TOO_LONG", f"{dur:.1f}s exceeds limit of {ctx.cfg.max_duration_s:.0f}s")
    if dur < ctx.cfg.min_duration_s:
        raise RejectError("TOO_SHORT", f"{dur:.2f}s")
    if dur > ctx.cfg.target_duration_s:
        ctx.warn("ABOVE_TARGET_DURATION", duration_s=dur)

    hdr = probe.get("format_duration")
    if hdr and abs(hdr - dur) / max(dur, 1e-6) > 0.10:
        raise RejectError("CORRUPT", f"header says {hdr:.1f}s, streams say {dur:.1f}s")

    ctx.probe, ctx.duration_s = probe, dur
    await db.execute(
        """UPDATE clips SET duration_s=$2, sample_rate=$3, channels=$4, mime=$5 WHERE id=$1""",
        ctx.clip_id, dur, probe.get("sample_rate"), probe.get("channels"), probe.get("codec"))
