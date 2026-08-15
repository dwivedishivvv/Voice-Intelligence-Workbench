"""Stitch one cluster's own speech into a single audio stream so a human can listen
to the voice in one go and recognise it, instead of opening every clip by hand."""
import hashlib
import os
import subprocess
from pathlib import Path

from common import db, storage

MIN_TURN_S = 0.4      # shorter turns are mostly clipped fragments — useless for recognition
PER_CLIP_CAP_S = 25.0  # keep one long clip from drowning out the rest of the cluster
TOTAL_CAP_S = 75.0
GAP_S = 0.25          # a beat between segments; back-to-back cuts are hard to follow


async def select_segments(cluster_id: str) -> list[dict]:
    """Deterministic pick of the turns that go into the montage."""
    rows = await db.fetch(
        """SELECT t.clip_id::text AS clip_id, t.idx, t.start_s, t.end_s,
                  cs.reliability, c.work_path, c.raw_path
           FROM clip_speakers cs
           JOIN speaker_turns t ON t.clip_id = cs.clip_id AND t.local_label = cs.local_label
           JOIN clips c ON c.id = cs.clip_id
           WHERE cs.cluster_id = $1 AND NOT t.is_overlap AND (t.end_s - t.start_s) >= $2
           ORDER BY cs.reliability DESC NULLS LAST, (t.end_s - t.start_s) DESC, t.clip_id, t.idx""",
        cluster_id, MIN_TURN_S)

    picked, per_clip, total = [], {}, 0.0
    for r in rows:
        dur = r["end_s"] - r["start_s"]
        if total + dur > TOTAL_CAP_S or per_clip.get(r["clip_id"], 0.0) + dur > PER_CLIP_CAP_S:
            continue
        rel = r["work_path"] or r["raw_path"]  # work_path is denoised/normalized — better to listen to
        if not rel:
            continue
        picked.append({**dict(r), "dur": dur, "rel_path": rel})
        per_clip[r["clip_id"]] = per_clip.get(r["clip_id"], 0.0) + dur
        total += dur
    # chronological within each clip reads as speech, not as a shuffle
    picked.sort(key=lambda s: (s["clip_id"], s["start_s"]))
    return picked


def _cache_path(cfg, cluster_id: str, segs: list[dict]) -> str:
    # the hash covers the exact segment set, so a re-clustered member list rebuilds instead
    # of serving a stale montage
    h = hashlib.sha1(
        "|".join(f"{s['clip_id']}:{s['idx']}:{s['start_s']:.2f}:{s['end_s']:.2f}" for s in segs)
        .encode()).hexdigest()[:12]
    p = Path(cfg.data_dir) / "montages" / f"{cluster_id}_{h}.wav"
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def build(cfg, cluster_id: str, segs: list[dict]) -> str:
    """Returns a cached-or-freshly-stitched wav. Blocking — call via to_thread."""
    out = _cache_path(cfg, cluster_id, segs)
    if os.path.exists(out):
        return out

    # -ss/-t before -i seeks on the input, so each segment is decoded from its own cheap
    # seek rather than trimming a full decode of the clip
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y"]
    for s in segs:
        cmd += ["-ss", f"{s['start_s']:.3f}", "-t", f"{s['dur']:.3f}",
                "-i", storage.resolve(cfg, s["rel_path"])]
    pads = "".join(f"[{i}:a]apad=pad_dur={GAP_S}[s{i}];" for i in range(len(segs)))
    joins = "".join(f"[s{i}]" for i in range(len(segs)))
    cmd += ["-filter_complex", f"{pads}{joins}concat=n={len(segs)}:v=0:a=1[out]",
            "-map", "[out]", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out]

    p = subprocess.run(cmd, capture_output=True, timeout=180)
    if p.returncode != 0:
        Path(out).unlink(missing_ok=True)  # never leave a truncated file to be cached
        raise RuntimeError(p.stderr.decode(errors="ignore")[:400])
    return out


def summary(segs: list[dict]) -> dict:
    return {
        "segments": len(segs),
        "seconds": round(sum(s["dur"] for s in segs) + GAP_S * len(segs), 2),
        "clips": sorted({s["clip_id"] for s in segs}),
    }
