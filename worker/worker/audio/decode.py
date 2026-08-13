import io
import json
import subprocess

import numpy as np
import soundfile as sf

from ..errors import RejectError

SR = 16000


def ffprobe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise RejectError("CORRUPT", out.stderr[:400])
    d = json.loads(out.stdout)
    audio = [s for s in d.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio:
        return {"audio_streams": []}
    a, fmt = audio[0], d.get("format", {})
    return {
        "audio_streams": audio,
        "duration": float(a.get("duration") or fmt.get("duration") or 0),
        "format_duration": float(fmt["duration"]) if fmt.get("duration") else None,
        "codec": a.get("codec_name"),
        "sample_rate": int(a.get("sample_rate", 0) or 0),
        "channels": int(a.get("channels", 0) or 0),
        "bitrate": int(fmt.get("bit_rate", 0) or 0),
        "container": fmt.get("format_name"),
        "size_bytes": int(fmt.get("size", 0) or 0),
    }


def decode_to_array(path: str, sr: int = SR) -> np.ndarray:
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-i", path,
         "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", "-f", "wav", "-"],
        capture_output=True, timeout=120)
    if p.returncode != 0:
        raise RejectError("DECODE_FAILED", p.stderr.decode(errors="ignore")[:400])
    audio, got_sr = sf.read(io.BytesIO(p.stdout), dtype="float32")
    assert got_sr == sr
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio


def write_wav(path: str, audio: np.ndarray, sr: int = SR):
    sf.write(path, audio, sr)
