import asyncio

import numpy as np

from common import db, storage
from ..ctx import SR
from ..audio.decode import decode_to_array, write_wav
from ..audio.enhance import highpass, loudness_normalize, denoise, spectral_flatness
from ..audio.vad import detect_vad
from ..audio.quality import compute_quality
from ..errors import RejectError


def normalized(path: str, cfg):
    """Decoded audio up to the fork: DC removed, high-passed, loudness-normalized. This is
    the signal voiceprints are taken from (audio_norm), shared with worker/reembed.py so a
    re-embedded clip is fingerprinted from exactly what the pipeline would have used."""
    audio = decode_to_array(path, SR)
    audio = audio - float(np.mean(audio))          # DC removal
    audio = highpass(audio, SR, cfg.highpass_hz)
    return loudness_normalize(audio, SR, cfg.target_lufs)


def _decode_and_clean(ctx):
    # ffmpeg subprocess + numpy/scipy/noisereduce CPU work — all blocking, so this
    # whole chain runs in a worker thread (see run() below) rather than the shared
    # event loop, so it doesn't stall every other clip's job while it runs.
    audio, input_lufs = normalized(ctx.raw_path, ctx.cfg)
    audio_norm = audio.copy()                      # fork: faithful copy for embeddings

    audio_clean = denoise(audio, SR, ctx.cfg, device=ctx.pool.device)
    return audio_norm, audio_clean, input_lufs


def _vad_and_quality(ctx, audio_norm, audio_clean, input_lufs):
    clip_ratio = float(np.mean(np.abs(audio_norm) >= ctx.cfg.clipping_threshold))
    vad = detect_vad(audio_clean, SR, ctx.pool.vad, ctx.cfg)
    speech_s = sum(e - s for s, e in vad)

    if speech_s < ctx.cfg.min_total_speech_s:
        flat = spectral_flatness(audio_norm)
        code = "NO_SPEECH_DETECTED" if flat < 0.4 else "INSUFFICIENT_SPEECH"
        raise RejectError(code, f"only {speech_s:.2f}s of speech detected")

    quality = compute_quality(audio_norm, vad, SR, clip_ratio, input_lufs, ctx.cfg)
    return vad, speech_s, quality


async def run(ctx):
    ctx.audio_norm, ctx.audio_clean, input_lufs = await asyncio.to_thread(_decode_and_clean, ctx)
    ctx.vad, ctx.speech_s, ctx.quality = await asyncio.to_thread(
        _vad_and_quality, ctx, ctx.audio_norm, ctx.audio_clean, input_lufs)

    if ctx.quality["grade"] == "poor":
        ctx.warn("POOR_AUDIO_QUALITY", snr_db=ctx.quality["snr_db"],
                  clipping_ratio=ctx.quality["clipping_ratio"], bandwidth_hz=ctx.quality["bandwidth_hz"])

    work_wav_rel = storage.work_path(ctx.cfg, ctx.clip_id, "clean.wav")
    await asyncio.to_thread(write_wav, storage.resolve(ctx.cfg, work_wav_rel), ctx.audio_clean, SR)
    await db.execute("UPDATE clips SET work_path=$2 WHERE id=$1", ctx.clip_id, work_wav_rel)

    q = ctx.quality
    await db.execute(
        """INSERT INTO quality_metrics (clip_id, snr_db, clipping_ratio, silence_ratio,
               speech_duration_s, bandwidth_hz, dc_offset, input_lufs, grade)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)""",
        ctx.clip_id, q["snr_db"], q["clipping_ratio"], q["silence_ratio"],
        q["speech_duration_s"], q["bandwidth_hz"], q["dc_offset"], q["input_lufs"], q["grade"])
    if ctx.vad:
        # one round trip, not one per region — a long recording has thousands of them
        await db.pool().executemany(
            "INSERT INTO vad_regions (clip_id, idx, start_s, end_s) VALUES ($1,$2,$3,$4)",
            [(ctx.clip_id, i, s, e) for i, (s, e) in enumerate(ctx.vad)])
