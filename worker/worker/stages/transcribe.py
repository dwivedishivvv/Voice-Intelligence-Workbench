import asyncio
import zlib

import structlog

from ..ctx import SR
from ..errors import RejectError

log = structlog.get_logger()

BOILERPLATE = {
    "thank you.", "thanks for watching!", "you", "bye.",
    "subtitles by the amara.org community", "♪",
}

# Whisper's own window. Batched decoding takes one window per clip timestamp and silently
# truncates anything longer, so VAD regions are packed up to — never past — this.
WINDOW_S = 30.0

# The thresholds at which the sequential decoder retries a segment at a higher temperature.
# Batched decoding has no such retry (it only ever uses the first temperature), so a window
# that crosses any of these is re-decoded the sequential way instead of being kept as-is.
COMPRESSION_RETRY = 2.4
LOGPROB_RETRY = -1.0
NO_SPEECH_RETRY = 0.6

# Batched decoding's other failure is silent: on noisy audio it can skip most of a window
# while every confidence number stays healthy. Measured on a noise-mixed meeting: one window
# came back at 0.76 words per second of speech where its neighbours ran at ~3, with a normal
# avg_logprob and no_speech_prob of 0. So word density is checked directly, against the
# clip's own median rather than a fixed rate — a slow speaker or a language with long
# words is not a dropped window.
DENSITY_FLOOR = 0.5        # fraction of the clip's median words-per-speech-second
DENSITY_MIN_SPEECH_S = 5.0  # too little speech in a window for its rate to mean anything


def compression_ratio(t: str) -> float:
    b = t.encode()
    return len(b) / max(len(zlib.compress(b)), 1)


def drop_hallucinations(words, segs, ctx):
    bad = set()
    for s in segs:
        text = s["text"].lower().strip()
        reasons = []
        if s["no_speech_prob"] > 0.85 and s["avg_logprob"] < -0.9:
            reasons.append("no_speech")
        if len(text) > 12 and compression_ratio(text) > 2.4:
            reasons.append("repetitive")
        if text in BOILERPLATE:
            reasons.append("boilerplate")
        if reasons:
            bad.add(s["idx"])
            ctx.warn("HALLUCINATION_FILTERED", segment=s["idx"], reasons=reasons)
    if not bad:
        return words
    spans = [(s["start"], s["end"]) for s in segs if s["idx"] in bad]
    return [w for w in words if not any(a <= w["start"] < b for a, b in spans)]


def pack_windows(vad, max_s: float = WINDOW_S) -> list[list[float]]:
    """Speech regions packed greedily into windows of at most `max_s`.

    One VAD region per window would hand the batched decoder a padded 30s window for every
    sentence — measured 4x *slower* than sequential decoding on a 100s meeting. Packing
    keeps the gaps between regions inside a window (Whisper handles a pause fine) and
    splits only a single region that is itself longer than a window.
    """
    out: list[list[float]] = []
    for s, e in vad:
        while e - s > max_s:
            out.append([s, s + max_s])
            s += max_s
        if out and e - out[-1][0] <= max_s:
            out[-1][1] = e
        else:
            out.append([s, e])
    return out


def needs_sequential(seg: dict) -> bool:
    """Would the sequential decoder have retried this at a higher temperature?"""
    return (compression_ratio(seg["text"]) > COMPRESSION_RETRY
            or seg["avg_logprob"] < LOGPROB_RETRY
            or seg["no_speech_prob"] > NO_SPEECH_RETRY)


def sparse_windows(windows: list[list[float]], vad, words: list[dict]) -> set[int]:
    """Windows whose words-per-speech-second fell well under the clip's own median."""
    rates: dict[int, float] = {}
    for i, (s, e) in enumerate(windows):
        speech = sum(max(0.0, min(e, b) - max(s, a)) for a, b in vad)
        if speech >= DENSITY_MIN_SPEECH_S:
            rates[i] = sum(1 for w in words if s <= w["start"] <= e) / speech
    if len(rates) < 2:
        return set()
    floor = DENSITY_FLOOR * float(sorted(rates.values())[len(rates) // 2])
    return {i for i, r in rates.items() if r < floor}


def _decode_kwargs(cfg) -> dict:
    return dict(task="transcribe", beam_size=cfg.asr_beam_size, word_timestamps=True,
                condition_on_previous_text=False,  # non-negotiable: prevents feedback-loop hallucination
                compression_ratio_threshold=COMPRESSION_RETRY, log_prob_threshold=LOGPROB_RETRY,
                no_speech_threshold=NO_SPEECH_RETRY)


def _collect(segments, offset: float = 0.0, between=None) -> tuple[list[dict], list[dict]]:
    # `segments` is a lazy generator — decoding happens here, as it is consumed, which is
    # why this has to run inside the worker thread and not just the transcribe() call.
    # `between` runs after each segment: the point where a waiting live chunk is let into
    # the model (lane.step_aside_from_thread) — the generator's state is all on the Python
    # side, so another transcribe() call in between does not disturb it.
    words, segs = [], []
    for seg in segments:
        if between is not None:
            between()
        segs.append({"start": seg.start + offset, "end": seg.end + offset, "text": seg.text.strip(),
                     "avg_logprob": seg.avg_logprob, "no_speech_prob": seg.no_speech_prob})
        for w in (seg.words or []):
            words.append({"word": w.word.strip(), "start": w.start + offset, "end": w.end + offset,
                          "probability": w.probability})
    return words, segs


def _sequential(ctx, audio, lang, offset: float = 0.0, between=None):
    segments, info = ctx.pool.asr.transcribe(
        audio, language=lang, temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], vad_filter=False,
        **_decode_kwargs(ctx.cfg))
    words, segs = _collect(segments, offset, between)
    return words, segs, info


def _batched(ctx, lang, between=None):
    """Windows decoded together in batches, then any doubtful window decoded again alone.

    Measured on this hardware (RTX 4050, large-v3-turbo fp16, beam 5, word timestamps):
    10 minutes of meeting audio went from ~17.7s to ~8.5s at batch 8, same word error rate.
    The things batching gives up are restored where they matter:

      - completeness: a window with far fewer words per second of speech than the rest of
        the clip (sparse_windows) is re-decoded sequentially.

      - temperature fallback: batched decoding only ever uses temperature 0. A window that
        crosses any sequential retry threshold is re-decoded sequentially, with fallback.
      - segment granularity: without timestamp tokens (the mode that is actually faster —
        with them batching measured no gain at all) each window comes back as one segment,
        and drop_hallucinations drops whole segments. Re-decoded windows come back at the
        sequential decoder's normal sentence-level granularity, so a filter hit there costs
        a sentence, not a 30s window; a clean window never trips the filter in the first
        place, since its thresholds sit above the retry thresholds.
    """
    from faster_whisper import BatchedInferencePipeline
    windows = pack_windows(ctx.vad)
    pipe = BatchedInferencePipeline(model=ctx.pool.asr)
    segments, info = pipe.transcribe(
        ctx.audio_clean, language=lang, batch_size=ctx.cfg.asr_batch_size,
        clip_timestamps=[{"start": s, "end": e} for s, e in windows], without_timestamps=True,
        temperature=0.0, **_decode_kwargs(ctx.cfg))
    words, segs = _collect(segments, between=between)

    redo = sparse_windows(windows, ctx.vad, words)
    for seg in segs:
        if needs_sequential(seg):
            mid = (seg["start"] + seg["end"]) / 2
            redo.update(i for i, (s, e) in enumerate(windows) if s <= mid <= e)
    if redo:
        spans = [windows[i] for i in sorted(redo)]
        inside = lambda t: any(s <= t <= e for s, e in spans)  # noqa: E731
        words = [w for w in words if not inside(w["start"])]
        segs = [s for s in segs if not inside((s["start"] + s["end"]) / 2)]
        # pinned to the detected language: these are fragments of a recording whose
        # language was already decided on its opening 30s, same as the sequential path
        for s, e in spans:
            w2, s2, _ = _sequential(ctx, ctx.audio_clean[int(s * SR):int(e * SR)], info.language, s, between)
            words += w2
            segs += s2
        words.sort(key=lambda w: w["start"])
        segs.sort(key=lambda s: s["start"])
        log.info("asr_windows_redecoded", n=len(redo), of=len(windows))
    return words, segs, info


def use_batched(ctx) -> bool:
    # Poor audio stays on the sequential path outright: that is where batched decoding was
    # measured dropping speech (WER 0.36 vs 0.32 on a 0.7 dB SNR recording), and where the
    # temperature fallback it lacks earns its keep. Preprocess grades quality before this runs.
    return (ctx.cfg.asr_batch_size > 0 and ctx.pool.device == "cuda" and bool(ctx.vad)
            and (ctx.quality or {}).get("grade") != "poor"
            and ctx.audio_clean.size / SR >= ctx.cfg.asr_batched_min_s)


def _transcribe(ctx, between=None):
    lang = None if ctx.cfg.asr_language == "auto" else ctx.cfg.asr_language
    if use_batched(ctx):
        try:
            words, segs, info = _batched(ctx, lang, between)
        except RuntimeError as e:
            # A batch of windows needs more VRAM than one window does; on OOM the
            # sequential path still fits, so take it rather than failing the clip.
            if "out of memory" not in str(e).lower():
                raise
            import torch
            torch.cuda.empty_cache()
            log.warning("asr_batched_oom_fallback", batch_size=ctx.cfg.asr_batch_size)
            words, segs, info = _sequential(ctx, ctx.audio_clean, lang, between=between)
    else:
        words, segs, info = _sequential(ctx, ctx.audio_clean, lang, between=between)
    for i, s in enumerate(segs):
        s["idx"] = i
    return words, segs, info


async def run(ctx):
    # Off the event loop: decoding an hour of audio held it for minutes, during which the
    # worker could not publish progress or answer arq's health check. Holding Whisper's own
    # lane, and stepping aside between windows for any live chunk waiting on it (lane.py).
    lane = getattr(ctx.pool, "asr_lane", None)
    if lane is None:
        words, segs, info = await asyncio.to_thread(_transcribe, ctx)
    else:
        loop = asyncio.get_running_loop()
        async with lane:
            words, segs, info = await asyncio.to_thread(
                _transcribe, ctx, lambda: lane.step_aside_from_thread(loop))

    if not words:
        raise RejectError("NO_TRANSCRIPT", "ASR produced no words")

    ctx.words = drop_hallucinations(words, segs, ctx)
    ctx.segments = segs
    ctx.language = info.language
    ctx.language_conf = float(info.language_probability)

    if ctx.language_conf < 0.5:
        ctx.warn("LOW_LANGUAGE_CONFIDENCE", language=ctx.language, confidence=ctx.language_conf)
