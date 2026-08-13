import zlib
from ..errors import RejectError

BOILERPLATE = {
    "thank you.", "thanks for watching!", "you", "bye.",
    "subtitles by the amara.org community", "♪",
}


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


async def run(ctx):
    lang = None if ctx.cfg.asr_language == "auto" else ctx.cfg.asr_language
    segments, info = ctx.pool.asr.transcribe(
        ctx.audio_clean, language=lang, task="transcribe",
        beam_size=ctx.cfg.asr_beam_size, word_timestamps=True,
        condition_on_previous_text=False,  # non-negotiable: prevents feedback-loop hallucination
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
        no_speech_threshold=0.6, vad_filter=False)

    words, segs = [], []
    for si, seg in enumerate(segments):
        segs.append({"idx": si, "start": seg.start, "end": seg.end, "text": seg.text.strip(),
                     "avg_logprob": seg.avg_logprob, "no_speech_prob": seg.no_speech_prob})
        for w in (seg.words or []):
            words.append({"word": w.word.strip(), "start": w.start, "end": w.end,
                          "probability": w.probability})

    if not words:
        raise RejectError("NO_TRANSCRIPT", "ASR produced no words")

    ctx.words = drop_hallucinations(words, segs, ctx)
    ctx.segments = segs
    ctx.language = info.language
    ctx.language_conf = float(info.language_probability)

    if ctx.language_conf < 0.5:
        ctx.warn("LOW_LANGUAGE_CONFIDENCE", language=ctx.language, confidence=ctx.language_conf)
