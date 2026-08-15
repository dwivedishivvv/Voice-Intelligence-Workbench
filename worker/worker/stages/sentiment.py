"""Per-utterance sentiment: a trained multilingual text classifier over what was said,
fused with the acoustic tone read of how it was said.

Two signals, stored separately and then fused, rather than one number:

  - text     — XLM-R sentiment head (negative/neutral/positive). Knows *content*: "we have
               a problem" is negative however evenly it's delivered.
  - acoustic — worker.audio.tone over that utterance's own audio slice, calibrated against a
               clip-scoped baseline. Knows *delivery*: pitch and speech rate rising against
               this speaker's own norm, which text can't see.

They disagree usefully often (flat delivery of bad news; a cheerful-sounding complaint), so
the UI gets all three columns and can show the disagreement instead of averaging it away.
Fusion is deliberately a small, tunable nudge (SENTIMENT_MOOD_WEIGHT) on top of the text
score — the text model is the trained one, and acoustics on a few seconds of radio-quality
audio are the noisier signal of the two.
"""
import asyncio

import numpy as np

from common import db
from ..ctx import SR
from ..audio import tone as tone_mod

# how far a stressed/tired delivery can pull the fused score away from what the words alone
# say, in units of the signed text score's own -1..1 range. Tuned by ear on radio-style
# audio, not fit on a labelled set — raise it if delivery should dominate content.
MOOD_SHIFT = {"stressed": -0.18, "tired": -0.10, "calm": 0.03}

# |fused| below this reads as neutral. The text model is confidently neutral on most
# routine speech, and without a deadband the mood nudge alone would flip those to
# negative/positive on nothing more than a slightly raised voice.
NEUTRAL_BAND = 0.20

# The nudge must stay strictly inside the deadband, or acoustics alone can label a
# textually-neutral utterance — which is the false positive the deadband exists to stop.
# Delivery may push a borderline score over the line; it may never create one on its own.
assert max(abs(v) for v in MOOD_SHIFT.values()) < NEUTRAL_BAND

# an utterance shorter than this has too few frames for pyin to say anything about pitch
# (tone.extract_features returns NEUTRAL_FEATURES below its own 1.5s floor anyway) — skip
# the acoustic read entirely rather than feed the fusion a hardcoded zero as if it measured one
MIN_ACOUSTIC_S = 1.5


def _normalize(label: str) -> str:
    """Collapse whatever the chosen checkpoint calls its classes down to neg/neu/pos.
    The catalog offers 3-class (negative/neutral/positive), 5-class ("Very Negative"...),
    and LABEL_0/1/2 heads; all three map onto the same three buckets here so the rest of
    the stage (and every consumer of utterances.sentiment) sees one vocabulary."""
    l = label.strip().lower().replace("_", " ")
    if l in ("label 0", "label 1", "label 2"):
        return {"label 0": "negative", "label 1": "neutral", "label 2": "positive"}[l]
    if "negative" in l:
        return "negative"
    if "positive" in l:
        return "positive"
    return "neutral"


def _signed(label: str, score: float) -> tuple[str, float]:
    """Model's argmax label + confidence -> (normalized label, signed valence in -1..1)."""
    label = _normalize(label)
    if label == "positive":
        return label, score
    if label == "negative":
        return label, -score
    return label, 0.0


def _label(score: float) -> str:
    if score > NEUTRAL_BAND:
        return "positive"
    if score < -NEUTRAL_BAND:
        return "negative"
    return "neutral"


def _classify_text(pool, texts: list[str]) -> list[tuple[str, float]]:
    import torch
    tok, model = pool.sentiment_tokenizer, pool.sentiment
    out = []
    # batched: one forward pass per 32 utterances instead of per utterance. On a clip with
    # 40 short radio calls that's the difference between ~40 GPU round-trips and 2.
    for i in range(0, len(texts), 32):
        batch = texts[i:i + 32]
        enc = tok(batch, return_tensors="pt", truncation=True, max_length=256,
                  padding=True).to(model.device)
        with torch.inference_mode():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        for row in probs:
            j = int(row.argmax())
            out.append((model.config.id2label[j].lower(), float(row[j])))
    return out


def _acoustic(ctx, u: dict, baseline) -> tuple[str | None, dict | None]:
    dur = u["end"] - u["start"]
    if dur < MIN_ACOUSTIC_S or ctx.audio_norm is None:
        return None, None
    seg = ctx.audio_norm[int(u["start"] * SR):int(u["end"] * SR)]
    if seg.size == 0:
        return None, None
    feats = tone_mod.extract_features(seg, SR, len(u["text"].split()), dur)
    mood = tone_mod.classify(feats, baseline)
    if mood == "calm":
        tone_mod.update_baseline(baseline, feats)
    return mood, feats


def _analyze(ctx, text_results):
    """Runs off the event loop — pyin over every utterance is the expensive half."""
    # one baseline per clip, so "stressed" means stressed *relative to this recording*
    # rather than against a fixed global pitch cutoff that varies by voice and mic
    baseline = tone_mod.new_baseline()
    rows = []
    for u, (raw_label, raw_score) in zip(ctx.utterances, text_results):
        tlabel, signed = _signed(raw_label, raw_score)
        mood, feats = _acoustic(ctx, u, baseline)
        fused = float(np.clip(signed + MOOD_SHIFT.get(mood, 0.0), -1.0, 1.0)) if mood else signed
        rows.append({
            "id": u["id"], "sentiment": _label(fused), "sentiment_score": round(fused, 3),
            "text_sentiment": tlabel, "text_score": round(signed, 3),
            "mood": mood, "mood_features": feats,
        })
    return rows


async def run(ctx):
    if not ctx.utterances:
        return
    if getattr(ctx.pool, "sentiment", None) is None:
        # model not present (never pulled, or an older MODEL_DIR) — the rest of the pipeline
        # is unaffected, so leave the columns null rather than failing the whole clip
        ctx.warn("SENTIMENT_MODEL_MISSING")
        return

    texts = [u["text"] for u in ctx.utterances]
    text_results = await asyncio.to_thread(_classify_text, ctx.pool, texts)
    rows = await asyncio.to_thread(_analyze, ctx, text_results)

    import json
    for r in rows:
        await db.execute(
            """UPDATE utterances SET sentiment=$2, sentiment_score=$3, text_sentiment=$4,
                   text_score=$5, mood=$6, mood_features=$7 WHERE id=$1""",
            r["id"], r["sentiment"], r["sentiment_score"], r["text_sentiment"],
            r["text_score"], r["mood"], json.dumps(r["mood_features"]) if r["mood_features"] else None)

    # clip rollup: duration-weighted, so a long angry utterance outweighs a one-word "ok".
    # Plain mean over utterances would let a burst of short acknowledgements drown out the
    # call that actually mattered.
    weights = [max(u["end"] - u["start"], 0.1) for u in ctx.utterances]
    score = float(np.average([r["sentiment_score"] for r in rows], weights=weights))
    moods = [r["mood"] for r in rows if r["mood"]]
    dominant = max(set(moods), key=moods.count) if moods else None
    await db.execute("UPDATE clips SET sentiment=$2, sentiment_score=$3, mood=$4 WHERE id=$1",
                      ctx.clip_id, _label(score), round(score, 3), dominant)
