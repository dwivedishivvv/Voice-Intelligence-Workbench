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
Fusion is deliberately a small, tunable nudge (MOOD_SHIFT) on top of the text score — the
text model is the trained one, and acoustics on a few seconds of radio-quality audio are the
noisier signal of the two.

Three corrections sit in front of the classifier, because a sentence model applied to radio
traffic one utterance at a time gets short turns badly wrong (measured on 223 F1 utterances:
a quarter were <=4 words, and half scored exactly 0.00):

  1. a protocol lexicon    — whole-utterance procedure words never reach the model
  2. short-turn merging    — consecutive same-speaker turns are judged as one thought
  3. a context window      — turns too short to judge alone are scored with their lead-in,
                             and the result is damped before it is attributed

None of this makes the model conversation-aware; it is a classifier over strings, and these
are the cheap corrections available in front of it. The real fix is an ERC (emotion
recognition in conversation) model that takes speaker-tagged turns natively.
"""
import asyncio
import re

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

# ── 1. protocol vocabulary ────────────────────────────────────────────────────
# Radio procedure words carry no emotion, but a general-purpose sentiment model has only
# ever seen them as ordinary English and scores them hard: "Negative." came back -0.60
# (it means "no"), "Head down." -0.70 (it means "keep pushing"), "Understood." +0.50.
# These were among the most confident scores in the whole corpus and every one was wrong.
#
# Only whole utterances are neutralized. "Negative." alone is procedure; "that's a negative
# result for us" is a real opinion that happens to contain the word, and must still score.
PROTOCOL_PHRASES = {
    "head down", "copy that", "stand by", "standby", "say again", "box box", "box box box",
    "keep pushing", "push push", "understood", "negative", "affirmative", "roger", "wilco",
    "confirm", "confirmed", "check", "checking", "no change", "as we were", "go ahead",
}
# If EVERY token of an utterance is in here, it is an acknowledgement, not a statement.
# Deliberately excludes anything with real valence ("thanks", "sorry", "great") — those are
# short too, but they genuinely carry sentiment and must keep scoring.
PROTOCOL_TOKENS = {
    "box", "copy", "negative", "affirmative", "roger", "wilco", "standby", "over", "out",
    "understood", "confirm", "confirmed", "check", "yes", "no", "yeah", "yep", "ok", "okay",
    "right", "alright", "that", "it", "is", "we", "are", "on", "in", "and", "a", "the",
    "mate", "mode", "lap", "laps", "one", "two", "three", "four", "five", "six", "seven",
}

# ── 2. short-turn merging ─────────────────────────────────────────────────────
# "Not yet," and "not yet." arrived as two separate 2-word utterances; each is meaningless
# alone. Consecutive turns from the SAME speaker close together are one thought, so they get
# classified together and share the result. Different speakers are never merged — that would
# attribute one person's words to another.
MERGE_UNDER_WORDS = 6      # keep absorbing the next turn while the group is still this short
MERGE_MAX_GAP_S = 2.0      # a longer pause than this is a new thought, not a continuation

# ── 3. context window ─────────────────────────────────────────────────────────
# Below this many words there is not enough text to judge alone — 25% of utterances in the
# F1 corpus, and the model punted half of everything to exactly 0.00.
SHORT_WORDS = 4
CONTEXT_TURNS = 2          # preceding turns (any speaker) shown to the model as lead-in

# Context is a genuine compromise and this constant is the honest part of it. The classifier
# is a *sentence* head: given "we have a problem / okay" it scores the whole string, not the
# "okay" in light of the problem. So a context-assisted score is partly someone else's words,
# and gets damped before it is attributed to this speaker. Without this, a neutral "Okay."
# following bad news would be reported as confidently negative *for the person who said ok*.
# Proper fix is a conversation-level (ERC) model that takes speaker-tagged turns natively.
CONTEXT_DAMPING = 0.6


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


_PUNCT = re.compile(r"[^\w\s']")


def _tokens(text: str) -> list[str]:
    return _PUNCT.sub(" ", text.lower()).split()


def _is_protocol(text: str) -> bool:
    """True when the whole utterance is radio procedure and carries no sentiment."""
    toks = _tokens(text)
    if not toks:
        return True
    if " ".join(toks) in PROTOCOL_PHRASES:
        return True
    # every token is an acknowledgement/filler word -> nothing was actually asserted
    return all(t in PROTOCOL_TOKENS for t in toks)


def _group(utterances: list) -> list[list[int]]:
    """Indices of utterances that should be classified as one unit.

    Consecutive turns from the same speaker, close in time, are absorbed while the group is
    still too short to judge on its own. Crossing a speaker boundary always starts a new
    group: merging there would score one person's words and file the result under another's.
    """
    groups: list[list[int]] = []
    for i, u in enumerate(utterances):
        # A procedure word is already resolved as neutral and must never merge in either
        # direction. Letting it absorb the sentence after it made "Negative." inherit that
        # sentence's -0.60 — the exact false positive the lexicon exists to remove.
        if _is_protocol(u["text"]):
            groups.append([i])
            continue
        if groups:
            g = groups[-1]
            prev = utterances[g[-1]]
            words = sum(len(utterances[j]["text"].split()) for j in g)
            if (not _is_protocol(prev["text"])
                    and u.get("local_label") == prev.get("local_label")
                    and u["start"] - prev["end"] <= MERGE_MAX_GAP_S
                    and words < MERGE_UNDER_WORDS):
                g.append(i)
                continue
        groups.append([i])
    return groups


def _context_prefix(utterances: list, first_idx: int) -> str:
    lead = [utterances[j]["text"].strip()
            for j in range(max(0, first_idx - CONTEXT_TURNS), first_idx)]
    return " ".join(p for p in lead if p)


def _group_results(pool, utterances: list, groups: list[list[int]]) -> list[tuple[str, float, bool]]:
    """One (label, signed score, used_context) per group. Protocol groups never reach the
    model at all — that is the point of the lexicon, and it also saves the forward pass."""
    results: list[tuple[str, float, bool]] = [("neutral", 0.0, False)] * len(groups)
    pending: list[tuple[int, bool]] = []
    texts: list[str] = []

    for gi, g in enumerate(groups):
        own = " ".join(utterances[i]["text"].strip() for i in g).strip()
        # Check the members too, not just the joined text. Merging two procedure calls
        # ("Negative." + "Head down.") builds a string that matches neither the phrase list
        # nor the all-tokens rule, and it went to the model and came back -0.52. A group of
        # procedure words is still procedure however many of them there are.
        if _is_protocol(own) or all(_is_protocol(utterances[i]["text"]) for i in g):
            continue                                    # stays the neutral default
        text, used_context = own, False
        if len(own.split()) < SHORT_WORDS:
            pre = _context_prefix(utterances, g[0])
            if pre:
                text, used_context = f"{pre} {own}", True
        pending.append((gi, used_context))
        texts.append(text)

    if texts:
        for (gi, used_context), (raw_label, raw_score) in zip(pending, _classify_text(pool, texts)):
            label, signed = _signed(raw_label, raw_score)
            if used_context:
                # part of this score came from the preceding speaker's words, so it is
                # damped and the label re-derived — a damped score that lands inside the
                # neutral band must read as neutral, not keep the model's raw verdict.
                signed *= CONTEXT_DAMPING
                label = _label(signed)
            results[gi] = (label, signed, used_context)
    return results


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


def _analyze(ctx, groups, group_results):
    """Runs off the event loop — pyin over every utterance is the expensive half."""
    # one baseline per clip, so "stressed" means stressed *relative to this recording*
    # rather than against a fixed global pitch cutoff that varies by voice and mic
    baseline = tone_mod.new_baseline()
    # every utterance in a merged group shares that group's text verdict; the acoustic read
    # below stays per-utterance, since delivery genuinely varies within one thought
    per_utterance = [None] * len(ctx.utterances)
    for g, res in zip(groups, group_results):
        for i in g:
            per_utterance[i] = res

    rows = []
    for u, (tlabel, signed, _used_context) in zip(ctx.utterances, per_utterance):
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

    groups = _group(ctx.utterances)
    group_results = await asyncio.to_thread(_group_results, ctx.pool, ctx.utterances, groups)
    rows = await asyncio.to_thread(_analyze, ctx, groups, group_results)

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
