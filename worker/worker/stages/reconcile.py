import numpy as np
from common import db


def nearest_turn_label(w, turns):
    if not turns:
        return "SPEAKER_00"
    return min(turns, key=lambda t: abs((t["start"] + t["end"]) / 2 - (w["start"] + w["end"]) / 2))["label"]


def assign_words(words, turns):
    for w in words:
        best, best_ov = None, 0.0
        for t in turns:
            if t["start"] >= w["end"]:
                break
            ov = min(w["end"], t["end"]) - max(w["start"], t["start"])
            if ov > best_ov:
                best, best_ov = t["label"], ov
        dur = max(w["end"] - w["start"], 1e-6)
        if best is None:
            best = nearest_turn_label(w, turns)
            w["speaker_conf"] = 0.0
        else:
            w["speaker_conf"] = min(1.0, best_ov / dur)
        w["local_label"] = best
        w["smoothed"] = False


def smooth(words, min_conf):
    for i in range(1, len(words) - 1):
        p, c, n = words[i - 1], words[i], words[i + 1]
        if (c["local_label"] != p["local_label"] and p["local_label"] == n["local_label"]
                and c["speaker_conf"] < min_conf):
            c["local_label"] = p["local_label"]
            c["smoothed"] = True


def finalize(u):
    ws = u["words"]
    return {**u, "end": ws[-1]["end"], "text": " ".join(w["word"] for w in ws).strip(),
            "mean_word_conf": float(np.mean([w["probability"] for w in ws])),
            "mean_speaker_conf": float(np.mean([w["speaker_conf"] for w in ws]))}


def regroup(words):
    utts, cur = [], None
    for w in words:
        boundary = cur is None or w["local_label"] != cur["local_label"]
        if not boundary and cur["words"]:
            prev = cur["words"][-1]
            gap = w["start"] - prev["end"]
            if prev["word"].endswith((".", "?", "!", "。", "؟")) and gap > 0.35:
                boundary = True
        if boundary:
            if cur:
                utts.append(finalize(cur))
            cur = {"local_label": w["local_label"], "start": w["start"], "words": []}
        cur["words"].append(w)
    if cur:
        utts.append(finalize(cur))
    for i, u in enumerate(utts):
        u["idx"] = i
    return utts


async def run(ctx):
    assign_words(ctx.words, ctx.turns)
    smooth(ctx.words, ctx.cfg.smooth_min_conf)
    ctx.utterances = regroup(ctx.words)

    smoothed_ratio = sum(1 for w in ctx.words if w["smoothed"]) / max(len(ctx.words), 1)
    if smoothed_ratio > 0.05:
        ctx.warn("HIGH_SMOOTH_RATE", ratio=smoothed_ratio)

    full_text = " ".join(u["text"] for u in ctx.utterances)
    avg_conf = float(np.mean([u["mean_word_conf"] for u in ctx.utterances])) if ctx.utterances else None
    ctx.transcript_id = await db.insert("transcripts", {
        "clip_id": ctx.clip_id, "run_id": ctx.run_id, "engine": "faster-whisper",
        "language": ctx.language, "language_conf": ctx.language_conf, "text": full_text,
    })

    for u in ctx.utterances:
        uid = await db.insert("utterances", {
            "transcript_id": ctx.transcript_id, "clip_id": ctx.clip_id, "idx": u["idx"],
            "start_s": u["start"], "end_s": u["end"], "local_label": u["local_label"],
            "text": u["text"], "mean_word_conf": u["mean_word_conf"],
            "mean_speaker_conf": u["mean_speaker_conf"],
        })
        u["id"] = uid
        for wi, w in enumerate(u["words"]):
            await db.execute(
                """INSERT INTO words (utterance_id, clip_id, idx, word, start_s, end_s,
                       probability, local_label, speaker_conf, smoothed)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)""",
                uid, ctx.clip_id, wi, w["word"], w["start"], w["end"],
                w["probability"], w["local_label"], w["speaker_conf"], w["smoothed"])
