import asyncio
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from common.speaker import reliability_score, clamp01
from ..audio.quality import rolloff_99
from ..ctx import SR, SpeakerResult

# A label needs at least this much speech before splitting it in half says anything: each
# half still has to be long enough to produce an embedding worth comparing.
SPLIT_HALF_MIN_S = 3.0
# Below this, the two halves of one label's audio disagree more than one person's voice
# should with itself, which is what a label holding *two* speakers looks like. Not a hard
# reject — diarization is usually right, and a hard reject on a soft signal would throw
# away good speakers — but it discounts reliability, which identify() then turns into a
# raised threshold. The failure mode this catches is the expensive one: a blended
# embedding can sit plausibly close to a third profile and get a confident wrong name.
SPLIT_HALF_SUSPECT = 0.60

# Longest single turn used whole for a voiceprint. An embedding stops improving well before
# this (embed_target_s is 8s), while the cost keeps growing: one 336s monologue was fed to
# ResNet293 whole, and the EMBEDDING stage of that 8-minute clip took 50s.
MAX_SEGMENT_S = 30.0


def _noise_power(audio: np.ndarray, vad: list) -> float:
    """Mean power of everything VAD called non-speech, as the noise floor to measure each
    speaker's own segments against. Clip-wide on purpose: a per-speaker noise estimate
    would have to come from that speaker's silence, which is exactly what diarization
    assigned to somebody else."""
    if audio is None or not len(audio):
        return 1e-12
    mask = np.zeros(len(audio), dtype=bool)
    for s, e in vad:
        mask[int(s * SR):int(e * SR)] = True
    ns = audio[~mask]
    return float(np.mean(ns ** 2)) if ns.size > SR * 0.1 else 1e-12


def _snr_db(seg: np.ndarray, noise_power: float) -> float:
    p = float(np.mean(seg ** 2)) if seg.size else 1e-12
    return float(np.clip(10 * np.log10(max(p, 1e-12) / max(noise_power, 1e-12)), -20, 60))


def segment_quality(seg: np.ndarray, noise_power: float, cfg) -> float:
    """0..1 quality of one candidate turn — the same SNR and clipping terms
    reliability_score uses, so a turn is ranked by the thing it will later be judged on.
    Bandwidth is deliberately absent: it is a property of the channel, identical across
    every turn in a clip, so it can only add a constant here."""
    if not seg.size:
        return 0.0
    q = clamp01((_snr_db(seg, noise_power) - 5.0) / 20.0)
    clipped = float(np.mean(np.abs(seg) >= cfg.clipping_threshold))
    c = 1.0 - min(1.0, clipped * 20.0)
    return 0.6 * q + 0.4 * c


def select_segments(audio_norm, turns, cfg, noise_power: float, min_s=None):
    """This label's best audio for fingerprinting, longest-and-cleanest first.

    Ranked by duration *scaled* by quality rather than duration alone. Duration still
    dominates — it is what actually stabilizes an embedding, and the old pure-duration
    sort was right about that — but between two comparable turns the clipped or noisy one
    now loses to the clean one instead of winning on length alone.

    Overlapped turns are excluded (two voices in one embedding is the thing the whole
    stage is trying to avoid), and so are turns diarization kept only for word attribution
    (see stages/diarize.py `too_short`): they are real speech and belong in the transcript,
    but they are too short to fingerprint from.
    """
    floor = cfg.embed_min_s if min_s is None else min_s
    usable = [t for t in turns if not t["is_overlap"] and not t.get("too_short")]

    scored = []
    cap = int(MAX_SEGMENT_S * SR)
    for t in usable:
        a, b = int(t["start"] * SR), int(t["end"] * SR)
        seg = audio_norm[a:b]
        if not seg.size:
            continue
        if seg.size > cap:
            # the middle of a very long turn: away from the edges where the diarizer's
            # boundary is least sure who is talking
            mid = seg.size // 2
            seg = seg[mid - cap // 2:mid - cap // 2 + cap]
        dur = seg.size / SR
        scored.append((dur * (0.5 + 0.5 * segment_quality(seg, noise_power, cfg)), seg))
    scored.sort(key=lambda x: x[0], reverse=True)

    chunks, total = [], 0.0
    for _rank, seg in scored:
        chunks.append(seg)
        total += seg.size / SR
        if total >= cfg.embed_target_s:
            break
    return (chunks, total) if total >= floor else ([], total)


def _split_halves(chunks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """Two disjoint halves of the selected audio, balanced by duration.

    Split across the selected turns rather than by cutting the concatenation in the
    middle: a mid-sample cut would put both halves inside the same few turns and the two
    embeddings would agree for reasons that have nothing to do with speaker identity.
    """
    total = sum(c.size for c in chunks)
    if total < SPLIT_HALF_MIN_S * SR or len(chunks) < 2:
        return None
    a, b, acc = [], [], 0
    for c in chunks:
        (a if acc < total / 2 else b).append(c)
        acc += c.size
    if not a or not b:
        return None
    return np.concatenate(a), np.concatenate(b)


# Padded audio per encoder forward pass. Every row in a batch is padded to the longest one,
# and select_segments can hand back a single uninterrupted turn of several minutes: an
# 8-minute clip with a 336s monologue padded all twelve rows to 336s — about an hour of
# audio in one pass — and ran a 6GB GPU out of memory in this stage.
EMBED_BATCH_BUDGET_S = 120.0


def _encode_padded(pool, segs: list[np.ndarray]) -> np.ndarray:
    lens = [int(s.size) for s in segs]
    width = max(lens)
    batch = np.zeros((len(segs), width), dtype=np.float32)
    for i, s in enumerate(segs):
        batch[i, :s.size] = s
    with torch.no_grad():
        out = pool.embedder.encode_batch(
            torch.from_numpy(batch),
            wav_lens=torch.tensor([l / width for l in lens], dtype=torch.float32))
    embs = out.reshape(len(segs), -1)
    return F.normalize(embs, dim=-1).cpu().numpy().astype(np.float32)


def encode_many(pool, segs: list[np.ndarray], budget_s: float = EMBED_BATCH_BUDGET_S) -> np.ndarray:
    """Embeddings for many audio segments, batched padded forward passes.

    Every speaker's full audio and both of its halves go through together where they fit,
    so a typical clip still costs one GPU round trip instead of one per row. `wav_lens`
    carries the true length of each row so the encoder's statistics pooling ignores the
    padding — without it the padded tail would be averaged in as if it were silence the
    speaker produced.

    Rows are grouped shortest-first under a padded-audio budget, so one long row no longer
    drags every other row out to its length; a row longer than the budget runs alone. Same
    rows, same weights, same results — only how many share a pass changes. Should a batch
    still not fit, its rows are retried one at a time before giving up.
    """
    budget = int(budget_s * SR)
    order = sorted(range(len(segs)), key=lambda i: segs[i].size)
    groups: list[list[int]] = []
    for i in order:
        # ascending order, so this row is the widest the group would have
        if groups and (len(groups[-1]) + 1) * segs[i].size <= budget:
            groups[-1].append(i)
        else:
            groups.append([i])

    out: list[np.ndarray | None] = [None] * len(segs)
    for g in groups:
        g = sorted(g)  # length only decides the grouping; rows keep their caller's order
        try:
            embs = _encode_padded(pool, [segs[i] for i in g])
        except torch.OutOfMemoryError:
            if len(g) == 1:
                raise
            torch.cuda.empty_cache()
            embs = np.concatenate([_encode_padded(pool, [segs[i]]) for i in g])
        for i, e in zip(g, embs):
            out[i] = e
    return np.stack(out)


def extract(audio_norm, turns, cfg, pool, min_s=None, noise_power: float | None = None):
    """Kept for callers that want one label's embedding on its own. run() uses the batched
    path instead, which is the same computation for every label at once."""
    np_ = _noise_power(audio_norm, []) if noise_power is None else noise_power
    chunks, total = select_segments(audio_norm, turns, cfg, np_, min_s)
    if not chunks:
        return None, total, 0
    return encode_many(pool, [np.concatenate(chunks)])[0], total, len(chunks)


def compute_speakers(audio_norm, turns_all, vad, cfg, pool, warn) -> dict[str, SpeakerResult]:
    """One voiceprint (plus its reliability and split-half check) per diarization label.

    Separate from run() so worker/reembed.py can rebuild stored voiceprints from a clip's
    saved audio and turns with exactly this logic when the embedding model changes — two
    copies of segment selection would drift, and the migrated vectors would then mean
    something slightly different from the ones every new clip gets.
    """
    by_label = defaultdict(list)
    for t in turns_all:
        by_label[t["label"]].append(t)

    noise_power = _noise_power(audio_norm, vad)

    # Everything the encoder has to see, gathered before it is called once. `plan` maps
    # each label to the rows it owns in the batch.
    segs: list[np.ndarray] = []
    plan: dict[str, dict] = {}
    for label, turns in by_label.items():
        chunks, total = select_segments(audio_norm, turns, cfg, noise_power)
        entry = {"total": total, "n_seg": len(chunks), "full": None, "halves": None,
                 "quality": None}
        if chunks:
            selected = np.concatenate(chunks)
            entry["quality"] = {
                "snr_db": _snr_db(selected, noise_power),
                "clipping_ratio": float(np.mean(np.abs(selected) >= cfg.clipping_threshold)),
                "bandwidth_hz": rolloff_99(selected, SR),
            }
            entry["full"] = len(segs)
            segs.append(selected)
            halves = _split_halves(chunks)
            if halves is not None:
                entry["halves"] = (len(segs), len(segs) + 1)
                segs.extend(halves)
        plan[label] = entry

    embs = encode_many(pool, segs) if segs else np.empty((0, 0), dtype=np.float32)

    speakers: dict[str, SpeakerResult] = {}
    for label, entry in plan.items():
        turns = by_label[label]
        emb = embs[entry["full"]] if entry["full"] is not None else None
        total, n_seg = entry["total"], entry["n_seg"]

        split_sim = None
        if emb is not None and entry["halves"] is not None:
            i, j = entry["halves"]
            split_sim = float(np.dot(embs[i], embs[j]))

        # Per-speaker quality, not the clip's. reliability_score used to be handed
        # ctx.quality — one SNR/clipping/bandwidth reading for the whole recording — so a
        # close-mic speaker and a distant noisy one in the same clip came out equally
        # reliable, and the difference that matters most to identification was the one
        # thing the score could not see.
        rel = reliability_score(total, entry["quality"] or {}, cfg) if emb is not None else 0.0
        reason = None if emb is not None else "insufficient_speech"
        if split_sim is not None and split_sim < SPLIT_HALF_SUSPECT:
            # halves of one voice should agree; scale reliability by how much they didn't
            rel *= clamp01(split_sim / SPLIT_HALF_SUSPECT)
            reason = "split_half_disagreement"
            warn("MIXED_LABEL_SUSPECTED", label=label, split_half_sim=round(split_sim, 3))

        speakers[label] = SpeakerResult(
            local_label=label, embedding=emb, speech_s=total, n_turns=len(turns),
            n_segments_used=n_seg, reliability=rel, reliability_reason=reason,
            split_half_sim=split_sim)
        if emb is None:
            warn("SPEAKER_TOO_SHORT", label=label, duration_s=total)
        elif rel < cfg.reliability_fair:
            warn("LOW_RELIABILITY_EMBEDDING", label=label, reliability=rel)
    return speakers


async def run(ctx):
    # off the event loop: the encoder forward passes are the slow part of this stage
    ctx.speakers.update(await asyncio.to_thread(
        compute_speakers, ctx.audio_norm, ctx.turns, ctx.vad, ctx.cfg, ctx.pool, ctx.warn))
