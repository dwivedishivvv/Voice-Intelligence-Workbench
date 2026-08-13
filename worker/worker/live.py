"""Per-chunk diarization + speaker identification for live mode — the live analog of the
batch pipeline's diarize -> embed -> identify chain (see stages/diarize.py, stages/embed.py,
stages/identify.py), condensed to run on one short chunk instead of a whole clip.

Real enrolled profiles are still matched via common.speaker.identify() same as batch. But
batch clips get a persisted speaker_clusters row to fall back on for an unenrolled voice;
a live session has no such thing, so unmatched speakers get an in-memory, session-scoped
running centroid instead (see _match_session_speaker) — stable within one live session
("Speaker 1", "Speaker 2", ...), reset when the session ends. Good enough to tell two
people apart live; won't survive across separate sessions the way a real profile does.
"""
import numpy as np
import torch

from common import db, speaker as sc
from .ctx import SR
from .audio.vad import detect_vad
from .audio.quality import compute_quality
from .stages.diarize import clean_turns
from .stages.embed import extract as embed_extract
from .stages.reconcile import assign_words, smooth, regroup

# batch mode's cluster_threshold (0.76) is tuned for embeddings pooled from up to
# embed_target_s=8s of a person's cleanest turns across a whole clip. A single live chunk
# usually can't offer that much of one speaker, so its embeddings are noisier — matched
# 1:1 against cluster_threshold, the same person's own voice was falling below it chunk to
# chunk, fragmenting into a new "Speaker N" almost every time instead of being recognized
# as someone already seen this session. Live-scoped matching gets its own, more forgiving bar.
LIVE_MATCH_THRESHOLD = 0.60

# batch's embed_min_s (1.5s) exists to keep genuinely-unusable slivers out of enrollment
# data, but for a read-only live *match* it was blocking the attempt entirely on anything
# shorter — a one-word "Hello?" or "1, 2, 3" chunk (well under 1.5s of actual speech) never
# even got an embedding, so it could never be recognized no matter who enrolled. Lower floor
# for live: reliability_score already penalizes a short/noisy embedding on its own, and
# identify() scales its match threshold by that reliability, so this doesn't loosen the
# actual matching decision — it just lets short speech attempt one, judged accordingly.
# There's a real floor beneath which no algorithm reliably fingerprints a voice (a fraction
# of a syllable), and this is a guess at where that floor sits, not a measured one.
LIVE_EMBED_MIN_S = 0.5

# label for a turn too short/unclear to embed at all — distinct from a real "Speaker N" so
# it's never treated as a known, trackable identity (deliberately not pyannote's own raw
# per-chunk label, which looks stable across chunks by coincidence but isn't — it resets
# arbitrarily every diarization call and was being displayed as if it tracked one person)
UNCLEAR_LABEL = "Speaker (unclear)"


def _match_session_speaker(embedding: np.ndarray, session_speakers: list, threshold: float = LIVE_MATCH_THRESHOLD) -> str:
    best_i, best_sim = None, -1.0
    for i, s in enumerate(session_speakers):
        sim = float(np.dot(embedding, s["centroid"]))
        if sim > best_sim:
            best_i, best_sim = i, sim
    if best_i is not None and best_sim >= threshold:
        s = session_speakers[best_i]
        s["centroid"] = (s["centroid"] * s["n"] + embedding) / (s["n"] + 1)
        s["centroid"] /= max(float(np.linalg.norm(s["centroid"])), 1e-6)
        s["n"] += 1
        return s["label"]
    label = f"Speaker {len(session_speakers) + 1}"
    session_speakers.append({"label": label, "centroid": embedding.copy(), "n": 1})
    return label


async def diarize_and_label(pool, cfg, audio: np.ndarray, words: list, session_speakers: list) -> list:
    """words: ASR word dicts (word, start, end, probability), already hallucination-filtered.
    session_speakers: this session's running local-speaker list, mutated in place.
    Returns a list of {label, is_known, start, end, text} segments — empty list if
    diarization found nothing usable (caller should fall back to an unsegmented transcript)."""
    vad = detect_vad(audio, SR, pool.vad, cfg)
    if not vad:
        return []

    ann = pool.diar({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SR},
                     min_speakers=cfg.diar_min_speakers, max_speakers=cfg.diar_max_speakers)
    raw = [{"start": seg.start, "end": seg.end, "label": lbl} for seg, _, lbl in ann.itertracks(yield_label=True)]
    turns = clean_turns(raw, vad, cfg)
    if not turns:
        return []

    clip_ratio = float(np.mean(np.abs(audio) >= cfg.clipping_threshold))
    quality = compute_quality(audio, vad, SR, clip_ratio, None, cfg)

    by_label = {}
    for t in turns:
        by_label.setdefault(t["label"], []).append(t)

    resolved = {}
    for label, label_turns in by_label.items():
        emb, total, _ = embed_extract(audio, label_turns, cfg, pool, min_s=LIVE_EMBED_MIN_S)
        if emb is None:
            resolved[label] = (UNCLEAR_LABEL, False)  # too little audio for this speaker to embed at all
            continue
        reliability = sc.reliability_score(total, quality, cfg)
        # no reliability gate here before attempting identify() — sc.identify() already
        # scales its own match threshold by reliability internally (stricter when lower,
        # not skipped), matching how the batch pipeline calls it. Gating the attempt on
        # top of that just meant a real enrolled match was never even tried for anything
        # short of a live chunk, falling straight to a fresh session-local label instead.
        m = await sc.identify(emb, reliability, cfg)
        display, is_known = None, False
        if m.result in ("confident", "suggested") and m.profile_id:
            display = await db.fetchval("SELECT display_name FROM speaker_profiles WHERE id=$1", m.profile_id)
            is_known = display is not None
        if display is None:
            # minting a brand-new persistent "Speaker N" on a low-reliability embedding is
            # how one noisy/uncertain moment (echo, a stray click, background noise briefly
            # reading as "voiced") turns into a permanent fake extra person for the rest of
            # the session. A confident-enough embedding that genuinely matches no one so far
            # is trusted as a real new speaker; a shaky one is left unclear instead — it can
            # still resolve later once a cleaner chunk of that same voice comes through.
            display = _match_session_speaker(emb, session_speakers) if reliability >= cfg.reliability_fair else UNCLEAR_LABEL
        resolved[label] = (display, is_known)

    for t in turns:
        t["label"] = resolved.get(t["label"], (t["label"], False))[0]

    assign_words(words, turns)
    smooth(words, cfg.smooth_min_conf)
    known_labels = {v[0] for v in resolved.values() if v[1]}
    return [{"label": u["local_label"], "is_known": u["local_label"] in known_labels,
             "start": u["start"], "end": u["end"], "text": u["text"]}
            for u in regroup(words)]
