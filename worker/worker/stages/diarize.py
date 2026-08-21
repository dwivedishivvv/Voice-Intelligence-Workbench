from common import db
from ..ctx import SR
from ..errors import RejectError


def merge_adjacent(turns, max_gap):
    out = []
    for t in turns:
        if out and out[-1]["label"] == t["label"] and t["start"] - out[-1]["end"] <= max_gap:
            out[-1]["end"] = max(out[-1]["end"], t["end"])
        else:
            out.append(dict(t))
    return out


def drop_onset_phantoms(turns, start_tol=0.3, dur_ratio=0.6):
    # pyannote sometimes emits a brief, confidently-labeled conflicting-speaker turn at
    # almost exactly the same instant a real (longer) turn from someone else begins —
    # not genuine cross-talk, just the clustering being unsettled in the first fraction
    # of a second before it locks onto the actual speaker. Left in, word-to-turn
    # reconciliation (reconcile.assign_words) picks whichever turn overlaps more per
    # word, splitting one continuous sentence across two speaker labels right at the
    # boundary — high per-word confidence either way, so smoothing can't catch it.
    # A turn that starts within `start_tol`s of a much longer different-speaker turn's
    # own start is almost certainly this artifact, not real simultaneous speech.
    #
    # Only the window of turns starting within `start_tol` is scanned, not every pair:
    # the input is sorted by start, so anything outside that window fails the
    # |a.start - b.start| test by construction. The all-pairs form has no early exit, and
    # O(n^2) stopped being free when the clip ceiling went from 90s to an hour — an hour
    # of conversation is thousands of turns, not dozens.
    n = len(turns)
    drop = set()
    lo = 0
    for i, a in enumerate(turns):
        while turns[lo]["start"] < a["start"] - start_tol:
            lo += 1
        a_dur = a["end"] - a["start"]
        j = lo
        while j < n and turns[j]["start"] <= a["start"] + start_tol:
            b = turns[j]
            if j != i and b["label"] != a["label"]:
                b_dur = b["end"] - b["start"]
                if b_dur > a_dur and a_dur < dur_ratio * b_dur:
                    drop.add(i)
                    break
            j += 1
    return [t for i, t in enumerate(turns) if i not in drop]


def annotate_overlaps(turns):
    for t in turns:
        t["is_overlap"] = False
    for i, a in enumerate(turns):
        for b in turns[i + 1:]:
            if b["start"] >= a["end"]:
                break
            if b["label"] != a["label"] and min(a["end"], b["end"]) > max(a["start"], b["start"]):
                a["is_overlap"] = b["is_overlap"] = True
    return turns


def snap_to_vad(turns, vad, tol):
    edges = sorted({e for r in vad for e in r})

    def nearest(x):
        if not edges:
            return x, False
        c = min(edges, key=lambda e: abs(e - x))
        return (c, True) if abs(c - x) <= tol else (x, False)

    for t in turns:
        t["start"], s1 = nearest(t["start"])
        t["end"], s2 = nearest(t["end"])
        t["snapped"] = s1 or s2
        if t["end"] <= t["start"]:
            t["end"] = t["start"] + 0.05
    return turns


def clean_turns(raw, vad, cfg):
    """Cleaned turns, each marked `too_short` rather than silently deleted when it is.

    Dropping a sub-`min_turn_s` turn outright conflated two different judgements: "too
    short to fingerprint from" (true) and "too short to have happened" (false). The words
    inside a dropped turn do not disappear with it — reconcile.assign_words simply gives
    them to whichever *surviving* turn overlaps most, so a second speaker's one-word
    interjection ("Copy", "Yes") was systematically attributed to the person talking
    around it. They are kept here and excluded at the point where the short-audio problem
    is real instead (stages/embed.py select_segments).

    The long turns are still cleaned exactly as before, on their own: merge_adjacent walks
    consecutive entries and stops merging across a label change, so leaving a short
    interjection in the list would silently prevent the two same-speaker turns on either
    side of it from joining. Overlap is likewise annotated among the long turns only —
    letting a 0.3s "yeah" mark an 8s turn as overlapped would disqualify that speaker's
    best audio from being fingerprinted at all.
    """
    t = sorted(raw, key=lambda x: x["start"])
    long_turns = [x for x in t if x["end"] - x["start"] >= cfg.min_turn_s]
    short_turns = [x for x in t if x["end"] - x["start"] < cfg.min_turn_s]

    long_turns = drop_onset_phantoms(long_turns)
    long_turns = merge_adjacent(long_turns, cfg.merge_gap_s)
    long_turns = annotate_overlaps(long_turns)
    long_turns = snap_to_vad(long_turns, vad, cfg.vad_snap_tol_s)
    long_turns = [x for x in long_turns if x["end"] - x["start"] >= cfg.min_turn_s]
    for x in long_turns:
        x["too_short"] = False

    kept_short = []
    for x in short_turns:
        y = dict(x)
        y["too_short"] = True
        y["snapped"] = False
        # a brief turn landing inside another speaker's is what an interjection *is*, so
        # flag it — but only on itself, never back onto the turn it interrupted
        y["is_overlap"] = any(
            o["label"] != y["label"] and min(o["end"], y["end"]) > max(o["start"], y["start"])
            for o in long_turns)
        kept_short.append(y)

    out = sorted(long_turns + kept_short, key=lambda x: x["start"])
    for i, x in enumerate(out):
        x["idx"] = i
    return out


async def run(ctx):
    import torch
    kwargs = {"min_speakers": ctx.cfg.diar_min_speakers, "max_speakers": ctx.cfg.diar_max_speakers}
    ann = ctx.pool.diar(
        {"waveform": torch.from_numpy(ctx.audio_clean).unsqueeze(0), "sample_rate": SR}, **kwargs)

    raw = [{"start": seg.start, "end": seg.end, "label": lbl}
           for seg, _, lbl in ann.itertracks(yield_label=True)]

    turns = clean_turns(raw, ctx.vad, ctx.cfg)
    if not turns:
        raise RejectError("NO_SPEAKER_TURNS", "diarization produced no usable turns")

    ctx.turns = turns
    labels = {t["label"] for t in turns}

    overlap_s = sum(t["end"] - t["start"] for t in turns if t["is_overlap"])
    if ctx.speech_s and overlap_s / ctx.speech_s > ctx.cfg.overlap_warn_ratio:
        ctx.warn("HIGH_OVERLAP", ratio=overlap_s / ctx.speech_s)
    if len(labels) >= ctx.cfg.diar_max_speakers:
        ctx.warn("SPEAKER_COUNT_AT_CEILING", n=len(labels))

    # one round trip for the whole clip rather than one per turn — an hour of conversation
    # is thousands of turns, and each await was a separate sequential query
    await db.pool().executemany(
        """INSERT INTO speaker_turns (clip_id, idx, local_label, start_s, end_s, is_overlap, snapped)
           VALUES ($1,$2,$3,$4,$5,$6,$7)""",
        [(ctx.clip_id, t["idx"], t["label"], t["start"], t["end"], t["is_overlap"], t["snapped"])
         for t in turns])
