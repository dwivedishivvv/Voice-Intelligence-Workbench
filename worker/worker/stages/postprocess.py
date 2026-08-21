from common import db
from ..export import srt, vtt, rttm, txt as txt_export, json_export


def count_interruptions_all(turns) -> dict[str, int]:
    """Per label, how many of its turns begin inside another speaker's turn.

    One pass in start order carrying the furthest end seen per label, rather than checking
    every turn against every other turn for every label. The pairwise form is O(n^2) per
    label, which was invisible while clips were capped at 90s and is not now that the
    ceiling is an hour — a busy hour of conversation runs to thousands of turns.

    Turns sharing an identical start are resolved as a group so neither can count as
    having interrupted the other: snap_to_vad can legitimately pull two turns onto the
    same VAD edge, and the pairwise version this replaces required a strictly earlier
    start.
    """
    ordered = sorted(turns, key=lambda t: t["start"])
    counts = {t["label"]: 0 for t in ordered}
    max_end: dict[str, float] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j]["start"] == ordered[i]["start"]:
            j += 1
        group = ordered[i:j]
        for t in group:
            if any(e > t["start"] for lbl, e in max_end.items() if lbl != t["label"]):
                counts[t["label"]] += 1
        for t in group:
            max_end[t["label"]] = max(max_end.get(t["label"], float("-inf")), t["end"])
        i = j
    return counts


CLIP_SPEAKER_SQL = """
INSERT INTO clip_speakers (clip_id, local_label, embedding, speech_s, n_turns,
       n_segments_used, reliability, reliability_reason, split_half_sim, profile_id,
       cluster_id, match_score, match_margin, match_result, match_z, runner_up_id,
       runner_up_score, talk_share, longest_turn_s, interruptions, overlap_s)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21)
"""


async def run(ctx):
    total = sum(sp.speech_s for sp in ctx.speakers.values()) or 1.0
    turns_by_label: dict[str, list] = {}
    for t in ctx.turns:
        turns_by_label.setdefault(t["label"], []).append(t)
    interruptions = count_interruptions_all(ctx.turns)

    rows = []
    for label, sp in ctx.speakers.items():
        turns = turns_by_label.get(label, [])
        sp.talk_share = sp.speech_s / total
        sp.longest_turn_s = max((t["end"] - t["start"] for t in turns), default=0)
        sp.overlap_s = sum(t["end"] - t["start"] for t in turns if t["is_overlap"])
        sp.interruptions = interruptions.get(label, 0)
        rows.append((
            ctx.clip_id, label,
            list(map(float, sp.embedding)) if sp.embedding is not None else None,
            sp.speech_s, sp.n_turns, sp.n_segments_used, sp.reliability,
            sp.reliability_reason, sp.split_half_sim, sp.profile_id, sp.cluster_id,
            sp.match_score, sp.match_margin, sp.match_result, sp.match_z, sp.runner_up_id,
            sp.runner_up_score, sp.talk_share, sp.longest_turn_s, sp.interruptions,
            sp.overlap_s))

    # one round trip for every speaker, and one for every utterance, instead of one per
    # row — on a long clip the utterance loop alone was hundreds of sequential awaits
    if rows:
        await db.pool().executemany(CLIP_SPEAKER_SQL, rows)

    utt_rows = [
        (u["id"], sp.profile_id, sp.cluster_id, u.get("overlap_ratio"))
        for u in ctx.utterances
        if (sp := ctx.speakers.get(u["local_label"])) is not None
    ]
    if utt_rows:
        await db.pool().executemany(
            "UPDATE utterances SET profile_id=$2, cluster_id=$3, overlap_ratio=$4 WHERE id=$1",
            utt_rows)

    ctx.needs_review = any(
        sp.match_result in ("suggested", "abstained") or (sp.reliability or 0) < ctx.cfg.reliability_fair
        for sp in ctx.speakers.values()
    ) or any(w["code"] in ("HIGH_OVERLAP", "POOR_AUDIO_QUALITY", "MIXED_LABEL_SUSPECTED")
             for w in ctx.warnings)

    await db.execute(
        "UPDATE clips SET n_speakers=$2, language=$3, language_conf=$4, needs_review=$5 WHERE id=$1",
        ctx.clip_id, len(ctx.speakers), ctx.language, ctx.language_conf, ctx.needs_review)

    speaker_names = {label: label for label in ctx.speakers}  # local labels; UI resolves display names
    rttm.write(ctx.cfg, ctx.clip_id, ctx.turns, speaker_names)
    srt.write(ctx.cfg, ctx.clip_id, ctx.utterances)
    vtt.write(ctx.cfg, ctx.clip_id, ctx.utterances)
    txt_export.write(ctx.cfg, ctx.clip_id, ctx.utterances)
    json_export.write(ctx.cfg, ctx.clip_id, ctx)
