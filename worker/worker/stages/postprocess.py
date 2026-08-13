from common import db
from ..export import srt, vtt, rttm, txt as txt_export, json_export


def count_interruptions(label, turns):
    n = 0
    for t in turns:
        if t["label"] != label:
            continue
        for o in turns:
            if o["label"] != label and o["start"] < t["start"] < o["end"]:
                n += 1
                break
    return n


async def run(ctx):
    total = sum(sp.speech_s for sp in ctx.speakers.values()) or 1.0

    for label, sp in ctx.speakers.items():
        turns = [t for t in ctx.turns if t["label"] == label]
        sp.talk_share = sp.speech_s / total
        sp.longest_turn_s = max((t["end"] - t["start"] for t in turns), default=0)
        sp.overlap_s = sum(t["end"] - t["start"] for t in turns if t["is_overlap"])
        sp.interruptions = count_interruptions(label, ctx.turns)

        await db.execute(
            """INSERT INTO clip_speakers (clip_id, local_label, embedding, speech_s, n_turns,
                   n_segments_used, reliability, reliability_reason, profile_id, cluster_id,
                   match_score, match_margin, match_result, runner_up_id, runner_up_score,
                   talk_share, longest_turn_s, interruptions, overlap_s)
               VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19)""",
            ctx.clip_id, label, list(map(float, sp.embedding)) if sp.embedding is not None else None,
            sp.speech_s, sp.n_turns, sp.n_segments_used, sp.reliability, sp.reliability_reason,
            sp.profile_id, sp.cluster_id, sp.match_score, sp.match_margin, sp.match_result,
            sp.runner_up_id, sp.runner_up_score, sp.talk_share, sp.longest_turn_s,
            sp.interruptions, sp.overlap_s)

    for u in ctx.utterances:
        sp = ctx.speakers.get(u["local_label"])
        if sp:
            await db.execute("UPDATE utterances SET profile_id=$2, cluster_id=$3 WHERE id=$1",
                              u["id"], sp.profile_id, sp.cluster_id)

    ctx.needs_review = any(
        sp.match_result in ("suggested", "abstained") or (sp.reliability or 0) < ctx.cfg.reliability_fair
        for sp in ctx.speakers.values()
    ) or any(w["code"] in ("HIGH_OVERLAP", "POOR_AUDIO_QUALITY") for w in ctx.warnings)

    await db.execute(
        "UPDATE clips SET n_speakers=$2, language=$3, language_conf=$4, needs_review=$5 WHERE id=$1",
        ctx.clip_id, len(ctx.speakers), ctx.language, ctx.language_conf, ctx.needs_review)

    speaker_names = {label: label for label in ctx.speakers}  # local labels; UI resolves display names
    rttm.write(ctx.cfg, ctx.clip_id, ctx.turns, speaker_names)
    srt.write(ctx.cfg, ctx.clip_id, ctx.utterances)
    vtt.write(ctx.cfg, ctx.clip_id, ctx.utterances)
    txt_export.write(ctx.cfg, ctx.clip_id, ctx.utterances)
    json_export.write(ctx.cfg, ctx.clip_id, ctx)
