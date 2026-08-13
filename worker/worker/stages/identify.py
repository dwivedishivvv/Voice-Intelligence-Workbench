from common import db, speaker as sc


async def run(ctx):
    for label, sp in ctx.speakers.items():
        if sp.embedding is None:
            sp.match_result = "abstained"
            continue

        m = await sc.identify(sp.embedding, sp.reliability, ctx.cfg)
        sp.profile_id, sp.match_score, sp.match_margin = m.profile_id, m.score, m.margin
        sp.match_result, sp.runner_up_id, sp.runner_up_score = m.result, m.runner_up_id, m.runner_up_score

        if m.result == "unknown" and sp.reliability >= ctx.cfg.reliability_fair:
            sp.cluster_id = await sc.assign_cluster(sp.embedding, sp.speech_s, ctx.cfg)

        if (ctx.cfg.auto_enroll and m.result == "confident"
                and m.score >= ctx.cfg.auto_enroll_min_sim
                and sp.reliability >= ctx.cfg.auto_enroll_min_reliability):
            status = await db.fetchval("SELECT status FROM speaker_profiles WHERE id=$1", m.profile_id)
            if status == "confirmed":
                await sc.add_enrollment(m.profile_id, sp, ctx.clip_id, source="auto_confirmed")
