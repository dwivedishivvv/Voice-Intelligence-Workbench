from common import db, speaker as sc
from .embed import SPLIT_HALF_SUSPECT


async def run(ctx):
    for label, sp in ctx.speakers.items():
        if sp.embedding is None:
            sp.match_result = "abstained"
            continue

        m = await sc.identify(sp.embedding, sp.reliability, ctx.cfg)
        sp.profile_id, sp.match_score, sp.match_margin = m.profile_id, m.score, m.margin
        sp.match_result, sp.runner_up_id, sp.runner_up_score = m.result, m.runner_up_id, m.runner_up_score
        sp.match_z = m.z

        if m.result == "unknown" and sp.reliability >= ctx.cfg.reliability_fair:
            # reliability rides along so the cluster centroid is weighted by how much this
            # embedding is worth, the same way a profile centroid always has been
            sp.cluster_id = await sc.assign_cluster(
                sp.embedding, sp.speech_s, ctx.cfg, reliability=sp.reliability)

        # Auto-enrollment is the one path that writes back into a profile, so a bad match
        # here does not just mislabel one clip — it poisons every later match against that
        # profile. A label whose two halves disagree is the specific case where a confident
        # score is least trustworthy (two voices blended into one embedding), so it is
        # barred from enrolling even when every other gate passes.
        halves_agree = sp.split_half_sim is None or sp.split_half_sim >= SPLIT_HALF_SUSPECT
        if (ctx.cfg.auto_enroll and m.result == "confident" and halves_agree
                and m.score >= ctx.cfg.auto_enroll_min_sim
                and sp.reliability >= ctx.cfg.auto_enroll_min_reliability):
            status = await db.fetchval("SELECT status FROM speaker_profiles WHERE id=$1", m.profile_id)
            if status == "confirmed":
                await sc.add_enrollment(m.profile_id, sp, ctx.clip_id, source="auto_confirmed")
