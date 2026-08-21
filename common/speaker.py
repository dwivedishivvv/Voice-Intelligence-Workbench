"""Speaker profile logic shared by the worker pipeline (identify stage) and the API
(manual correction endpoint) — one implementation, not two copies that drift."""
from dataclasses import dataclass

import numpy as np

from . import db


# Cohort size for the AS-Norm z-test in identify(). 10 keeps the pgvector lookup cheap
# while leaving ~9 impostors to estimate this embedding's own score distribution from.
KNN_K = 10
# Below this many cohort members the mean/std estimate is too noisy to gate on, so the
# z-test is skipped rather than applied to a number nobody should trust.
AS_NORM_MIN_COHORT = 3
# Floor under the cohort's standard deviation before it is divided by. A cohort whose
# scores sit almost on top of each other has a near-zero spread, and dividing a small
# honest gap by it manufactures a huge z — the exact opposite of what the gate is for,
# since everything scoring alike is what a non-discriminative embedding looks like. The
# floor caps how much confidence a tight cohort can be read as conferring; a genuinely
# separated match clears it many times over anyway.
AS_NORM_MIN_SD = 0.05


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def weighted_centroid(E: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Reliability-weighted mean of a set of embeddings, L2-normalized.

    Shared by recompute_centroid() (which builds the stored profile centroid) and
    calibrate() (which must reproduce that exact geometry to derive a threshold that
    means anything at inference time). Two copies of this formula silently drifting
    apart is the whole reason calibration measured the wrong distribution before.
    """
    c = (E * w[:, None]).sum(0) / max(float(w.sum()), 1e-6)
    return c / max(float(np.linalg.norm(c)), 1e-6)


def reliability_score(duration_s: float, quality: dict, cfg) -> float:
    d = min(1.0, duration_s / cfg.embed_target_s) ** 0.5
    q = clamp01((quality.get("snr_db", 10.0) - 5.0) / 20.0)
    c = 1.0 - min(1.0, quality.get("clipping_ratio", 0.0) * 20.0)
    b = clamp01((quality.get("bandwidth_hz", 4000.0) - 3000.0) / 5000.0)
    return float(d * (0.5 + 0.5 * (0.5 * q + 0.3 * c + 0.2 * b)))


def reliability_band(r: float, cfg) -> str:
    if r >= cfg.reliability_good:
        return "good"
    if r >= cfg.reliability_fair:
        return "fair"
    if r >= cfg.reliability_poor:
        return "poor"
    return "unusable"


@dataclass
class Match:
    profile_id: str | None
    score: float
    margin: float
    result: str
    runner_up_id: str | None = None
    runner_up_score: float | None = None
    # AS-Norm z-score (see identify). None when the cohort was too small to estimate.
    z: float | None = None


async def knn_profiles(emb: np.ndarray, k: int = KNN_K):
    return await db.fetch(
        """SELECT id AS profile_id, 1 - (centroid <=> $1::vector) AS sim
           FROM speaker_profiles WHERE centroid IS NOT NULL
           ORDER BY centroid <=> $1::vector LIMIT $2""",
        list(map(float, emb)), k)


async def identify(emb: np.ndarray, reliability: float, cfg) -> Match:
    hits = await knn_profiles(emb, k=KNN_K)
    if not hits:
        return Match(None, 0.0, 0.0, "unknown")

    top, second = hits[0], (hits[1] if len(hits) > 1 else None)
    margin = top["sim"] - (second["sim"] if second else 0.0)
    thr = cfg.id_threshold + cfg.id_threshold_penalty * (1.0 - reliability)

    # AS-Norm: how far the best score sits above *this embedding's own* impostor
    # distribution, rather than above a fixed cutoff. `margin` only looks at rank 2, so it
    # cannot tell "clearly the best of a well-separated field" from "top of a crowd where
    # everything scores high" — and the second is what a blended (two voices in one
    # diarization label) or low-quality embedding looks like. That case is exactly how a
    # confident *wrong* name gets written, so it is gated here rather than left to the
    # raw cosine, which a fixed threshold cannot distinguish.
    #
    # Deliberately an extra gate on top of the cosine, never a replacement for it: the
    # stored match_score, the calibrated thresholds and the Settings UI all speak cosine,
    # and switching the units under them would invalidate every one of those at once.
    cohort = [h["sim"] for h in hits[1:]]
    z = None
    if len(cohort) >= AS_NORM_MIN_COHORT:
        sd = max(float(np.std(cohort)), AS_NORM_MIN_SD)
        z = float((top["sim"] - float(np.mean(cohort))) / sd)

    if (top["sim"] >= thr and margin >= cfg.id_min_margin
            and (z is None or z >= cfg.id_min_z)):
        res = "confident"
    elif top["sim"] >= thr - cfg.id_suggest_delta:
        res = "suggested"
    else:
        return Match(None, top["sim"], margin, "unknown", top["profile_id"], top["sim"], z)

    if reliability < cfg.reliability_fair and res == "confident":
        res = "suggested"

    return Match(top["profile_id"], top["sim"], margin, res,
                 second["profile_id"] if second else None,
                 second["sim"] if second else None, z)


async def add_enrollment(profile_id: str, sp, clip_id: str, source: str, actor=None) -> str:
    # called with either a SpeakerResult dataclass (attribute access only) from the worker
    # or an asyncpg.Record (dict-like __getitem__, but NOT a dict instance and no attribute
    # access) from the API's manual-correction endpoint — isinstance(sp, dict) is false for
    # both dataclasses AND Record, so it always fell through to attribute access and crashed
    # on Record. hasattr(sp, "keys") is true for dict and Record, false for the dataclass.
    dict_like = hasattr(sp, "keys")
    reliability = sp["reliability"] if dict_like else sp.reliability
    embedding = sp["embedding"] if dict_like else sp.embedding
    if reliability is None or reliability < 0:
        raise ValueError("missing reliability")
    speech_s = sp["speech_s"] if dict_like else sp.speech_s
    local_label = sp["local_label"] if dict_like else sp.local_label

    eid = await db.insert("speaker_enrollments", {
        "profile_id": profile_id, "clip_id": clip_id, "local_label": local_label,
        "embedding": list(map(float, embedding)), "duration_s": speech_s,
        "reliability": reliability, "source": source,
    })
    await recompute_centroid(profile_id)
    return eid


async def recompute_centroid(profile_id: str):
    rows = await db.fetch(
        """SELECT id, embedding, reliability, duration_s FROM speaker_enrollments
           WHERE profile_id=$1 AND NOT is_outlier""", profile_id)
    if not rows:
        await db.execute(
            """UPDATE speaker_profiles SET centroid=NULL, n_enrollments=0, intra_cohesion=NULL,
               updated_at=now() WHERE id=$1""", profile_id)
        return

    E = np.stack([np.array(r["embedding"], dtype=np.float32) for r in rows])
    w = np.array([r["reliability"] for r in rows], dtype=np.float32)
    c = weighted_centroid(E, w)

    sims = E @ c
    cohesion = float(sims.mean())
    sd = float(sims.std())
    if len(rows) >= 4:
        for r, s in zip(rows, sims):
            outlier = bool(s < cohesion - 2 * sd) and s < 0.5
            await db.execute("UPDATE speaker_enrollments SET is_outlier=$2, cos_to_centroid=$3 WHERE id=$1",
                              r["id"], outlier, float(s))

    await db.execute(
        """UPDATE speaker_profiles SET centroid=$2, n_enrollments=$3, total_speech_s=$4,
           mean_reliability=$5, intra_cohesion=$6, updated_at=now() WHERE id=$1""",
        profile_id, list(map(float, c)), len(rows),
        float(sum(r["duration_s"] for r in rows)), float(w.mean()), cohesion)


async def assign_cluster(emb: np.ndarray, speech_s: float, cfg, reliability: float = 1.0) -> str:
    """Nearest-centroid online clustering of voices nobody has named yet.

    Held to the same standard as profile centroids, which it was not before: the running
    mean is reliability-weighted (an unweighted mean let a 1s scrap of audio move the
    centroid as far as 8s of clean speech, while recompute_centroid has always weighted
    profiles), and an ambiguous embedding is not allowed to *teach* the cluster anything.

    Two clusters and one query returns top-2 in a single round trip, carrying the columns
    the update needs — the old code did a LIMIT 1 lookup and then a second SELECT for the
    row it had just found.
    """
    hits = await db.fetch(
        """SELECT id, centroid, n_members, total_speech_s, weight_sum,
                  1 - (centroid <=> $1::vector) AS sim
           FROM speaker_clusters
           WHERE promoted_to IS NULL AND centroid IS NOT NULL
           ORDER BY centroid <=> $1::vector LIMIT 2""", list(map(float, emb)))

    w = max(float(reliability) * float(speech_s), 1e-6)
    if hits and hits[0]["sim"] >= cfg.cluster_threshold:
        top = hits[0]
        cid = top["id"]
        runner = float(hits[1]["sim"]) if len(hits) > 1 else 0.0
        # An embedding sitting between two clusters is evidence those two clusters may be
        # one person — not evidence about which of them this segment belongs to. It is
        # still assigned to the nearer one (spawning a third cluster would only fragment
        # an identity that is already split), but it must not drag either centroid toward
        # the other: consolidate_clusters() is what resolves that ambiguity, with the whole
        # picture rather than one borderline segment.
        if (float(top["sim"]) - runner) >= cfg.cluster_min_margin:
            # weight_sum is 0 on rows written before it existed; fall back to the seconds
            # (then members) they did record, so a pre-existing centroid keeps its mass
            # instead of being wiped out by the first weighted update.
            old_w = (float(top["weight_sum"] or 0.0) or float(top["total_speech_s"] or 0.0)
                     or float(top["n_members"] or 1))
            old_c = (np.array(top["centroid"], dtype=np.float32)
                     if top["centroid"] is not None else emb)
            new_c = (old_c * old_w + emb * w) / max(old_w + w, 1e-6)
            new_c /= max(float(np.linalg.norm(new_c)), 1e-6)
            await db.execute(
                """UPDATE speaker_clusters SET centroid=$2, n_members=$3,
                       total_speech_s=total_speech_s+$4, weight_sum=$5 WHERE id=$1""",
                cid, list(map(float, new_c)), top["n_members"] + 1, speech_s, old_w + w)
        else:
            await db.execute(
                """UPDATE speaker_clusters SET n_members=$2, total_speech_s=total_speech_s+$3
                   WHERE id=$1""", cid, top["n_members"] + 1, speech_s)
        return cid

    return await db.insert("speaker_clusters", {
        "label": "Unknown", "centroid": list(map(float, emb)), "n_members": 1,
        "total_speech_s": speech_s, "weight_sum": w})


async def consolidate_clusters(cfg) -> dict:
    """Merge unclaimed clusters that are the same voice.

    assign_cluster() is greedy and order-dependent: it can only ever compare a new segment
    against the clusters that happen to exist at that moment, so one person reliably ends
    up spread across several clusters and nothing ever puts them back together. (The
    cluster_threshold comment in common/config.py documents this failure measured on real
    radio: 59 clusters for 65 assignments at the old cutoff.) Lowering the threshold
    reduced the symptom; this is the part that actually resolves it, with every centroid
    visible at once instead of one segment at a time.

    Merging two whole clusters asserts far more than admitting one segment, so it takes a
    stricter cutoff (cluster_merge_threshold) than assign_cluster's.

    ponytail: recomputes the full similarity matrix each merge — O(n^3) overall, a second
    or so at a few hundred clusters, and this is an explicitly-triggered admin pass rather
    than something on the ingest path. Switch to a heap of candidate pairs if it ever runs
    against thousands.
    """
    rows = await db.fetch(
        """SELECT id, centroid, n_members, total_speech_s, weight_sum
           FROM speaker_clusters WHERE promoted_to IS NULL AND centroid IS NOT NULL""")
    if len(rows) < 2:
        return {"clusters": len(rows), "merged": 0, "remaining": len(rows)}

    live: dict[str, dict] = {}
    for r in rows:
        w = (float(r["weight_sum"] or 0.0) or float(r["total_speech_s"] or 0.0)
             or float(r["n_members"] or 1))
        live[str(r["id"])] = {
            "c": np.array(r["centroid"], dtype=np.float32), "w": w,
            "n": int(r["n_members"] or 0), "s": float(r["total_speech_s"] or 0.0),
            "dirty": False,
        }

    absorbed: list[tuple[str, str]] = []
    while len(live) >= 2:
        ids = list(live)
        C = np.stack([live[i]["c"] for i in ids])
        S = C @ C.T
        np.fill_diagonal(S, -1.0)
        a, b = np.unravel_index(int(np.argmax(S)), S.shape)
        if float(S[a, b]) < cfg.cluster_merge_threshold:
            break
        ia, ib = ids[int(a)], ids[int(b)]
        # the heavier cluster survives, so the merged identity keeps the id that most of
        # its evidence is already filed under
        keep, drop = (ia, ib) if live[ia]["w"] >= live[ib]["w"] else (ib, ia)
        k, d = live[keep], live[drop]
        merged = (k["c"] * k["w"] + d["c"] * d["w"]) / max(k["w"] + d["w"], 1e-6)
        k["c"] = merged / max(float(np.linalg.norm(merged)), 1e-6)
        k["w"] += d["w"]
        k["n"] += d["n"]
        k["s"] += d["s"]
        k["dirty"] = True
        del live[drop]
        absorbed.append((drop, keep))

    for drop, keep in absorbed:
        # repoint every reference before the row goes, so nothing is left pointing at a
        # cluster id that no longer exists
        await db.execute("UPDATE clip_speakers SET cluster_id=$2 WHERE cluster_id=$1", drop, keep)
        await db.execute("UPDATE utterances SET cluster_id=$2 WHERE cluster_id=$1", drop, keep)
        await db.execute("DELETE FROM speaker_clusters WHERE id=$1", drop)

    for cid, v in live.items():
        if v["dirty"]:
            await db.execute(
                """UPDATE speaker_clusters SET centroid=$2, n_members=$3, total_speech_s=$4,
                       weight_sum=$5 WHERE id=$1""",
                cid, list(map(float, v["c"])), v["n"], v["s"], v["w"])

    return {"clusters": len(rows), "merged": len(absorbed), "remaining": len(live)}


async def merge_profiles(src_id: str, dst_id: str):
    await db.execute("UPDATE speaker_enrollments SET profile_id=$2 WHERE profile_id=$1", src_id, dst_id)
    await db.execute("UPDATE clip_speakers SET profile_id=$2 WHERE profile_id=$1", src_id, dst_id)
    await db.execute("UPDATE utterances SET profile_id=$2 WHERE profile_id=$1", src_id, dst_id)
    await db.execute("DELETE FROM speaker_profiles WHERE id=$1", src_id)
    await recompute_centroid(dst_id)


async def promote_cluster(cluster_id: str, name: str, cfg) -> str:
    pid = await db.insert("speaker_profiles", {"display_name": name, "status": "provisional"})
    members = await db.fetch("SELECT * FROM clip_speakers WHERE cluster_id=$1", cluster_id)
    for m in members:
        if (m["reliability"] or 0) >= cfg.reliability_fair and m["embedding"] is not None:
            await db.insert("speaker_enrollments", {
                "profile_id": pid, "clip_id": m["clip_id"], "embedding": m["embedding"],
                "duration_s": m["speech_s"], "reliability": m["reliability"], "source": "promoted"})
    # Order matters: the utterances UPDATE finds its rows *through* clip_speakers.cluster_id,
    # so it has to run while that column still holds the cluster. Clearing clip_speakers
    # first (as this did) left the second statement matching zero rows every time, so a
    # promoted cluster got its name on clip_speakers but never on its utterances — the
    # transcript kept showing the voice as unnamed no matter how many times you named it.
    await db.execute(
        """UPDATE utterances u SET profile_id=$2 FROM clip_speakers cs
           WHERE cs.cluster_id=$1 AND u.clip_id=cs.clip_id AND u.local_label=cs.local_label""", cluster_id, pid)
    await db.execute("UPDATE clip_speakers SET profile_id=$2, cluster_id=NULL WHERE cluster_id=$1", cluster_id, pid)
    await db.execute("UPDATE speaker_clusters SET promoted_to=$2 WHERE id=$1", cluster_id, pid)
    await recompute_centroid(pid)
    return pid


def compute_eer(genuine: list[float], impostor: list[float]):
    """Threshold where false-accept and false-reject rates meet, and that rate.

    Vectorized over the candidate thresholds via two sorted searches rather than
    re-scanning both score lists 500 times — same answer, and it stops being the slow part
    of calibrate() once a corpus has enough profiles to be worth calibrating on.
    """
    if not genuine or not impostor:
        return 0.5, 1.0
    g = np.sort(np.asarray(genuine, dtype=np.float64))
    im = np.sort(np.asarray(impostor, dtype=np.float64))
    ts = np.linspace(min(g[0], im[0]), max(g[-1], im[-1]), 500)
    # counted, not derived as 1 - below/n: the two disagree in the last bit, which is
    # enough to pick a different threshold whenever two candidates tie on |far - frr|
    far = (im.size - np.searchsorted(im, ts, side="left")) / im.size  # impostors scoring >= t
    frr = np.searchsorted(g, ts, side="left") / g.size                # genuine scoring < t
    k = int(np.argmin(np.abs(far - frr)))
    return float(ts[k]), float((far[k] + frr[k]) / 2.0)


async def calibrate() -> dict:
    """Re-derive the identification thresholds from this deployment's own voices.

    Scored the way identification actually scores: a single embedding against a *profile
    centroid*, leave-one-out. The previous version compared enrollments against each other
    pairwise, which is a different distribution — averaging cancels noise, so centroid
    similarity runs systematically higher and tighter than enrollment-to-enrollment
    similarity. Calibrating on the pairwise numbers therefore produced a cutoff biased low
    for the geometry it was about to be applied to, i.e. quietly too permissive, in the one
    routine whose entire job is to make the cutoff trustworthy.

    Both sides use weighted_centroid(), the same function that builds the stored centroid,
    so this cannot drift away from inference again.
    """
    profiles = await db.fetch(
        """SELECT p.id, array_agg(e.embedding) AS embs, array_agg(e.reliability) AS rels
           FROM speaker_profiles p
           JOIN speaker_enrollments e ON e.profile_id=p.id AND NOT e.is_outlier
           GROUP BY p.id HAVING count(e.id) >= 2""")
    if len(profiles) < 3:
        return {"error": "need >=3 profiles with >=2 enrollments each"}

    mats = []
    for p in profiles:
        E = np.stack([np.array(e, dtype=np.float32) for e in p["embs"]])
        w = np.asarray([float(r) for r in p["rels"]], dtype=np.float32)
        mats.append((E, w, weighted_centroid(E, w)))

    genuine, impostor = [], []
    for i, (E, w, _c) in enumerate(mats):
        others = [mats[j][2] for j in range(len(mats)) if j != i]
        O = np.stack(others)
        for j in range(len(E)):
            keep = np.arange(len(E)) != j
            # held-out enrollment vs the centroid of the rest of its own profile: exactly
            # what identify() computes for a query it has never seen
            genuine.append(float(E[j] @ weighted_centroid(E[keep], w[keep])))
            impostor.extend((O @ E[j]).tolist())

    thr, eer = compute_eer(genuine, impostor)
    result = {
        "n_profiles": len(profiles), "n_genuine": len(genuine), "n_impostor": len(impostor),
        "eer": eer, "eer_threshold": thr,
        "genuine_mean": float(np.mean(genuine)), "impostor_mean": float(np.mean(impostor)),
        "separation": float(np.mean(genuine) - np.mean(impostor)),
        "suggested_id_threshold": float(thr + 0.03), "suggested_verify_threshold": float(thr + 0.08),
        "histogram": {"genuine": genuine[:500], "impostor": impostor[:500]},
    }
    await db.insert("calibration_runs", result)
    return result
