"""Speaker profile logic shared by the worker pipeline (identify stage) and the API
(manual correction endpoint) — one implementation, not two copies that drift."""
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from . import db


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


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


async def knn_profiles(emb: np.ndarray, k: int = 5):
    return await db.fetch(
        """SELECT id AS profile_id, 1 - (centroid <=> $1::vector) AS sim
           FROM speaker_profiles WHERE centroid IS NOT NULL
           ORDER BY centroid <=> $1::vector LIMIT $2""",
        list(map(float, emb)), k)


async def identify(emb: np.ndarray, reliability: float, cfg) -> Match:
    hits = await knn_profiles(emb, k=5)
    if not hits:
        return Match(None, 0.0, 0.0, "unknown")

    top, second = hits[0], (hits[1] if len(hits) > 1 else None)
    margin = top["sim"] - (second["sim"] if second else 0.0)
    thr = cfg.id_threshold + cfg.id_threshold_penalty * (1.0 - reliability)

    if top["sim"] >= thr and margin >= cfg.id_min_margin:
        res = "confident"
    elif top["sim"] >= thr - cfg.id_suggest_delta:
        res = "suggested"
    else:
        return Match(None, top["sim"], margin, "unknown", top["profile_id"], top["sim"])

    if reliability < cfg.reliability_fair and res == "confident":
        res = "suggested"

    return Match(top["profile_id"], top["sim"], margin, res,
                 second["profile_id"] if second else None,
                 second["sim"] if second else None)


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
    c = (E * w[:, None]).sum(0) / max(float(w.sum()), 1e-6)
    c /= max(float(np.linalg.norm(c)), 1e-6)

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


async def assign_cluster(emb: np.ndarray, speech_s: float, cfg) -> str:
    hits = await db.fetch(
        """SELECT id, 1 - (centroid <=> $1::vector) AS sim FROM speaker_clusters
           WHERE promoted_to IS NULL AND centroid IS NOT NULL
           ORDER BY centroid <=> $1::vector LIMIT 1""", list(map(float, emb)))
    if hits and hits[0]["sim"] >= cfg.cluster_threshold:
        cid = hits[0]["id"]
        row = await db.fetchrow("SELECT centroid, n_members, total_speech_s FROM speaker_clusters WHERE id=$1", cid)
        n = row["n_members"]
        old_c = np.array(row["centroid"], dtype=np.float32) if row["centroid"] is not None else emb
        new_c = (old_c * n + emb) / (n + 1)
        new_c /= max(float(np.linalg.norm(new_c)), 1e-6)
        await db.execute(
            "UPDATE speaker_clusters SET centroid=$2, n_members=$3, total_speech_s=total_speech_s+$4 WHERE id=$1",
            cid, list(map(float, new_c)), n + 1, speech_s)
        return cid
    return await db.insert("speaker_clusters", {
        "label": "Unknown", "centroid": list(map(float, emb)), "n_members": 1, "total_speech_s": speech_s})


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
    await db.execute("UPDATE clip_speakers SET profile_id=$2, cluster_id=NULL WHERE cluster_id=$1", cluster_id, pid)
    await db.execute(
        """UPDATE utterances u SET profile_id=$2 FROM clip_speakers cs
           WHERE cs.cluster_id=$1 AND u.clip_id=cs.clip_id AND u.local_label=cs.local_label""", cluster_id, pid)
    await db.execute("UPDATE speaker_clusters SET promoted_to=$2 WHERE id=$1", cluster_id, pid)
    await recompute_centroid(pid)
    return pid


def compute_eer(genuine: list[float], impostor: list[float]):
    scores = np.array(genuine + impostor)
    best_t, best_gap, best_eer = 0.5, 9e9, 1.0
    for t in np.linspace(scores.min(), scores.max(), 500):
        far = float(np.mean(np.array(impostor) >= t)) if impostor else 0.0
        frr = float(np.mean(np.array(genuine) < t)) if genuine else 0.0
        if abs(far - frr) < best_gap:
            best_gap, best_t, best_eer = abs(far - frr), float(t), (far + frr) / 2
    return best_t, best_eer


async def calibrate() -> dict:
    profiles = await db.fetch(
        """SELECT p.id, array_agg(e.embedding) AS embs FROM speaker_profiles p
           JOIN speaker_enrollments e ON e.profile_id=p.id AND NOT e.is_outlier
           GROUP BY p.id HAVING count(e.id) >= 2""")
    if len(profiles) < 3:
        return {"error": "need >=3 profiles with >=2 enrollments each"}

    genuine, impostor = [], []
    for p in profiles:
        E = np.stack([np.array(e, dtype=np.float32) for e in p["embs"]])
        genuine += [float(a @ b) for a, b in combinations(E, 2)]
    for p, q in combinations(profiles, 2):
        A = np.stack([np.array(e, dtype=np.float32) for e in p["embs"]])
        B = np.stack([np.array(e, dtype=np.float32) for e in q["embs"]])
        impostor += (A @ B.T).ravel().tolist()

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
