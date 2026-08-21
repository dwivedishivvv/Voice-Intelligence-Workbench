"""Fingerprinting changes: calibration geometry, AS-Norm gating, cluster weighting.

All pure arithmetic or monkeypatched I/O — no database, no models. These cover the parts
that fail *silently*: a threshold calibrated against the wrong distribution and a centroid
dragged around by junk audio both keep returning plausible numbers while getting steadily
worse, which is exactly what nothing was catching before.
"""
import asyncio

import numpy as np
import pytest

from common import speaker as sc


class Cfg:
    id_threshold = 0.55
    id_suggest_delta = 0.08
    id_min_margin = 0.04
    id_threshold_penalty = 0.10
    id_min_z = 1.5
    reliability_fair = 0.45
    cluster_threshold = 0.52
    cluster_min_margin = 0.04
    cluster_merge_threshold = 0.62
    embed_target_s = 8.0


def unit(*xs) -> np.ndarray:
    v = np.asarray(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


# ---- compute_eer: vectorized form must equal the loop it replaced ----

def _eer_reference(genuine, impostor):
    """The pre-vectorization implementation, kept here as the oracle."""
    scores = np.array(genuine + impostor)
    best_t, best_gap, best_eer = 0.5, 9e9, 1.0
    for t in np.linspace(scores.min(), scores.max(), 500):
        far = float(np.mean(np.array(impostor) >= t)) if impostor else 0.0
        frr = float(np.mean(np.array(genuine) < t)) if genuine else 0.0
        if abs(far - frr) < best_gap:
            best_gap, best_t, best_eer = abs(far - frr), float(t), (far + frr) / 2
    return best_t, best_eer


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_compute_eer_matches_the_loop_it_replaced(seed):
    rng = np.random.default_rng(seed)
    genuine = (0.55 + 0.12 * rng.standard_normal(200)).tolist()
    impostor = (0.15 + 0.10 * rng.standard_normal(600)).tolist()
    t_new, eer_new = sc.compute_eer(genuine, impostor)
    t_ref, eer_ref = _eer_reference(genuine, impostor)
    assert t_new == pytest.approx(t_ref, abs=1e-9)
    assert eer_new == pytest.approx(eer_ref, abs=1e-9)


def test_compute_eer_survives_an_empty_side():
    """calibrate() can only reach this with real data, but a crash here would take down
    the whole endpoint rather than degrade it."""
    assert sc.compute_eer([], [0.1, 0.2]) == (0.5, 1.0)
    assert sc.compute_eer([0.9], []) == (0.5, 1.0)


# ---- the geometry that made calibration wrong ----

def test_centroid_similarity_runs_higher_than_pairwise_similarity():
    """The reason calibrating on enrollment-vs-enrollment pairs biased the threshold low.

    Averaging cancels the noise in individual enrollments, so a held-out sample scores
    higher against its profile's centroid than against any single sibling enrollment.
    Calibrating on the pairwise numbers and then applying the cutoff to centroid scores
    therefore admits matches the operator believed they had excluded.
    """
    rng = np.random.default_rng(7)
    true_voice = unit(1.0, 0.0, 0.0, 0.0)
    E = np.stack([unit(*(true_voice + 0.45 * rng.standard_normal(4))) for _ in range(6)])
    w = np.ones(len(E), dtype=np.float32)

    held_out, rest = E[0], E[1:]
    pairwise = float(np.mean([held_out @ e for e in rest]))
    to_centroid = float(held_out @ sc.weighted_centroid(rest, w[1:]))
    assert to_centroid > pairwise


def test_weighted_centroid_is_normalized_and_follows_the_weights():
    a, b = unit(1.0, 0.0), unit(0.0, 1.0)
    E = np.stack([a, b])
    heavy_a = sc.weighted_centroid(E, np.asarray([9.0, 1.0], dtype=np.float32))
    assert float(np.linalg.norm(heavy_a)) == pytest.approx(1.0, abs=1e-5)
    assert heavy_a @ a > heavy_a @ b


# ---- AS-Norm ----

def _identify(hits, reliability=0.9, monkeypatch=None):
    async def fake_knn(emb, k=sc.KNN_K):
        return hits
    monkeypatch.setattr(sc, "knn_profiles", fake_knn)
    return asyncio.run(sc.identify(np.zeros(4, dtype=np.float32), reliability, Cfg))


def test_a_clear_winner_over_a_spread_field_is_confident(monkeypatch):
    hits = [{"profile_id": "a", "sim": 0.70}] + [
        {"profile_id": str(i), "sim": s} for i, s in enumerate([0.20, 0.18, 0.15, 0.12])]
    m = _identify(hits, monkeypatch=monkeypatch)
    assert m.result == "confident" and m.profile_id == "a"
    assert m.z is not None and m.z > Cfg.id_min_z


def test_top_of_a_crowd_is_downgraded_even_with_a_high_score(monkeypatch):
    """The failure AS-Norm exists to catch: every profile scores high against this
    embedding, which is what a blended or low-quality one looks like. The raw cosine and
    the rank-2 margin both pass; only the cohort shows it is not really a match."""
    hits = [{"profile_id": "a", "sim": 0.69}] + [
        {"profile_id": str(i), "sim": s} for i, s in enumerate([0.64, 0.63, 0.62, 0.61])]
    m = _identify(hits, monkeypatch=monkeypatch)
    assert m.score >= Cfg.id_threshold                 # cosine alone would have passed
    assert m.margin >= Cfg.id_min_margin               # so would the rank-2 margin
    assert m.z is not None and m.z < Cfg.id_min_z
    assert m.result == "suggested"                     # never silently promoted to a name


def test_a_tight_cohort_cannot_manufacture_confidence(monkeypatch):
    """Without a floor under the cohort's spread this is the pathological case: four
    impostors within a thousandth of each other give a near-zero std, so dividing even a
    trivial gap by it yields an enormous z and the crowd reads as a clean win."""
    hits = [{"profile_id": "a", "sim": 0.660}] + [
        {"profile_id": str(i), "sim": s} for i, s in enumerate([0.6001, 0.6002, 0.6003, 0.6004])]
    m = _identify(hits, monkeypatch=monkeypatch)
    assert float(np.std([0.6001, 0.6002, 0.6003, 0.6004])) < 0.001   # spread is ~nothing
    assert m.z is not None and m.z < Cfg.id_min_z
    assert m.result == "suggested"


def test_the_z_gate_can_only_downgrade_never_rescue(monkeypatch):
    """A score below threshold stays unknown however distinctive the cohort makes it look —
    the gate tightens, it does not vote."""
    hits = [{"profile_id": "a", "sim": 0.20}] + [
        {"profile_id": str(i), "sim": s} for i, s in enumerate([0.01, 0.01, 0.01, 0.01])]
    m = _identify(hits, monkeypatch=monkeypatch)
    assert m.result == "unknown" and m.profile_id is None


def test_z_is_skipped_when_the_cohort_is_too_small_to_estimate(monkeypatch):
    """Two enrolled profiles cannot describe an impostor distribution. Gating on a
    std-dev computed from one sample would be worse than not gating at all."""
    hits = [{"profile_id": "a", "sim": 0.70}, {"profile_id": "b", "sim": 0.30}]
    m = _identify(hits, monkeypatch=monkeypatch)
    assert m.z is None
    assert m.result == "confident"


def test_low_reliability_still_downgrades_a_confident_match(monkeypatch):
    hits = [{"profile_id": "a", "sim": 0.95}] + [
        {"profile_id": str(i), "sim": s} for i, s in enumerate([0.10, 0.09, 0.08, 0.07])]
    m = _identify(hits, reliability=0.1, monkeypatch=monkeypatch)
    assert m.result == "suggested"


# ---- clustering ----

class FakeDB:
    """Records writes so a test can assert what clustering *decided*, not just what it
    returned — whether a centroid moved is the whole point of the ambiguity guard."""

    def __init__(self, rows):
        self.rows, self.executed, self.inserted = rows, [], []

    async def fetch(self, _sql, *args):
        return self.rows

    async def execute(self, sql, *args):
        self.executed.append((" ".join(sql.split()), args))

    async def insert(self, table, values):
        self.inserted.append((table, values))
        return "new-cluster"


def _cluster_row(cid, centroid, sim, **kw):
    return {"id": cid, "centroid": list(centroid), "sim": sim,
            "n_members": kw.get("n_members", 3),
            "total_speech_s": kw.get("total_speech_s", 9.0),
            "weight_sum": kw.get("weight_sum", 9.0)}


def test_an_ambiguous_segment_joins_a_cluster_without_moving_its_centroid(monkeypatch):
    """A segment sitting between two clusters says those clusters may be one person — it
    says nothing about which of them it belongs to. Filing it under the nearer one avoids
    fragmenting an already-split identity; letting it drag the centroid toward the other
    would corrupt both. consolidate_clusters() is what resolves the ambiguity."""
    db = FakeDB([_cluster_row("c1", unit(1, 0, 0, 0), 0.60),
                 _cluster_row("c2", unit(0, 1, 0, 0), 0.59)])   # gap 0.01 < cluster_min_margin
    monkeypatch.setattr(sc, "db", db)
    cid = asyncio.run(sc.assign_cluster(unit(1, 1, 0, 0), 2.0, Cfg, reliability=0.8))
    assert cid == "c1"
    sql = db.executed[0][0]
    assert "centroid" not in sql, "an ambiguous segment must not teach the cluster"
    assert "n_members" in sql and "total_speech_s" in sql


def test_a_clear_match_does_move_the_centroid(monkeypatch):
    db = FakeDB([_cluster_row("c1", unit(1, 0, 0, 0), 0.90),
                 _cluster_row("c2", unit(0, 1, 0, 0), 0.20)])
    monkeypatch.setattr(sc, "db", db)
    cid = asyncio.run(sc.assign_cluster(unit(1, 0, 0, 0), 2.0, Cfg, reliability=0.8))
    assert cid == "c1"
    assert "centroid" in db.executed[0][0]


def test_a_weak_segment_moves_a_centroid_less_than_a_strong_one(monkeypatch):
    """The unweighted running mean this replaced let one scrap of audio pull as hard as a
    long clean turn, while profile centroids have always been reliability-weighted."""
    def moved_by(reliability, speech_s):
        db = FakeDB([_cluster_row("c1", unit(1, 0, 0, 0), 0.90, weight_sum=10.0)])
        monkeypatch.setattr(sc, "db", db)
        asyncio.run(sc.assign_cluster(unit(0, 1, 0, 0), speech_s, Cfg, reliability=reliability))
        new_c = np.asarray(db.executed[0][1][1], dtype=np.float32)
        return float(new_c @ unit(0, 1, 0, 0))

    assert moved_by(0.9, 8.0) > moved_by(0.2, 0.5)


def test_no_match_seeds_a_new_cluster_carrying_its_weight(monkeypatch):
    db = FakeDB([_cluster_row("c1", unit(1, 0, 0, 0), 0.10)])
    monkeypatch.setattr(sc, "db", db)
    cid = asyncio.run(sc.assign_cluster(unit(0, 1, 0, 0), 4.0, Cfg, reliability=0.5))
    assert cid == "new-cluster"
    _table, values = db.inserted[0]
    assert values["weight_sum"] == pytest.approx(2.0)      # reliability * seconds


def test_consolidation_merges_two_clusters_that_are_one_voice(monkeypatch):
    """Online clustering can only ever compare against the clusters that existed at that
    moment, so one person reliably ends up spread across several and nothing puts them back
    together. This is the pass that does."""
    a, b = unit(1.0, 0.02, 0.0), unit(1.0, 0.0, 0.02)      # ~the same direction
    db = FakeDB([_cluster_row("c1", a, 0.0, weight_sum=10.0),
                 _cluster_row("c2", b, 0.0, weight_sum=4.0)])
    monkeypatch.setattr(sc, "db", db)
    out = asyncio.run(sc.consolidate_clusters(Cfg))
    assert out == {"clusters": 2, "merged": 1, "remaining": 1}
    joined = " | ".join(sql for sql, _ in db.executed)
    assert "UPDATE clip_speakers SET cluster_id" in joined
    assert "UPDATE utterances SET cluster_id" in joined
    assert "DELETE FROM speaker_clusters" in joined


def test_consolidation_leaves_genuinely_different_voices_alone(monkeypatch):
    db = FakeDB([_cluster_row("c1", unit(1, 0, 0), 0.0),
                 _cluster_row("c2", unit(0, 1, 0), 0.0)])
    monkeypatch.setattr(sc, "db", db)
    out = asyncio.run(sc.consolidate_clusters(Cfg))
    assert out["merged"] == 0 and out["remaining"] == 2
    assert not any("DELETE" in sql for sql, _ in db.executed)


def test_the_heavier_cluster_survives_a_merge(monkeypatch):
    """The merged identity should keep the id most of its evidence is already filed under,
    so the fewest rows have to be repointed."""
    a, b = unit(1.0, 0.02, 0.0), unit(1.0, 0.0, 0.02)
    db = FakeDB([_cluster_row("light", a, 0.0, weight_sum=1.0),
                 _cluster_row("heavy", b, 0.0, weight_sum=50.0)])
    monkeypatch.setattr(sc, "db", db)
    asyncio.run(sc.consolidate_clusters(Cfg))
    repoint = [args for sql, args in db.executed if "clip_speakers" in sql][0]
    assert repoint == ("light", "heavy")


def test_consolidation_is_a_no_op_below_two_clusters(monkeypatch):
    db = FakeDB([_cluster_row("c1", unit(1, 0, 0), 0.0)])
    monkeypatch.setattr(sc, "db", db)
    assert asyncio.run(sc.consolidate_clusters(Cfg))["merged"] == 0
    assert db.executed == []
