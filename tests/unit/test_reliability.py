from types import SimpleNamespace
from common.speaker import reliability_score, reliability_band


CFG = SimpleNamespace(embed_target_s=8.0, reliability_good=0.75, reliability_fair=0.45, reliability_poor=0.20)


def test_reliability_duration_dominates():
    """Pristine 0.8s must not beat mediocre 8s — duration multiplies quality, doesn't average with it."""
    perfect = {"snr_db": 40, "clipping_ratio": 0, "bandwidth_hz": 8000}
    mediocre = {"snr_db": 12, "clipping_ratio": 0.01, "bandwidth_hz": 4000}
    assert reliability_score(0.8, perfect, CFG) < reliability_score(8.0, mediocre, CFG)


def test_reliability_band_thresholds():
    assert reliability_band(0.9, CFG) == "good"
    assert reliability_band(0.5, CFG) == "fair"
    assert reliability_band(0.3, CFG) == "poor"
    assert reliability_band(0.05, CFG) == "unusable"


if __name__ == "__main__":
    test_reliability_duration_dominates()
    test_reliability_band_thresholds()
    print("ok")
