"""classify() must not call a speaker "stressed" for having a naturally higher voice than
whoever set the session baseline first — that is speaker identity, not arousal, and a live
session (mic or F1 broadcast demo) has no per-chunk speaker ID to tell the two apart."""
from worker.audio import tone


def _features(pitch_mean=150.0, pitch_std=20.0, rate=2.5, voiced_ratio=0.8):
    return {"pitch_mean_hz": pitch_mean, "pitch_std_hz": pitch_std,
            "speech_rate_wps": rate, "voiced_ratio": voiced_ratio}


def test_higher_pitch_alone_does_not_read_as_stressed_against_baseline():
    baseline = {"n": 5, "pitch_mean": 150.0, "pitch_std": 20.0, "rate": 2.5}
    # same rate and variability as baseline, just a higher-pitched voice (a different speaker)
    louder_voice = _features(pitch_mean=150.0 * 2.0, pitch_std=20.0, rate=2.5)
    assert tone.classify(louder_voice, baseline) == "calm"


def test_higher_pitch_alone_does_not_read_as_stressed_without_baseline():
    louder_voice = _features(pitch_mean=300.0, pitch_std=20.0, rate=2.5)
    assert tone.classify(louder_voice, None) == "calm"


def test_wide_pitch_swing_still_reads_as_stressed():
    baseline = {"n": 5, "pitch_mean": 150.0, "pitch_std": 20.0, "rate": 2.5}
    agitated = _features(pitch_mean=150.0, pitch_std=20.0 * 2.0, rate=2.5)
    assert tone.classify(agitated, baseline) == "stressed"


def test_fast_speech_still_reads_as_stressed():
    baseline = {"n": 5, "pitch_mean": 150.0, "pitch_std": 20.0, "rate": 2.5}
    fast = _features(pitch_mean=150.0, pitch_std=20.0, rate=2.5 * 1.5)
    assert tone.classify(fast, baseline) == "stressed"
