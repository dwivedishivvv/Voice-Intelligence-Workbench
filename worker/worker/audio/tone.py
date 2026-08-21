"""Heuristic acoustic tone classifier — NOT a trained model. Speech rate and pitch
(F0) mean/variability are the two acoustic features most consistently linked to
vocal stress in the speech-emotion literature; this buckets on simple thresholds
over them rather than running a real speech-emotion-recognition classifier, which
would need a new HF model pulled and wired into the offline model-registry
pattern (see worker/models/REGISTRY.yaml) — a bigger lift than this pass covers.
Good enough to show a signal; swap for a trained SER model if this proves noisy.
"""
import numpy as np


NEUTRAL_FEATURES = {
    "pitch_mean_hz": 0.0, "pitch_std_hz": 0.0, "energy_std": 0.0,
    "speech_rate_wps": 0.0, "voiced_ratio": 0.0,
}

# pyin needs a handful of full analysis frames to say anything about pitch — F1 radio
# calls run from ~1s "copy" acknowledgements up to 10+s explanations, and the short
# end isn't long enough for reliable pitch tracking. Below this, skip straight to
# NEUTRAL_FEATURES rather than handing librosa a signal it can't analyze.
MIN_DURATION_S = 1.5


def extract_features(audio: np.ndarray, sr: int, word_count: int, duration_s: float) -> dict:
    import librosa

    speech_rate = word_count / duration_s if duration_s > 0 else 0.0

    if duration_s < MIN_DURATION_S:
        return {**NEUTRAL_FEATURES, "speech_rate_wps": round(speech_rate, 2)}

    try:
        f0, voiced_flag, _ = librosa.pyin(audio, fmin=65, fmax=400, sr=sr)
    except Exception:
        # pyin can still fail on clean, adequately-long clips that are just quiet/noisy
        # (radio static, no voice) — that's a "no signal" case, not a job failure
        return {**NEUTRAL_FEATURES, "speech_rate_wps": round(speech_rate, 2)}

    voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
    pitch_mean = float(np.nanmean(voiced)) if voiced.size else 0.0
    pitch_std = float(np.nanstd(voiced)) if voiced.size else 0.0
    if not np.isfinite(pitch_mean):
        pitch_mean, pitch_std = 0.0, 0.0

    rms = librosa.feature.rms(y=audio)[0]
    energy_std = float(np.std(rms))

    return {
        "pitch_mean_hz": round(pitch_mean, 1),
        "pitch_std_hz": round(pitch_std, 1),
        "energy_std": round(energy_std, 4),
        "speech_rate_wps": round(speech_rate, 2),
        "voiced_ratio": round(float(voiced.size) / max(f0.size, 1), 2),
    }


MIN_BASELINE_SAMPLES = 3  # calm-classified chunks needed before trusting the session baseline over fixed thresholds
BASELINE_ALPHA = 0.2  # EMA weight — slow enough to resist one noisy chunk, fast enough to track a session


def new_baseline() -> dict:
    return {"n": 0, "pitch_mean": 0.0, "pitch_std": 0.0, "rate": 0.0}


def update_baseline(baseline: dict, features: dict) -> None:
    if baseline["n"] == 0:
        baseline["pitch_mean"] = features["pitch_mean_hz"]
        baseline["pitch_std"] = features["pitch_std_hz"]
        baseline["rate"] = features["speech_rate_wps"]
    else:
        a = BASELINE_ALPHA
        baseline["pitch_mean"] += a * (features["pitch_mean_hz"] - baseline["pitch_mean"])
        baseline["pitch_std"] += a * (features["pitch_std_hz"] - baseline["pitch_std"])
        baseline["rate"] += a * (features["speech_rate_wps"] - baseline["rate"])
    baseline["n"] += 1


def classify(features: dict, baseline: dict | None = None) -> str:
    if features["voiced_ratio"] < 0.15:
        return "calm"  # too little voiced signal to say anything meaningful

    rate = features["speech_rate_wps"]
    pitch_mean = features["pitch_mean_hz"]
    pitch_std = features["pitch_std_hz"]

    # fixed absolute thresholds (tuned loosely off one test clip) don't generalize across
    # voices/mics — someone's natural pitch might sit above or below the global cutoff for
    # reasons that have nothing to do with stress. Once a session has a few calm-looking
    # samples to anchor to, judge relative to THIS speaker's own baseline instead.
    #
    # Absolute pitch height (pitch_mean) is deliberately not part of either "stressed"
    # trigger below. It is mostly vocal anatomy — who is talking — not arousal, and a live
    # session has no per-chunk speaker ID (see the Live page copy: "Identification is not
    # run per chunk"). A single EMA baseline gets built from whichever voice happened to
    # read as calm first, so on any session with more than one speaker — two people
    # sharing a mic, or a broadcast alternating between commentators — a naturally
    # higher-pitched second speaker cleared the old baseline["pitch_mean"] * 1.25 bar on
    # pitch alone and got called "stressed" regardless of how they actually sounded.
    # Pitch *variability* (pitch_std) and speech rate are far more speaker-invariant
    # arousal signals, so they carry the whole test now.
    if baseline and baseline["n"] >= MIN_BASELINE_SAMPLES and baseline["pitch_mean"] > 0:
        if rate < baseline["rate"] * 0.55 and pitch_std < baseline["pitch_std"] * 0.6:
            return "tired"
        if rate > baseline["rate"] * 1.45 or pitch_std > baseline["pitch_std"] * 1.8:
            return "stressed"
        return "calm"

    if rate < 1.8 and pitch_std < 15:
        return "tired"
    if rate > 3.8 or pitch_std > 40:
        return "stressed"
    return "calm"
