import numpy as np


def rolloff_99(audio, sr, n_fft=2048):
    import librosa
    if audio.size < n_fft:
        return 0.0
    S = np.abs(librosa.stft(audio, n_fft=n_fft))
    p = np.mean(S, axis=1)
    c = np.cumsum(p)
    if c[-1] <= 0:
        return 0.0
    idx = int(np.searchsorted(c, 0.99 * c[-1]))
    return float(idx * sr / n_fft)


def grade(q: dict, cfg) -> str:
    if (q["snr_db"] >= cfg.quality_good_snr_db
            and q["clipping_ratio"] <= cfg.quality_max_clipping
            and q["bandwidth_hz"] >= cfg.quality_min_bandwidth_hz):
        return "good"
    if q["snr_db"] >= cfg.quality_fair_snr_db:
        return "fair"
    return "poor"


def compute_quality(audio, vad, sr, clipping_ratio, input_lufs, cfg) -> dict:
    mask = np.zeros(len(audio), dtype=bool)
    for s, e in vad:
        mask[int(s * sr):int(e * sr)] = True

    sp, ns = audio[mask], audio[~mask]
    p_sp = float(np.mean(sp ** 2)) if sp.size else 1e-12
    p_ns = float(np.mean(ns ** 2)) if ns.size > sr * 0.1 else 1e-12
    snr = float(np.clip(10 * np.log10(max(p_sp, 1e-12) / max(p_ns, 1e-12)), -20, 60))

    q = {
        "snr_db": snr,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": 1.0 - (mask.sum() / max(len(audio), 1)),
        "speech_duration_s": float(mask.sum() / sr),
        "bandwidth_hz": rolloff_99(audio, sr),
        "dc_offset": float(np.mean(audio)) if audio.size else 0.0,
        "input_lufs": input_lufs,
    }
    q["grade"] = grade(q, cfg)
    return q
