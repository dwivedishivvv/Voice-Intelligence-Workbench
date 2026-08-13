import numpy as np
from scipy.signal import butter, sosfilt
import pyloudnorm as pyln
import noisereduce as nr


def highpass(x, sr, cutoff_hz, order=4):
    sos = butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def loudness_normalize(x, sr, target_lufs=-23.0):
    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(x)
    except Exception:
        return x, None
    if not np.isfinite(loudness) or loudness < -70:
        return x, loudness  # effectively silent — don't amplify noise
    y = pyln.normalize.loudness(x, loudness, target_lufs)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype(np.float32), float(loudness)


def denoise(x, sr, cfg):
    if not cfg.denoise_enabled:
        return x
    return nr.reduce_noise(y=x, sr=sr, stationary=False,
                            prop_decrease=cfg.denoise_prop_decrease).astype(np.float32)


def spectral_flatness(x, n_fft=1024):
    import librosa
    if x.size < n_fft:
        return 0.0
    return float(np.mean(librosa.feature.spectral_flatness(y=x, n_fft=n_fft)))
