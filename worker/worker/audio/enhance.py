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


def quick_snr_db(x, sr, frame_s=0.03):
    """Rough SNR from the spread of frame energies: loud frames (speech) against quiet ones
    (the floor). No VAD needed, so it can run *before* denoise to decide whether denoise is
    worth running at all. Coarse on purpose — it only has to tell clean from noisy, not
    grade either one (compute_quality does that properly, after VAD)."""
    n = int(sr * frame_s)
    if x.size < n * 10:
        return 0.0
    frames = x[: (x.size // n) * n].reshape(-1, n)
    energy = np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12
    floor, peak = np.percentile(energy, 10), np.percentile(energy, 95)
    return float(10 * np.log10(peak / floor))


def denoise(x, sr, cfg, device="cpu"):
    if not cfg.denoise_enabled:
        return x
    if cfg.denoise_skip_snr_db > 0 and quick_snr_db(x, sr) >= cfg.denoise_skip_snr_db:
        return x
    if device == "cuda":
        # Same spectral gate, run as torch on the GPU. noisereduce streams it in the same
        # padded chunks as the CPU path, so an hour of audio does not become one giant STFT
        # in VRAM beside the resident models.
        try:
            return np.asarray(nr.reduce_noise(
                y=x, sr=sr, stationary=False, prop_decrease=cfg.denoise_prop_decrease,
                use_torch=True, device=device), dtype=np.float32)
        except Exception:
            # OOM or a noisereduce build without the torch gate: the CPU path below is the
            # one this always used, so falling back changes speed, never the result shape
            import torch
            torch.cuda.empty_cache()
    # n_jobs stays 1: noisereduce's parallel path hands chunks to joblib worker processes
    # through a temp-file memmap, which those processes cannot open on Windows
    # (PermissionError, then BrokenProcessPool) — it fails rather than just being slower.
    return nr.reduce_noise(y=x, sr=sr, stationary=False,
                           prop_decrease=cfg.denoise_prop_decrease).astype(np.float32)


def spectral_flatness(x, n_fft=1024):
    import librosa
    if x.size < n_fft:
        return 0.0
    return float(np.mean(librosa.feature.spectral_flatness(y=x, n_fft=n_fft)))
