"""quick_snr_db decides whether the most expensive non-model step runs at all, so it has to
tell clean audio from noisy audio — not grade either precisely, just never call noise clean."""
from types import SimpleNamespace

import numpy as np

from worker.audio import enhance

SR = 16000


def _speech_with_gaps(noise_amp, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(SR * 6) / SR
    tone = 0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)  # on/off "speech"
    return (tone + noise_amp * rng.standard_normal(t.size)).astype(np.float32)


def test_clean_audio_reads_high_and_noisy_audio_low():
    clean = enhance.quick_snr_db(_speech_with_gaps(0.0005), SR)
    noisy = enhance.quick_snr_db(_speech_with_gaps(0.05), SR)
    assert clean > 30 > noisy


def test_skip_returns_input_untouched(monkeypatch):
    called = []
    monkeypatch.setattr(enhance.nr, "reduce_noise", lambda **kw: called.append(1))
    cfg = SimpleNamespace(denoise_enabled=True, denoise_skip_snr_db=30.0, denoise_prop_decrease=0.75)
    x = _speech_with_gaps(0.0005)
    assert enhance.denoise(x, SR, cfg) is x
    assert not called


def test_gpu_failure_falls_back_to_cpu(monkeypatch):
    calls = []

    def fake(**kw):
        calls.append(kw.get("use_torch", False))
        if kw.get("use_torch"):
            raise RuntimeError("CUDA out of memory")
        return kw["y"]

    monkeypatch.setattr(enhance.nr, "reduce_noise", fake)
    import torch
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    cfg = SimpleNamespace(denoise_enabled=True, denoise_skip_snr_db=0.0, denoise_prop_decrease=0.75)
    enhance.denoise(_speech_with_gaps(0.05), SR, cfg, device="cuda")
    assert calls == [True, False]
