import torch
from silero_vad import get_speech_timestamps


def detect_vad(audio, sr, vad_model, cfg):
    ts = get_speech_timestamps(
        torch.from_numpy(audio), vad_model, sampling_rate=sr,
        min_speech_duration_ms=cfg.vad_min_speech_ms,
        min_silence_duration_ms=cfg.vad_min_silence_ms,
        speech_pad_ms=cfg.vad_speech_pad_ms)
    return [(t["start"] / sr, t["end"] / sr) for t in ts]
