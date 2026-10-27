"""WeSpeaker ResNet speaker embeddings behind speechbrain's `encode_batch` interface.

Why ResNet293 replaced ECAPA as the default. Measured on this machine (RTX 4050) against
LibriSpeech test-clean, 40 speakers, every trial across two different recording sessions,
segments preprocessed exactly as stages/preprocess.py does (DC, high-pass, loudness):

    cross-session EER        clean   telephone+noise   clean vs degraded   3s test
    ECAPA (speechbrain)      0.33%        1.67%              2.22%           0.89%
    WeSpeaker ResNet34       0.17%        0.99%              1.96%           0.52%
    WeSpeaker ResNet293      0.03%        0.52%              0.68%           0.26%
    ECAPA + ResNet293 fused  0.02%        0.68%              1.18%           0.26%

About 3x fewer errors in exactly the conditions cross-recording identity lives in (a voice
heard once on a clean mic and again on a compressed channel), for 83 ms per 8s segment.
Fusing with ECAPA made the mismatched case worse, so it replaces ECAPA rather than joining
it. The two score on almost the same cosine scale (impostor mean 0.103 vs 0.107, p99 0.409
vs 0.413), which is why the identification thresholds only moved by the equal-false-accept
shift measured alongside (see common/config.py).

pyannote.audio already ships the WeSpeaker ResNet architectures (it uses ResNet34 inside
its diarization pipeline), so this needs no new dependency — only the checkpoint.
"""
from pathlib import Path

import numpy as np
import torch
import yaml


class WeSpeakerEmbedder:
    """Drop-in for speechbrain's EncoderClassifier where stages/embed.py uses it.

    `encode_batch(wavs, wav_lens)` takes a zero-padded (batch, samples) tensor plus each
    row's relative length, and returns (batch, 1, dim), like speechbrain does.

    Rows are never padded. Masking the padding was tried first (features computed per row
    at true length, pooling told which frames were real) and measured not good enough:
    ResNet293 is deep enough that zero frames leak through its receptive field into the
    real ones, and a 0.7s row batched beside an 8s row came out at cosine 0.85 to itself
    embedded alone. A voiceprint must not depend on what it was batched with, so only rows
    of identical length share a forward pass, and the rest go one at a time — at ~80ms per
    8s row that costs well under a second per clip.
    """

    def __init__(self, model, device: str):
        self.model = model
        self.device = device
        self.dim = int(model.resnet.embed_dim)

    @classmethod
    def load(cls, model_dir: Path, device: str) -> "WeSpeakerEmbedder":
        from pyannote.audio.models.embedding import wespeaker
        cfg = yaml.safe_load((model_dir / "config.yaml").read_text())
        depth = str(cfg["model"]).removeprefix("ResNet")
        model = getattr(wespeaker, f"WeSpeakerResNet{depth}")()
        # weights_only: this is a pickle from the Hub, and a plain torch.load would run
        # whatever code it carried
        state = torch.load(model_dir / "avg_model.pt", map_location="cpu", weights_only=True)
        state.pop("projection.weight", None)  # the training-time classifier head
        model.resnet.load_state_dict(state, strict=True)
        return cls(model.to(device).eval(), device)

    def encode_batch(self, wavs: torch.Tensor, wav_lens: torch.Tensor | None = None) -> torch.Tensor:
        if wavs.dim() == 1:
            wavs = wavs.unsqueeze(0)
        b, width = wavs.shape
        lens = ([max(1, int(round(float(r) * width))) for r in wav_lens]
                if wav_lens is not None else [width] * b)
        wavs = wavs.to(self.device)
        by_len: dict[int, list[int]] = {}
        for i, n in enumerate(lens):
            by_len.setdefault(n, []).append(i)
        out = torch.empty(b, self.dim, device=self.device)
        with torch.inference_mode():
            for n, rows in by_len.items():
                out[rows] = self.model(wavs[rows, :n].unsqueeze(1))
        return out.unsqueeze(1)


def is_wespeaker_dir(model_dir: Path) -> bool:
    return (model_dir / "avg_model.pt").exists() and (model_dir / "config.yaml").exists()


def embedding_dim(embedder) -> int:
    """Width of the vectors this embedder produces, whichever implementation it is."""
    dim = getattr(embedder, "dim", None)
    if dim:
        return int(dim)
    probe = np.zeros(16000, dtype=np.float32)
    probe[::100] = 0.01
    with torch.no_grad():
        return int(embedder.encode_batch(torch.from_numpy(probe).unsqueeze(0)).reshape(1, -1).shape[1])
