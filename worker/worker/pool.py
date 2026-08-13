"""Warm, load-once model pool. Cold-constructing pyannote/whisper per job would dominate
the latency budget — this is built once at worker startup and reused for every clip."""
import time
from pathlib import Path

import numpy as np
import torch

from .ctx import SR


def resolve_device(cfg) -> str:
    # cuda is selectable from the Settings page, where the person choosing it can't see
    # whether this worker's host actually has a usable GPU. Falling back beats every model
    # load in the pool failing at once on a machine that simply doesn't have one.
    if cfg.device == "cuda" and not torch.cuda.is_available():
        import structlog
        structlog.get_logger().warning("cuda_unavailable_falling_back_to_cpu")
        return "cpu"
    if cfg.device in ("cpu", "cuda"):
        return cfg.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def verify_models(model_dir: str) -> dict:
    import yaml
    reg = yaml.safe_load((Path(__file__).resolve().parents[1] / "models" / "REGISTRY.yaml").read_text())
    versions = {}
    for name, spec in reg["models"].items():
        d = Path(spec["local_dir"])
        if not d.exists() or not any(d.iterdir()):
            raise RuntimeError(f"model {name} missing at {d} — run models/prefetch.py at build time")
        versions[name] = f"{spec['id']}@{spec.get('revision', 'main')}"
    return versions


class ModelPool:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = resolve_device(cfg)
        self.versions = verify_models(cfg.model_dir)

    async def load(self):
        t0 = time.perf_counter()
        from faster_whisper import WhisperModel
        from pyannote.audio import Pipeline
        from speechbrain.inference import EncoderClassifier
        from sentence_transformers import SentenceTransformer

        md = Path(self.cfg.model_dir)
        self.asr = WhisperModel(
            str(md / "asr" / self.cfg.asr_model), device=self.device,
            compute_type=self.cfg.precision if self.device == "cuda" else "int8",
            local_files_only=True, num_workers=1)

        from silero_vad import load_silero_vad  # weights bundled in the wheel — no download
        self.vad = load_silero_vad()

        self.diar = Pipeline.from_pretrained(str(md / "diar" / self.cfg.diar_model / "config.yaml"))
        self.diar.to(torch.device(self.device))
        # pyannote defaults to batch_size=1 for both sub-models, i.e. one forward pass per
        # sliding-window chunk — on CPU that per-call overhead dominates. Batching chunks
        # together is the single biggest lever for CPU diarization latency (no accuracy
        # tradeoff, same chunks/weights, just processed together).
        if self.device == "cpu":
            self.diar._segmentation.batch_size = 64
            self.diar.embedding_batch_size = 64
            # ponytail: default sliding-window stride is 10% of window duration (90%
            # overlap) — great for accuracy, expensive on CPU. Widening to 50% overlap
            # cuts the chunk count ~5x; min_turn_s + VAD-snapping in diarize.py already
            # smooth turn boundaries, so this is a reasonable throughput/precision
            # tradeoff. Drop back toward *0.1 if diarization boundaries get noticeably
            # sloppier than this is worth.
            self.diar._segmentation.step = self.diar._segmentation.duration * 0.5

        embed_dir = str(md / "embed" / self.cfg.embed_model)
        # speechbrain defaults to LocalStrategy.SYMLINK when linking the hub cache into
        # savedir, which is fatal on Windows: creating a symlink needs admin or developer
        # mode, otherwise WinError 1314. Copying costs a few MB and works everywhere, so
        # it's not worth branching on os.name.
        from speechbrain.utils.fetching import LocalStrategy
        self.embedder = EncoderClassifier.from_hparams(
            source=embed_dir, savedir=embed_dir, run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY)

        self.text_embedder = SentenceTransformer(
            str(md / "text" / self.cfg.text_embed_model), device=self.device)

        await self.warmup()
        return int((time.perf_counter() - t0) * 1000)

    async def warmup(self):
        dummy = np.zeros(SR, dtype=np.float32)
        dummy[::100] = 0.01
        list(self.asr.transcribe(dummy, beam_size=1)[0])
        self.diar({"waveform": torch.from_numpy(dummy).unsqueeze(0), "sample_rate": SR}, num_speakers=1)
        self.embedder.encode_batch(torch.from_numpy(dummy).unsqueeze(0))
        self.text_embedder.encode(["warmup"])
