"""Model pool with lazy, priority-based GPU loading. Every model loads on first use and
stays cached across jobs (jobs are serialized through gpu_sem/live_sem — see main.py — so
there's never more than one job's worth of models needed at once). If loading or warming a
model hits a CUDA OOM, the lowest-priority already-cached model is evicted and the load is
retried, so five models that together don't fit in VRAM still work: cheap ones just get
reloaded on demand instead of all five staying permanently resident."""
import time
from pathlib import Path

import numpy as np
import structlog
import torch

from .ctx import SR

log = structlog.get_logger()

# Eviction order when a load hits CUDA OOM: first entry goes first. sentiment/text_embedder
# are the smallest wins and least central to the pipeline (sentiment is even optional);
# asr is needed by every job type (live chunks included) so it's evicted only as a last resort.
_PRIORITY = ["sentiment", "text_embedder", "embedder", "diar", "asr"]


def resolve_device(cfg) -> str:
    # cuda is selectable from the Settings page, where the person choosing it can't see
    # whether this worker's host actually has a usable GPU. Falling back beats every model
    # load in the pool failing at once on a machine that simply doesn't have one.
    if cfg.device == "cuda" and not torch.cuda.is_available():
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


def _is_oom(e: RuntimeError) -> bool:
    return "out of memory" in str(e).lower()


class ModelPool:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = resolve_device(cfg)
        self.versions = verify_models(cfg.model_dir)
        self._cache = {}
        self.vad = None

    async def load(self):
        t0 = time.perf_counter()
        from silero_vad import load_silero_vad  # weights bundled in the wheel — no download
        self.vad = load_silero_vad()
        self._get("asr")  # every job type (live chunks included) needs this one
        return int((time.perf_counter() - t0) * 1000)

    def _get(self, name):
        if name not in self._cache:
            self._cache[name] = self._load_with_eviction(name)
        return self._cache[name]

    def _load_with_eviction(self, name):
        loader = getattr(self, f"_load_{name}")
        while True:
            try:
                obj = loader()
                self._warm(name, obj)
                return obj
            except RuntimeError as e:
                if self.device != "cuda" or not _is_oom(e):
                    raise
                evicted = self._evict_lowest_priority(exclude=name)
                if evicted is None:
                    raise
                log.warning("model_evicted_for_vram", evicted=evicted, loading=name)

    def _evict_lowest_priority(self, exclude):
        for name in _PRIORITY:
            if name != exclude and name in self._cache:
                del self._cache[name]
                torch.cuda.empty_cache()
                return name
        return None

    def _warm(self, name, obj):
        dummy = np.zeros(SR, dtype=np.float32)
        dummy[::100] = 0.01
        if name == "asr":
            list(obj.transcribe(dummy, beam_size=1)[0])
        elif name == "diar":
            obj({"waveform": torch.from_numpy(dummy).unsqueeze(0), "sample_rate": SR}, num_speakers=1)
        elif name == "embedder":
            obj.encode_batch(torch.from_numpy(dummy).unsqueeze(0))
        elif name == "text_embedder":
            obj.encode(["warmup"])
        # sentiment: a small classifier — first real call's cold-start isn't worth warming

    def _load_asr(self):
        from faster_whisper import WhisperModel
        md = Path(self.cfg.model_dir)
        return WhisperModel(
            str(md / "asr" / self.cfg.asr_model), device=self.device,
            compute_type=self.cfg.precision if self.device == "cuda" else "int8",
            local_files_only=True, num_workers=1)

    def _load_diar(self):
        from pyannote.audio import Pipeline
        md = Path(self.cfg.model_dir)
        diar = Pipeline.from_pretrained(str(md / "diar" / self.cfg.diar_model / "config.yaml"))
        diar.to(torch.device(self.device))
        # pyannote defaults to batch_size=1 for both sub-models, i.e. one forward pass per
        # sliding-window chunk — on CPU that per-call overhead dominates. Batching chunks
        # together is the single biggest lever for CPU diarization latency (no accuracy
        # tradeoff, same chunks/weights, just processed together).
        if self.device == "cpu":
            diar._segmentation.batch_size = 64
            diar.embedding_batch_size = 64
            # ponytail: default sliding-window stride is 10% of window duration (90%
            # overlap) — great for accuracy, expensive on CPU. Widening to 50% overlap
            # cuts the chunk count ~5x; min_turn_s + VAD-snapping in diarize.py already
            # smooth turn boundaries, so this is a reasonable throughput/precision
            # tradeoff. Drop back toward *0.1 if diarization boundaries get noticeably
            # sloppier than this is worth.
            diar._segmentation.step = diar._segmentation.duration * 0.5
        return diar

    def _load_embedder(self):
        from speechbrain.inference import EncoderClassifier
        # speechbrain defaults to LocalStrategy.SYMLINK when linking the hub cache into
        # savedir, which is fatal on Windows: creating a symlink needs admin or developer
        # mode, otherwise WinError 1314. Copying costs a few MB and works everywhere, so
        # it's not worth branching on os.name.
        from speechbrain.utils.fetching import LocalStrategy
        md = Path(self.cfg.model_dir)
        embed_dir = str(md / "embed" / self.cfg.embed_model)
        return EncoderClassifier.from_hparams(
            source=embed_dir, savedir=embed_dir, run_opts={"device": self.device},
            local_strategy=LocalStrategy.COPY)

    def _load_text_embedder(self):
        from sentence_transformers import SentenceTransformer
        md = Path(self.cfg.model_dir)
        return SentenceTransformer(str(md / "text" / self.cfg.text_embed_model), device=self.device)

    def _load_sentiment(self):
        # Sentiment is the one optional model: it isn't in REGISTRY.yaml's required set, so a
        # MODEL_DIR without it still yields a fully working pipeline (stages/sentiment.py
        # no-ops and warns).
        md = Path(self.cfg.model_dir)
        sent_dir = md / "sentiment" / self.cfg.sentiment_model
        if not sent_dir.exists():
            return None, None
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(str(sent_dir), local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(sent_dir), local_files_only=True,
            # fp16 halves both VRAM and latency for a classifier this size; the argmax
            # label is unchanged by the precision drop, only the confidence's last digits
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device).eval()
        return tok, model

    @property
    def asr(self):
        return self._get("asr")

    @property
    def diar(self):
        return self._get("diar")

    @property
    def embedder(self):
        return self._get("embedder")

    @property
    def text_embedder(self):
        return self._get("text_embedder")

    @property
    def sentiment_tokenizer(self):
        return self._get("sentiment")[0]

    @property
    def sentiment(self):
        return self._get("sentiment")[1]
