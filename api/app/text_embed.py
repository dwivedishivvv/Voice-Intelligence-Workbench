"""Query-side encoder for semantic search. Small MiniLM only — the heavy ASR/diar/ECAPA
stack stays in the worker; the API just needs to embed short search strings."""
from functools import lru_cache
from common.config import get_settings


@lru_cache
def _model():
    import torch
    from sentence_transformers import SentenceTransformer
    cfg = get_settings()
    # ponytail: GPU when the API's interpreter has a CUDA torch, else CPU. On a small card
    # this competes with the worker for VRAM to speed up encoding a handful of query words —
    # pin device="cpu" here if the worker starts hitting OOM.
    device = "cpu" if cfg.device == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")
    return SentenceTransformer(f"{cfg.model_dir}/text/{cfg.text_embed_model}", device=device)


def encode(text: str):
    return _model().encode([text], normalize_embeddings=True)[0]
