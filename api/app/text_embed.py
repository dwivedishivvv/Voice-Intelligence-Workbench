"""Query-side encoder for semantic search. Small MiniLM only — the heavy ASR/diar/ECAPA
stack stays in the worker; the API just needs to embed short search strings."""
from functools import lru_cache
from common.config import get_settings


@lru_cache
def _model():
    from sentence_transformers import SentenceTransformer
    cfg = get_settings()
    return SentenceTransformer(f"{cfg.model_dir}/text/{cfg.text_embed_model}", device="cpu")


def encode(text: str):
    return _model().encode([text], normalize_embeddings=True)[0]
