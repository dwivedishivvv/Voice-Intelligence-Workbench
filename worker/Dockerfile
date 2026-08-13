# ── model stage: bake all HF weights into the image at build time ──
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS models
WORKDIR /build
RUN uv pip install --system huggingface_hub pyyaml
COPY models/ models/
# HF_HOME here matters: nested repo-id refs inside pyannote's config.yaml (e.g. its
# segmentation submodel) resolve via hf_hub_download's *cache*, not local_dir copies —
# baking that cache under /models makes it survive into the runtime image below.
ENV HF_HOME=/models/.cache
RUN --mount=type=secret,id=hf_token \
    python models/prefetch.py --registry models/REGISTRY.yaml \
      --token "$(cat /run/secrets/hf_token 2>/dev/null || echo '')"

# ── runtime stage ──
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=models /models /models
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MODEL_DIR=/models HF_HOME=/models/.cache PYTHONPATH=/app

WORKDIR /app
COPY worker/pyproject.toml worker/pyproject.toml
RUN cd worker && uv pip install --system -r pyproject.toml

COPY common/ common/
COPY models/REGISTRY.yaml models/REGISTRY.yaml
COPY worker/worker/ worker/

# pyannote.audio's Model.from_pretrained ignores HF_HOME — it has its own cache_dir
# default (PYANNOTE_CACHE, or ~/.cache/torch/pyannote) for resolving nested submodel
# refs like segmentation-3.0. Point it at the same hub cache HF_HOME already resolved
# above (HF_HOME/hub) so offline lookups actually find what prefetch.py downloaded.
ENV PYANNOTE_CACHE=/models/.cache/hub

# fail fast at build time if a model is missing or corrupt
RUN python -c "from worker.pool import verify_models; verify_models('/models')"

CMD ["arq", "worker.main.WorkerSettings"]
