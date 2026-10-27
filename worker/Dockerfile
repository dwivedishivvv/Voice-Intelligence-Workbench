FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workbench

RUN pip install uv

# uv's default HTTP timeout (30s) is too short for the large NVIDIA CUDA wheels.
ENV UV_HTTP_TIMEOUT=600

# ── dependency install layers (cached unless pyproject.toml changes) ──────────
COPY pyproject.toml .

# Install PyTorch with CUDA 12.1 first — separate layer caches across code-only changes.
RUN uv pip install --system \
    "torch==2.4.*" "torchaudio==2.4.*" \
    --index-url https://download.pytorch.org/whl/cu121

# Remaining deps from pyproject.toml (no build backend, avoids setuptools auto-discovery).
# torch/torchaudio are already installed; uv skips them.
# transformers floor bumped to 4.41.0 because sentence-transformers 3.x imports
# is_torch_npu_available which was added in that release; other deps only require >=4.40.
RUN python3 -c "import tomllib; f=open('pyproject.toml','rb'); t=tomllib.load(f); f.close(); d=t['project']['dependencies']+t['project']['optional-dependencies']['worker']; open('/tmp/wr.txt','w').write('\n'.join(d))" \
    && uv pip install --system -r /tmp/wr.txt "transformers>=4.41.0"

# ── application code (copied last so code edits don't bust pip layers) ────────
COPY common/ common/
# common/schema.py replays the idempotent db/*.sql files at startup
COPY db/ db/
COPY worker/worker/ worker/worker/
COPY worker/models/ worker/models/

ENV PYTHONPATH=/workbench

# Entrypoint last — changing it doesn't bust any install layer
COPY worker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
