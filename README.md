# Speaker Intelligence Workbench

Offline audio pipeline: upload a clip (≤90s) → preprocess → transcribe (faster-whisper) →
diarize (pyannote 3.1) → reconcile → embed + identify speakers (ECAPA + pgvector) →
search/export. Everything runs on-premises; models are baked into the worker image at
build time (`HF_HUB_OFFLINE=1`).

## Run it

```bash
cp .env.example .env        # edit POSTGRES_PASSWORD / API_KEY
echo "<your-hf-token>" > .hf_token   # needed once, to accept pyannote's gated-model terms
make up
```

Deps are managed with [uv](https://docs.astral.sh/uv/) — `api/pyproject.toml` and
`worker/pyproject.toml` are installed via `uv pip install --system -r pyproject.toml`
inside each Dockerfile. For local (non-Docker) dev:

```bash
uv pip install -r worker/pyproject.toml   # or api/pyproject.toml
uv pip install pytest
make test
```

Then: API on `:8000`, web on `:5173`, Postgres on `:5432`.

## Offline verification

```bash
docker compose exec worker env | grep OFFLINE   # HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1
docker compose logs worker | grep models_loaded  # confirms models loaded from local disk only
```

The worker Dockerfile fails the *build* (not the first request) if any model in
`models/REGISTRY.yaml` is missing — see `worker/Dockerfile`'s `verify_models` step.

## What's scoped out (and where the seam is)

This follows the original spec's own instruction to name what's cut rather than pretend
it isn't. Everything below is a `# ponytail:`-style deliberate simplification, not an
oversight:

- **Object storage**: local disk under `DATA_DIR`, not MinIO/S3. `common/storage.py` is
  the seam — swap its four functions for an S3 client if this goes multi-node.
- **Migrations**: one `db/init.sql` run by Postgres's own `docker-entrypoint-initdb.d`,
  not Alembic. Fine until the schema needs versioned upgrades against live data.
- **Auth**: single static API key (`common.config.api_key`) checked in
  `api/app/auth.py:get_current_user`. That one function is the documented seam for
  swapping in real SSO/RBAC — every route already depends on it, not on ad-hoc checks.
- **Pagination**: offset/limit, not cursor. Correct up to a few thousand clips; add
  cursor pagination when the library actually gets that large.
- **Phonetic search** (`dmetaphone` for ASR-mangled names) and the **2D embedding plot**:
  not implemented. Full-text, semantic, and hybrid (RRF) search all work.
- **GPU OOM recovery**: retries via arq's normal backoff, not a lower-precision
  `pool.degraded()` variant. Add if OOM under load turns out to be common.
- **Multi-tenancy / RBAC beyond admin**: out of scope per the original spec; no
  `tenant_id` column exists. Add it + a row-level-security policy if needed.

## What's real

Full pipeline stages (validate → preprocess → transcribe → diarize → reconcile → embed →
identify → postprocess → index), reliability-gated identification with margin/abstention
logic, enrollment + centroid recomputation with outlier detection, unknown-speaker
clustering with promote-to-profile, EER calibration, hash-chained audit log, structured
logs with correlation IDs, Prometheus metrics, SRT/VTT/RTTM/JSON/TXT export, hybrid
search, live-tunable pipeline settings, and an on-demand model catalog (download +
activate alternate ASR/diarization/embedding models from the Settings UI). See
`tests/unit` for the logic that has the sharpest edges (word/turn reconciliation,
reliability scoring, turn cleanup).

## Layout

```
common/        shared by api + worker: config, db (asyncpg+pgvector), storage, audit, speaker logic
api/           FastAPI — upload, listing, search, admin, correction, model catalog, WS progress
worker/        arq consumer — the ML pipeline, one stage per file under worker/worker/stages/
web/           React/Vite/TS + Tailwind/shadcn — Library, Upload, ClipDetail, Speakers, Settings
db/init.sql    full schema (pgvector + FTS)
models/        REGISTRY.yaml + prefetch.py, baked into the worker image
```
