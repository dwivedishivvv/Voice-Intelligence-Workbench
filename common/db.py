"""Thin asyncpg wrapper. No ORM — the schema is simple enough that raw SQL is less code
and less indirection than mapping it through SQLAlchemy models."""
import json
import uuid
import asyncpg
from pgvector.asyncpg import register_vector
from .config import Settings

_pool: asyncpg.Pool | None = None


def _encode(v):
    # embedding columns arrive as plain python lists; pgvector's codec (registered below)
    # handles those. Only dict/list values meant for JSONB need manual encoding, and those
    # are always dicts in this schema (config_snapshot, model_versions, before/after) —
    # embeddings are the only lists, so this stays a dict-only check.
    if isinstance(v, dict):
        return json.dumps(v)
    return v


async def init_pool(cfg: Settings) -> asyncpg.Pool:
    global _pool
    _pool = await asyncpg.create_pool(
        host=cfg.postgres_host, port=cfg.postgres_port,
        database=cfg.postgres_db, user=cfg.postgres_user,
        password=cfg.postgres_password, min_size=2, max_size=cfg.worker_concurrency + 4,
        init=register_vector,
    )
    return _pool


async def close_pool():
    if _pool:
        await _pool.close()


def pool() -> asyncpg.Pool:
    assert _pool is not None, "db pool not initialized"
    return _pool


async def fetch(query: str, *args):
    return await pool().fetch(query, *args)


async def fetchrow(query: str, *args):
    return await pool().fetchrow(query, *args)


async def fetchval(query: str, *args):
    return await pool().fetchval(query, *args)


async def execute(query: str, *args):
    return await pool().execute(query, *args)


async def insert(table: str, values: dict) -> str:
    """Omits `id` so the column's own DEFAULT (uuid_generate_v4() or BIGSERIAL) fires —
    works for both key types without the caller knowing which the table uses.
    Tables without an id column (quality_metrics) must use db.execute directly."""
    values = dict(values)
    values.pop("id", None)
    cols = list(values.keys())
    params = [_encode(values[c]) for c in cols]
    placeholders = ", ".join(f"${i+1}" for i in range(len(cols)))
    q = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) RETURNING id"
    # asyncpg returns UUID columns as asyncpg.pgproto.UUID objects, not str — callers pass
    # this id into Path("/foo") / clip_id (storage.py) which raises TypeError on anything
    # that isn't str/PathLike. BIGSERIAL ids come back as plain int and must stay that way
    # (asyncpg param binding is strict about int vs str), so only the UUID case is cast.
    row_id = await fetchval(q, *params)
    return str(row_id) if isinstance(row_id, uuid.UUID) else row_id
