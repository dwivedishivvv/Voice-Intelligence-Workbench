import uuid

from arq import create_pool
from arq.connections import RedisSettings

_redis = None


async def init_queue(redis_url: str):
    global _redis
    _redis = await create_pool(RedisSettings.from_dsn(redis_url))
    return _redis


async def close_queue():
    if _redis:
        await _redis.close()


async def enqueue_clip(clip_id: str) -> str:
    # arq's _job_id must be unique per *attempt* — reusing clip_id verbatim makes
    # enqueue_job silently return None (no exception) once that job's result is
    # still cached, which breaks reprocess() forever after the first run. The
    # public job/ws id returned here stays clip_id — worker events publish to
    # `job:{clip_id}` regardless of arq's internal job_id, so this doesn't affect
    # the websocket contract.
    await _redis.enqueue_job("process_clip_job", clip_id, _job_id=f"clip-{clip_id}-{uuid.uuid4().hex[:8]}")
    return clip_id


async def enqueue_live_chunk(session_id: str, chunk_rel_path: str, seq: int):
    await _redis.enqueue_job("transcribe_live_chunk_job", session_id, chunk_rel_path, seq,
                              _job_id=f"live-{session_id}-{seq}-{uuid.uuid4().hex[:8]}")


async def enqueue_f1_radio(job_id: str, rel_path: str, session_key: int | None = None,
                            driver_number: int | None = None):
    await _redis.enqueue_job("analyze_f1_radio_job", job_id, rel_path, session_key, driver_number,
                              _job_id=f"f1-{job_id}")
