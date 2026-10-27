import time
import uuid
from datetime import datetime, timezone

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


def redis():
    """The arq pool, for callers that schedule worker jobs through common/ helpers
    (graph_sync.resync) rather than one of the enqueue_* functions below."""
    return _redis


async def enqueue_retag(clip_ids: list[str] | None = None) -> str:
    """Re-derive auto tags (signal + topic) on the worker, which holds the text encoder.
    None means every COMPLETE clip."""
    job_id = f"retag-{uuid.uuid4().hex[:8]}"
    await _redis.enqueue_job("retag_clips_job", clip_ids, _job_id=job_id)
    return job_id


# Short on purpose: someone who just named a voice is looking at the screen waiting for the
# other clips to follow. Long enough that naming three voices in a row runs one pass.
REIDENTIFY_WINDOW_S = 3


async def enqueue_reidentify() -> None:
    """Re-run identification over every stored voice, coalesced like graph_sync.schedule:
    requests inside one short window share a job id, so a burst of corrections costs one
    pass. Corpus-wide because one new enrolment can match a voice in any clip."""
    bucket = int(time.time() // REIDENTIFY_WINDOW_S)
    await _redis.enqueue_job(
        "reidentify_job", None, _job_id=f"reidentify-{bucket}",
        _defer_until=datetime.fromtimestamp((bucket + 1) * REIDENTIFY_WINDOW_S, tz=timezone.utc))


async def voice_query(rel_path: str, timeout_s: float = 180.0) -> dict:
    """Transcribe and fingerprint a spoken query on the worker, and wait for the answer.

    Waits rather than streaming: the result is small and the caller is a person holding a
    microphone. The timeout is generous because the query shares the model lane with clip
    processing and may queue behind a clip already mid-pipeline."""
    job = await _redis.enqueue_job("voice_query_job", rel_path,
                                   _job_id=f"voice-{uuid.uuid4().hex[:12]}")
    return await job.result(timeout=timeout_s)


async def enqueue_maintenance() -> str:
    """Run the worker's periodic upkeep pass now instead of waiting for its next tick."""
    job_id = f"maintenance-{uuid.uuid4().hex[:8]}"
    await _redis.enqueue_job("maintenance_job", True, _job_id=job_id)
    return job_id


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
