import json
import time
import redis.asyncio as redis

_redis: "redis.Redis | None" = None


async def init_events(redis_url: str):
    global _redis
    _redis = redis.from_url(redis_url)


async def emit(clip_id: str, stage: str, state: str, **kw):
    if _redis is None:
        return
    await _redis.publish(f"job:{clip_id}",
                          json.dumps({"type": "stage", "stage": stage, "state": state, "t": time.time(), **kw}))


async def emit_live(session_id: str, seq: int, text: str, mood: str | None = None,
                     features: dict | None = None, segments: list | None = None,
                     error: str | None = None):
    # reuses the same job:{id} pubsub channel / WS route as clip processing —
    # the WS endpoint just forwards whatever's published, no live-specific plumbing needed
    if _redis is None:
        return
    await _redis.publish(f"job:{session_id}",
                          json.dumps({"type": "live_transcript", "seq": seq, "text": text,
                                      "mood": mood, "features": features, "segments": segments,
                                      "error": error, "t": time.time()}))


async def emit_f1_result(job_id: str, text: str, mood: str, features: dict | None, error: str | None = None):
    if _redis is None:
        return
    await _redis.publish(f"job:{job_id}",
                          json.dumps({"type": "f1_result", "text": text, "mood": mood,
                                      "features": features, "error": error, "t": time.time()}))
