"""Redis pub/sub progress events.

Shared by the worker (pipeline stages, live chunks, radio analysis) and the api (agent
turns). Lives in common/ rather than the worker because both publish to the same channel:
`api/app/routers/ws.py` subscribes to `job:{id}` and forwards whatever it finds, so a new
producer needs no new transport, no new route and no frontend change.
"""
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


async def emit_live(session_id: str, seq: int, turns: list[dict], error: str | None = None):
    """One chunk, one or more diarized speaker turns (worker/main.py now runs the same
    diarize -> assign_words -> regroup pipeline a batch clip does, scoped to one short
    chunk). Each turn dict already carries text/mood/speaker/speaker_result/speaker_score/
    start_s/end_s — forwarded as-is, so a client only needs to know the list can hold more
    than one entry now. speaker_result is identify()'s own vocabulary (confident |
    suggested); "unknown" is never sent, so a client can't mistake abstention for a guess.

    Reuses the same job:{id} pubsub channel / WS route as clip processing — the WS endpoint
    just forwards whatever's published, no live-specific plumbing needed.
    """
    if _redis is None:
        return
    await _redis.publish(f"job:{session_id}",
                          json.dumps({"type": "live_transcript", "seq": seq,
                                      "turns": turns, "error": error, "t": time.time()}))


async def emit_f1_result(job_id: str, text: str, mood: str, features: dict | None, error: str | None = None):
    if _redis is None:
        return
    await _redis.publish(f"job:{job_id}",
                          json.dumps({"type": "f1_result", "text": text, "mood": mood,
                                      "features": features, "error": error, "t": time.time()}))


async def emit_agent(conversation_id: str, kind: str, **kw):
    """Agent progress. Rides the same `job:{id}` channel as everything else, so the
    existing websocket route and the frontend hook both work unchanged.

    `kind` is the event's own type (thinking / tool_use / message / done / error) and is
    nested under a fixed outer "agent" type so a client can route on `msg.type` exactly as
    it does for stage / live_transcript / f1_result."""
    if _redis is None:
        return
    await _redis.publish(f"job:{conversation_id}",
                          json.dumps({"type": "agent", "kind": kind, "t": time.time(), **kw}))


async def close_events():
    global _redis
    if _redis is not None:
        await _redis.close()
        _redis = None
