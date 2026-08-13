from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import redis.asyncio as redis

from common.config import get_settings

router = APIRouter()


@router.websocket("/v1/ws/jobs/{job_id}")
async def job_progress(ws: WebSocket, job_id: str):
    await ws.accept()
    cfg = get_settings()
    r = redis.from_url(cfg.redis_url)
    pubsub = r.pubsub()
    await pubsub.subscribe(f"job:{job_id}")
    try:
        async for msg in pubsub.listen():
            if msg["type"] != "message":
                continue
            await ws.send_text(msg["data"].decode() if isinstance(msg["data"], bytes) else msg["data"])
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(f"job:{job_id}")
        await r.close()
