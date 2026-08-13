from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest

from common.config import get_settings
from common import db
from .observability import configure_logging
from .services import queue
from .routers import clips, speakers, clusters, search, admin, models, ws, live, f1


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    configure_logging(cfg.log_level)
    await db.init_pool(cfg)
    await queue.init_queue(cfg.redis_url)
    yield
    await queue.close_queue()
    await db.close_pool()


app = FastAPI(title="Speaker Intelligence Workbench", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

for r in (clips.router, speakers.router, clusters.router, search.router, admin.router, models.router, live.router, f1.router):
    app.include_router(r)
app.include_router(ws.router)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/readyz")
async def readyz():
    await db.fetchval("SELECT 1")
    return {"ok": True}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest())
