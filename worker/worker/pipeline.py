import json
import os
import time
import uuid

import structlog

from common import db, graph_sync
from common.config import config_snapshot
from common.graph import GraphUnavailable
from .ctx import Ctx
from .errors import RejectError
from common.events import emit
from .stages import (validate, preprocess, transcribe, diarize, reconcile,
                      embed, identify, postprocess, sentiment, index)

log = structlog.get_logger()


async def _sync_graph(cfg):
    # Best-effort: Neo4j is a derived read model (GRAPH_RAG_PLAN.md), and an optional
    # datastore being down or slow must never fail an otherwise-successful clip. Full
    # rebuild rather than incremental — see the ponytail note on graph_sync.rebuild().
    if not cfg.graph_enabled:
        return
    try:
        await graph_sync.rebuild(cfg)
    except GraphUnavailable:
        pass
    except Exception as e:
        log.warning("graph_sync_failed", error=str(e)[:200])

PIPELINE_VERSION = os.environ.get("PIPELINE_VERSION", "dev")

STAGES = [
    ("VALIDATING", validate.run),
    ("PREPROCESSING", preprocess.run),
    ("TRANSCRIBING", transcribe.run),
    ("DIARIZING", diarize.run),
    ("RECONCILING", reconcile.run),
    ("EMBEDDING", embed.run),
    ("IDENTIFYING", identify.run),
    ("POSTPROCESSING", postprocess.run),
    ("SENTIMENT", sentiment.run),
    ("INDEXING", index.run),
]

# Relative share of total wall-clock time each stage typically takes, for a UI progress bar.
# Fixed weights rather than live-averaged timings: simpler, and "roughly right" is all a
# progress bar needs — an ETA that's occasionally 20% off beats no ETA at all.
STAGE_WEIGHTS = {
    "VALIDATING": 2, "PREPROCESSING": 14, "TRANSCRIBING": 33, "DIARIZING": 23,
    "RECONCILING": 3, "EMBEDDING": 5, "IDENTIFYING": 3, "POSTPROCESSING": 5,
    "SENTIMENT": 7, "INDEXING": 5,
}
_CUM_BEFORE = {}
_running = 0
for _name, _ in STAGES:
    _CUM_BEFORE[_name] = _running
    _running += STAGE_WEIGHTS[_name]


# Tables holding derived-per-clip data that stages INSERT into. Deleted (not upserted)
# at the start of every run — including the first — so reprocess() doesn't hit unique
# constraints (quality_metrics.clip_id is a PK) or leave stale duplicate rows behind
# from a previous attempt (vad_regions, speaker_turns have no such constraint, so without
# this they'd silently accumulate duplicates instead of erroring). ON DELETE CASCADE on
# transcripts/clip_id takes utterances and words with it.
_DERIVED_TABLES = ["quality_metrics", "vad_regions", "speaker_turns", "transcripts", "clip_speakers"]


async def _reset_derived_data(clip_id: str):
    for table in _DERIVED_TABLES:
        await db.execute(f"DELETE FROM {table} WHERE clip_id=$1", clip_id)
    # The clip-level sentiment rollup lives in columns on clips, not in one of the tables
    # above, so it survived a rerun. A clip that processed fine once and is REJECTED the
    # next time kept displaying its old sentiment badge next to "no speech" — stale
    # analysis presented as current. Clear it here, with everything else derived.
    await db.execute(
        "UPDATE clips SET sentiment=NULL, sentiment_score=NULL, mood=NULL WHERE id=$1", clip_id)


async def process_clip(clip_id: str, pool, cfg):
    await _reset_derived_data(clip_id)
    corr = uuid.uuid4().hex[:12]
    run_id = await db.insert("processing_runs", {
        "clip_id": clip_id, "pipeline_version": PIPELINE_VERSION,
        "config_snapshot": config_snapshot(cfg), "model_versions": pool.versions,
        "device": pool.device, "corr_id": corr,
    })
    ctx = Ctx(clip_id=clip_id, run_id=run_id, corr_id=corr, cfg=cfg, pool=pool)

    t0 = time.perf_counter()
    try:
        for stage_name, fn in STAGES:
            await db.execute("UPDATE clips SET status=$2 WHERE id=$1", clip_id, stage_name)
            await emit(clip_id, stage_name, "started", progress=_CUM_BEFORE[stage_name])
            t = time.perf_counter()
            try:
                await fn(ctx)
            except RejectError as e:
                await db.execute(
                    "UPDATE clips SET status='REJECTED', error_code=$2, error_detail=$3 WHERE id=$1",
                    clip_id, e.code, e.detail)
                await db.execute(
                    """UPDATE processing_runs SET finished_at=now(), outcome='rejected',
                       error_code=$2, error_detail=$3, total_ms=$4 WHERE id=$1""",
                    run_id, e.code, e.detail, int((time.perf_counter() - t0) * 1000))
                await emit(clip_id, stage_name, "rejected", code=e.code, message=e.detail,
                           progress=_CUM_BEFORE[stage_name])
                return
            ms = int((time.perf_counter() - t) * 1000)
            ctx.timings[stage_name] = ms
            progress = _CUM_BEFORE[stage_name] + STAGE_WEIGHTS[stage_name]
            await emit(clip_id, stage_name, "done", ms=ms, progress=progress,
                       elapsed_ms=int((time.perf_counter() - t0) * 1000))

        total_ms = int((time.perf_counter() - t0) * 1000)
        await db.execute(
            """UPDATE processing_runs SET finished_at=now(), outcome='success', total_ms=$2,
               stage_timings=$3, warnings=$4 WHERE id=$1""",
            run_id, total_ms, json.dumps(ctx.timings), json.dumps(ctx.warnings))
        await db.execute("UPDATE clips SET status='COMPLETE' WHERE id=$1", clip_id)
        await emit(clip_id, "COMPLETE", "done", total_ms=total_ms, progress=100)
        await _sync_graph(cfg)

    except Exception as e:
        await db.execute(
            "UPDATE clips SET status='FAILED', error_code=$2, error_detail=$3 WHERE id=$1",
            clip_id, type(e).__name__, str(e)[:2000])
        await db.execute(
            """UPDATE processing_runs SET finished_at=now(), outcome='failed', error_code=$2,
               error_detail=$3, total_ms=$4 WHERE id=$1""",
            run_id, type(e).__name__, str(e)[:2000], int((time.perf_counter() - t0) * 1000))
        await emit(clip_id, "FAILED", "error", message=str(e)[:200])
        raise
