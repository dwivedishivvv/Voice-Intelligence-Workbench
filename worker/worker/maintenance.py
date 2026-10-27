"""Background upkeep the worker runs on a timer (main.WorkerSettings.cron_jobs).

Each task here used to be either a manual step or not done at all:

  - stalled clips    — a worker killed mid-job (OOM, restart, arq timeout cancelling the
                       coroutine past process_clip's `except Exception`) left its clip in an
                       intermediate status forever, showing "TRANSCRIBING" to everyone.
  - orphaned queue   — a clip marked QUEUED whose arq job no longer exists (Redis flushed or
                       restarted without persistence) waited for a job that would never come.
  - retention        — retention_days was configurable and surfaced, and nothing enforced it.
  - tag backfill     — clips processed before the TAGGING stage existed, or whose tagging
                       failed, have auto_tags NULL until something tags them.
  - clusters         — consolidate_clusters() only ever ran from an admin button. Opt-in
                       here (auto_consolidate_clusters), for the reason given in config.py.

Every task that changes something writes an audit row, so autonomous changes leave the same
tamper-evident trail a person's do. A tick that finds nothing to do writes nothing.
"""
import uuid

import structlog

from common import db, storage, speaker as sc
from common.audit import audit
from .stages import tag as tag_stage

log = structlog.get_logger()

ACTOR = "system:maintenance"
STALLED_BATCH = 50
ORPHAN_BATCH = 50
RETENTION_BATCH = 200
BACKFILL_BATCH = 25
# A QUEUED clip younger than this may simply be between the api's status write and its
# enqueue call; only older ones are worth checking for a missing job.
ORPHAN_GRACE_S = 120

TERMINAL = ("COMPLETE", "FAILED", "REJECTED", "DEAD")


def stall_cutoff_s(cfg) -> float:
    # arq cancels a job at job_timeout_s, so a run unfinished well past that is not running.
    # The floor keeps a deliberately tiny job_timeout_s from flagging live work.
    return float(max(cfg.stalled_after_s, cfg.job_timeout_s + 60))


async def recover_stalled(cfg) -> list[str]:
    """Mark clips a lost worker left mid-pipeline as FAILED, with a code that says so.

    FAILED rather than re-queued: arq already retries a job whose worker died, so a clip
    still stuck after that has failed every attempt it was going to get, and the honest
    state is the one the UI already knows how to offer Reprocess on.
    """
    rows = await db.fetch(
        f"""SELECT c.id, r.id AS run_id FROM clips c
            JOIN LATERAL (SELECT id, started_at, finished_at FROM processing_runs pr
                          WHERE pr.clip_id = c.id ORDER BY started_at DESC LIMIT 1) r ON true
            WHERE c.status::text NOT IN ('QUEUED', {", ".join(f"'{s}'" for s in TERMINAL)})
              AND r.finished_at IS NULL
              AND r.started_at < now() - make_interval(secs => $1)
            LIMIT {STALLED_BATCH}""", stall_cutoff_s(cfg))
    ids = []
    for r in rows:
        detail = "processing stopped without finishing; the worker running it was lost"
        await db.execute(
            "UPDATE clips SET status='FAILED', error_code='STALLED', error_detail=$2 WHERE id=$1",
            r["id"], detail)
        await db.execute(
            """UPDATE processing_runs SET finished_at=now(), outcome='failed',
                   error_code='STALLED', error_detail=$2 WHERE id=$1""", r["run_id"], detail)
        ids.append(str(r["id"]))
    return ids


async def requeue_orphans(redis) -> list[str]:
    """Re-enqueue QUEUED clips whose arq job is gone.

    The api names each job `clip-{clip_id}-{suffix}` (services/queue.py), and arq keeps an
    `arq:job:<id>` key from enqueue until the job finishes — so no such key means nothing
    is waiting to process this clip.
    """
    rows = await db.fetch(
        f"""SELECT c.id FROM clips c
            WHERE c.status = 'QUEUED'
              AND c.created_at < now() - make_interval(secs => $1)
              AND NOT EXISTS (SELECT 1 FROM processing_runs pr WHERE pr.clip_id = c.id
                              AND pr.started_at > now() - make_interval(secs => $1))
            LIMIT {ORPHAN_BATCH}""", float(ORPHAN_GRACE_S))
    ids = []
    for r in rows:
        cid = str(r["id"])
        if await redis.keys(f"arq:job:clip-{cid}-*"):
            continue
        await redis.enqueue_job("process_clip_job", cid,
                                _job_id=f"clip-{cid}-{uuid.uuid4().hex[:8]}")
        ids.append(cid)
    return ids


async def enforce_retention(cfg) -> list[str]:
    """Delete clips older than retention_days, the same way a person's delete does: files,
    row (cascading to everything derived), and a deletion receipt each."""
    if cfg.retention_days <= 0:
        return []
    rows = await db.fetch(
        f"""SELECT id, sha256 FROM clips
            WHERE created_at < now() - make_interval(days => $1)
              AND status::text IN ({", ".join(f"'{s}'" for s in TERMINAL)})
            ORDER BY created_at LIMIT {RETENTION_BATCH}""", int(cfg.retention_days))
    ids = []
    for r in rows:
        cid = str(r["id"])
        storage.delete_clip_files(cfg, cid)
        await db.execute("DELETE FROM clips WHERE id=$1", cid)
        await db.insert("deletion_receipts", {"object_type": "clip", "object_id": cid,
                                               "sha256": r["sha256"], "reason": "retention"})
        ids.append(cid)
    return ids


async def backfill_tags(pool, cfg, limit: int = BACKFILL_BATCH) -> int:
    """Tag COMPLETE clips that have no auto tags yet. Caller holds the model lane."""
    if not cfg.auto_tag_enabled:
        return 0
    rows = await db.fetch(
        """SELECT id FROM clips WHERE status='COMPLETE' AND auto_tags IS NULL
           ORDER BY created_at DESC LIMIT $1""", limit)
    n = 0
    for r in rows:
        try:
            await tag_stage.tag_clip(str(r["id"]), pool, cfg)
            n += 1
        except Exception as e:
            # an empty array means "tagged, nothing applied" — without it a clip that always
            # fails would be retried every tick forever
            log.warning("tag_backfill_failed", clip_id=str(r["id"]), error=str(e)[:200])
            await db.execute(
                "UPDATE clips SET auto_tags='{}', tagged_at=now() WHERE id=$1", r["id"])
    return n


async def run(ctx, cfg) -> dict:
    """One pass of every task. Each is isolated: one failing never stops the others."""
    summary: dict = {}

    async def step(name, coro):
        try:
            summary[name] = await coro
        except Exception as e:
            log.warning("maintenance_step_failed", step=name, error=str(e)[:200])
            summary[name] = {"error": str(e)[:200]}

    await step("stalled", recover_stalled(cfg))
    await step("requeued", requeue_orphans(ctx["redis"]))
    await step("retention_deleted", enforce_retention(cfg))
    async with ctx["gpu_sem"]:
        await step("tagged", backfill_tags(ctx["pool"], cfg))
    if cfg.auto_consolidate_clusters:
        await step("clusters", sc.consolidate_clusters(cfg))

    for key, action in (("stalled", "maintenance.stalled"), ("requeued", "maintenance.requeue"),
                        ("retention_deleted", "maintenance.retention")):
        if isinstance(summary.get(key), list) and summary[key]:
            await audit(action, "clip", "batch", actor=ACTOR,
                        after={"count": len(summary[key]), "clip_ids": summary[key]})
    clusters = summary.get("clusters")
    if isinstance(clusters, dict) and clusters.get("merged"):
        await audit("maintenance.consolidate", "speaker_cluster", "all", actor=ACTOR, after=clusters)
    if isinstance(summary.get("tagged"), int) and summary["tagged"]:
        await audit("maintenance.tag_backfill", "clip", "batch", actor=ACTOR,
                    after={"count": summary["tagged"]})
    return summary
