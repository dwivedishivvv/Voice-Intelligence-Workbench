"""Bring an existing database up to the current schema.

db/init.sql only runs on a fresh postgres volume, so every later schema change is written
as a replayable file (tests/unit/test_sql_replayable.py enforces that) and used to need a
manual `psql < db/x.sql` against each existing deployment — a step that was easy to miss
and only surfaced later as a missing column. The api now replays them itself at startup.
"""
from pathlib import Path

import structlog

from . import db

log = structlog.get_logger(__name__)

DB_DIR = Path(__file__).resolve().parent.parent / "db"

# retire_f1_sentiment.sql drops what the removed F1 and sentiment features created on
# databases that predate their removal; it is a no-op on a fresh volume.
REPLAYABLE = ["speaker_quality.sql", "tagging.sql", "speaker_space.sql",
              "retire_f1_sentiment.sql"]


async def apply_replayable() -> list[str]:
    """Replay every idempotent schema file; returns the ones that applied.

    Best-effort per file: a file that fails is logged and skipped rather than stopping the
    api from starting, because the api serves plenty that a missing optional column does
    not touch. A missing db/ directory (an image built without it) is not an error either.
    """
    applied = []
    for name in REPLAYABLE:
        path = DB_DIR / name
        if not path.exists():
            continue
        try:
            # No bind parameters, so asyncpg sends this over the simple-query protocol,
            # which accepts a whole multi-statement script in one round trip.
            await db.execute(path.read_text(encoding="utf-8"))
            applied.append(name)
        except Exception as e:
            log.warning("schema_replay_failed", file=name, error=str(e)[:200])
    return applied
