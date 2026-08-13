import hashlib
import json
from datetime import datetime, timezone
from . import db


async def audit(action: str, object_type: str, object_id, before=None, after=None,
                 actor: str | None = None, corr_id: str | None = None):
    prev = await db.fetchval("SELECT row_hash FROM audit_log ORDER BY id DESC LIMIT 1")
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(), "actor": actor, "action": action,
        "object_type": object_type, "object_id": str(object_id),
        "before": before, "after": after, "prev": prev,
    }
    row_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    await db.execute(
        """INSERT INTO audit_log(actor_id, action, object_type, object_id, before, after,
                                  corr_id, prev_hash, row_hash)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)""",
        actor, action, object_type, str(object_id),
        json.dumps(before, default=str) if before is not None else None,
        json.dumps(after, default=str) if after is not None else None,
        corr_id, prev, row_hash,
    )
