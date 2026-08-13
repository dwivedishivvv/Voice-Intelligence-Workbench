"""Single get_current_user dependency — the documented seam for swapping in real
auth (SSO/RBAC/multi-tenant) later without touching callers."""
from fastapi import Header, HTTPException, Query
from common.config import get_settings


async def get_current_user(authorization: str | None = Header(default=None),
                            key: str | None = Query(default=None)) -> str:
    # `key` query param exists only so <a href>/<audio src> links (browsers can't set
    # headers on those) can still hit authenticated GET endpoints.
    cfg = get_settings()
    token = (authorization or "").removeprefix("Bearer ").strip() or (key or "")
    if token != cfg.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return "admin"
