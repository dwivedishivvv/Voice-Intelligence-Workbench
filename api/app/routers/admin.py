from typing import get_args

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from common import db, speaker as sc, graph_sync
from common.audit import audit
from common.graph import GraphUnavailable
from common.config import (get_settings, get_effective_settings, config_snapshot,
                            Settings, TUNABLE_FIELDS, RESTART_TUNABLE_FIELDS,
                            SETTINGS_CATEGORIES, LLM_MODELS)
from ..auth import get_current_user
from ..services import agent as agent_svc

router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.get("/config")
async def get_config(user=Depends(get_current_user)):
    return config_snapshot(get_settings())


# Fields ModelPool bakes in at worker startup — shown read-only on the Settings page so it's
# obvious *why* they can't be edited there, instead of just omitting them silently.
RESTART_REQUIRED_FIELDS = ["device", "precision", "worker_concurrency", "model_dir",
                           "asr_model", "vad_model", "diar_model", "embed_model", "text_embed_model"]


@router.get("/settings")
async def get_settings_view(user=Depends(get_current_user)):
    defaults = get_settings()
    effective = await get_effective_settings()
    override_rows = {r["key"]: r for r in await db.fetch(
        "SELECT key, value, updated_at FROM settings_overrides")}

    def field(name: str):
        f = Settings.model_fields[name]
        return {
            "key": name, "value": getattr(effective, name), "default": getattr(defaults, name),
            "overridden": name in override_rows,
            "updated_at": override_rows[name]["updated_at"].isoformat() if name in override_rows else None,
            "type": "bool" if f.annotation is bool else "int" if f.annotation is int
                    else "float" if f.annotation is float else "str",
        }

    def restart_field(name: str):
        pending = override_rows.get(name)
        ann = Settings.model_fields[name].annotation
        return {"key": name, "value": getattr(defaults, name),
                "pending": pending["value"] if pending else None,
                "editable": name in RESTART_TUNABLE_FIELDS,
                # Literal fields carry their own allowed values — hand them to the UI so it
                # can offer a picker instead of a free-text box that can only be got wrong.
                "options": list(get_args(ann)) or None}

    return {
        "categories": [{"name": cat, "fields": [field(k) for k in keys]}
                        for cat, keys in SETTINGS_CATEGORIES.items()],
        "restart_required": [restart_field(k) for k in RESTART_REQUIRED_FIELDS],
    }


class SettingsPatch(BaseModel):
    values: dict[str, bool | int | float | str]


@router.patch("/settings")
async def patch_settings(body: SettingsPatch, user=Depends(get_current_user)):
    unknown = set(body.values) - TUNABLE_FIELDS - RESTART_TUNABLE_FIELDS
    if unknown:
        raise HTTPException(400, f"not tunable: {sorted(unknown)}")
    for key, value in body.values.items():
        try:
            # validate against a fresh dict, not get_settings()'s cached singleton — that
            # instance is read everywhere else in the process, so mutating it in place to
            # validate would corrupt every subsequent request until the worker restarts
            Settings.model_validate({**get_settings().model_dump(), key: value})
        except Exception as e:
            raise HTTPException(400, f"{key}: {e}") from e
        await db.execute(
            """INSERT INTO settings_overrides (key, value, updated_at, updated_by)
               VALUES ($1, $2, now(), $3)
               ON CONFLICT (key) DO UPDATE SET value=$2, updated_at=now(), updated_by=$3""",
            key, str(value), user)
    return {"ok": True}


@router.delete("/settings/{setting_key}")
async def reset_setting(setting_key: str, user=Depends(get_current_user)):
    await db.execute("DELETE FROM settings_overrides WHERE key=$1", setting_key)
    return {"ok": True}


def _key_hint(key: str) -> str:
    """Enough to tell one key from another, not enough to use. Four characters of a 40+
    character secret identifies which key is installed without handing it back."""
    return f"…{key[-4:]}" if len(key) > 8 else ("set" if key else "")


@router.get("/llm")
async def get_llm_config(user=Depends(get_current_user)):
    """The agent's configuration, with the keys deliberately absent.

    A GET that returned the key would put it in the browser's memory, its devtools, and
    anything that ever logs a response body — for a value the UI only needs to *replace*,
    never to read. So each provider reports whether a key is installed and the last four
    characters, which is what an operator needs to tell "already configured" from "empty"
    and one key from another.
    """
    cfg = await get_effective_settings()
    return {
        "enabled": cfg.llm_enabled,
        "provider": cfg.llm_provider,
        "model": cfg.llm_model,
        "providers": [
            {"name": name,
             "models": models,
             "default_model": agent_svc.DEFAULT_MODELS.get(name, ""),
             "key_set": bool(getattr(cfg, f"{name}_api_key", "")),
             "key_hint": _key_hint(getattr(cfg, f"{name}_api_key", "")),
             "base_url": getattr(cfg, f"{name}_base_url", None)}
            for name, models in LLM_MODELS.items()
        ],
    }


class LLMPatch(BaseModel):
    enabled: bool | None = None
    provider: str | None = None
    model: str | None = None
    # Which provider the key belongs to, and the key. Sent only when it changes; the UI
    # never holds the existing one, so an empty string here means "clear it" rather than
    # "unchanged" — the caller has to be explicit either way.
    api_key_provider: str | None = None
    api_key: str | None = None


@router.patch("/llm")
async def patch_llm_config(body: LLMPatch, user=Depends(get_current_user)):
    """Configure the agent. The only write path for these fields — the generic settings
    PATCH refuses all of them, so turning the agent on or repointing it at another vendor
    cannot happen as a side effect of a caller sweeping the settings surface."""
    values: dict[str, bool | str] = {}
    if body.enabled is not None:
        values["llm_enabled"] = body.enabled
    if body.provider is not None:
        if body.provider not in LLM_MODELS:
            raise HTTPException(400, f"unknown provider {body.provider!r}; "
                                     f"expected one of {', '.join(sorted(LLM_MODELS))}")
        values["llm_provider"] = body.provider
    if body.model is not None:
        values["llm_model"] = body.model
    if body.api_key is not None:
        if body.api_key_provider not in LLM_MODELS:
            raise HTTPException(400, "api_key needs api_key_provider naming which provider "
                                     "it belongs to")
        values[f"{body.api_key_provider}_api_key"] = body.api_key

    for key, value in values.items():
        await db.execute(
            """INSERT INTO settings_overrides (key, value, updated_at, updated_by)
               VALUES ($1, $2, now(), $3)
               ON CONFLICT (key) DO UPDATE SET value=$2, updated_at=now(), updated_by=$3""",
            key, str(value), user)

    # Names only. The audit log is append-only and readable through the API, so writing the
    # key here would durably publish the secret this endpoint exists to keep out of reads.
    await audit("agent.configure", "settings", "llm", actor=user,
                after={"changed": sorted(values)})
    return await get_llm_config(user)


@router.post("/calibrate")
async def calibrate(user=Depends(get_current_user)):
    return await sc.calibrate()


@router.get("/calibration/latest")
async def latest_calibration(user=Depends(get_current_user)):
    row = await db.fetchrow("SELECT * FROM calibration_runs ORDER BY ran_at DESC LIMIT 1")
    return dict(row) if row else {}


@router.get("/corrections")
async def corrections(days: int = 30, user=Depends(get_current_user)):
    rows = await db.fetch(
        """SELECT * FROM audit_log WHERE action='identification.correct'
           AND ts > now() - ($1 || ' days')::interval ORDER BY ts DESC""", str(days))
    return {"items": [dict(r) for r in rows]}


@router.get("/stats")
async def stats(user=Depends(get_current_user)):
    total = await db.fetchval("SELECT count(*) FROM clips")
    by_status = await db.fetch("SELECT status, count(*) c FROM clips GROUP BY status")
    by_result = await db.fetch("SELECT match_result, count(*) c FROM clip_speakers GROUP BY match_result")
    return {"total_clips": total,
            "by_status": {r["status"]: r["c"] for r in by_status},
            "identification_results": {r["match_result"]: r["c"] for r in by_result}}


@router.post("/graph/sync")
async def rebuild_graph(user=Depends(get_current_user)):
    """Rebuild the Neo4j projection from Postgres. Safe to run at any time — it wipes and
    re-projects a derived read model, so the worst case is a stale graph for the duration."""
    try:
        sent = await graph_sync.rebuild()
        got = await graph_sync.summary()
    except GraphUnavailable as e:
        # 503, not 500: an optional datastore being off is a deployment state, not a bug.
        raise HTTPException(503, str(e)) from e
    await audit("graph.sync", "graph", "neo4j", after={"projected": sent, **got}, actor=user)
    return {"projected": sent, **got}


@router.get("/graph/summary")
async def graph_summary(user=Depends(get_current_user)):
    try:
        return await graph_sync.summary()
    except GraphUnavailable as e:
        raise HTTPException(503, str(e)) from e


@router.get("/audit")
async def audit_log(limit: int = 100, user=Depends(get_current_user)):
    rows = await db.fetch("SELECT * FROM audit_log ORDER BY id DESC LIMIT $1", limit)
    return {"items": [dict(r) for r in rows]}
