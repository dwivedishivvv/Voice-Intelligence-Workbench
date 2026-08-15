import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from common import db
from common.config import get_effective_settings
from ..auth import get_current_user
from ..services import queue

router = APIRouter(prefix="/v1/f1", tags=["f1"])

log = structlog.get_logger()

OPENF1 = "https://api.openf1.org/v1"


async def _proxy(path: str, params: dict):
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{OPENF1}/{path}", params={k: v for k, v in params.items() if v is not None})
        r.raise_for_status()
        return r.json()


def _ts(value):
    """OpenF1 ISO-8601 string -> aware datetime, for binding to a TIMESTAMPTZ column.

    asyncpg binds timestamps from datetime objects only and rejects the raw string
    outright, so without this every timestamped upsert -- sessions, laps, radio calls --
    fails the whole batch while the untimestamped ones (drivers) succeed. That failure is
    quiet by design: _persist logs and continues so the endpoint still serves OpenF1 data,
    which is exactly why it needs a test rather than a code read to catch.

    An unparseable value becomes NULL rather than raising. That is the honest reading --
    the time really is unknown -- and it fails safe downstream: DURING_LAP requires a
    non-null timestamp, so such a row is left off the lap timeline instead of being placed
    on it wrongly."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    # Naive input is documented UTC; TIMESTAMPTZ needs the offset to be explicit.
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


async def _persist(label: str, sql: str, rows: list[tuple]):
    """Write-through cache of OpenF1 data. A completed session's results never change, so
    upserting on every fetch is both safe and the whole cost of keeping local history.

    Deliberately non-fatal: this endpoint's job is to return OpenF1 data, and a failed
    cache write does not make the returned payload wrong. Logged rather than raised so a
    database hiccup degrades persistence, not the Race Radio page."""
    if not rows:
        return
    try:
        await db.pool().executemany(sql, rows)
    except Exception as e:
        log.warning("f1_persist_failed", what=label, n=len(rows), error=str(e)[:200])


@router.get("/sessions")
async def sessions(year: int | None = None, session_type: str = "Race", user=Depends(get_current_user)):
    data = await _proxy("sessions", {"year": year, "session_type": session_type})
    await _persist("sessions", """
        INSERT INTO f1_sessions (session_key, meeting_key, year, name, session_type,
                                 circuit, country, location, date_start, synced_at)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9, now())
        ON CONFLICT (session_key) DO UPDATE SET
          meeting_key=EXCLUDED.meeting_key, year=EXCLUDED.year, name=EXCLUDED.name,
          session_type=EXCLUDED.session_type, circuit=EXCLUDED.circuit,
          country=EXCLUDED.country, location=EXCLUDED.location,
          date_start=EXCLUDED.date_start, synced_at=now()""",
        [(s.get("session_key"), s.get("meeting_key"), s.get("year"), s.get("session_name"),
          s.get("session_type"), s.get("circuit_short_name"), s.get("country_name"),
          s.get("location"), _ts(s.get("date_start"))) for s in data if s.get("session_key")])
    return data


@router.get("/drivers")
async def drivers(session_key: int, user=Depends(get_current_user)):
    data = await _proxy("drivers", {"session_key": session_key})
    # FK to f1_sessions: a driver fetched for a session we've never listed (deep link,
    # or a session_type filter that excluded it) would violate it, so make sure the
    # parent row exists first. Only the key is known here; the rest fills in on /sessions.
    await _persist("session_stub",
        "INSERT INTO f1_sessions (session_key) VALUES ($1) ON CONFLICT DO NOTHING",
        [(session_key,)])
    await _persist("drivers", """
        INSERT INTO f1_drivers (session_key, driver_number, full_name, name_acronym,
                                team_name, team_colour, headshot_url)
        VALUES ($1,$2,$3,$4,$5,$6,$7)
        ON CONFLICT (session_key, driver_number) DO UPDATE SET
          full_name=EXCLUDED.full_name, name_acronym=EXCLUDED.name_acronym,
          team_name=EXCLUDED.team_name, team_colour=EXCLUDED.team_colour,
          headshot_url=EXCLUDED.headshot_url""",
        [(session_key, d.get("driver_number"), d.get("full_name"), d.get("name_acronym"),
          d.get("team_name"), d.get("team_colour"), d.get("headshot_url"))
         for d in data if d.get("driver_number") is not None])
    return data


@router.get("/laps")
async def laps(session_key: int, driver_number: int, user=Depends(get_current_user)):
    data = await _proxy("laps", {"session_key": session_key, "driver_number": driver_number})
    await _persist("laps", """
        INSERT INTO f1_laps (session_key, driver_number, lap_number, date_start,
                             lap_duration, sector_1, sector_2, sector_3, is_pit_out)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
        ON CONFLICT (session_key, driver_number, lap_number) DO UPDATE SET
          date_start=EXCLUDED.date_start, lap_duration=EXCLUDED.lap_duration,
          sector_1=EXCLUDED.sector_1, sector_2=EXCLUDED.sector_2,
          sector_3=EXCLUDED.sector_3, is_pit_out=EXCLUDED.is_pit_out""",
        [(session_key, driver_number, l.get("lap_number"), _ts(l.get("date_start")),
          l.get("lap_duration"), l.get("duration_sector_1"), l.get("duration_sector_2"),
          l.get("duration_sector_3"), l.get("is_pit_out_lap"))
         for l in data if l.get("lap_number") is not None])
    return data


@router.get("/team_radio")
async def team_radio(session_key: int, driver_number: int | None = None, user=Depends(get_current_user)):
    data = await _proxy("team_radio", {"session_key": session_key, "driver_number": driver_number})
    # Recorded up front, before anyone asks for analysis, so `recorded_at` is captured
    # from OpenF1's own timestamp rather than inferred later from when we happened to
    # process the file. DO NOTHING on conflict: an existing row may already hold an
    # analysis, and this fetch knows nothing that would improve it.
    await _persist("team_radio", """
        INSERT INTO radio_calls (session_key, driver_number, recording_url, recorded_at)
        VALUES ($1,$2,$3,$4) ON CONFLICT (recording_url) DO NOTHING""",
        [(session_key, r.get("driver_number"), r.get("recording_url"), _ts(r.get("date")))
         for r in data if r.get("recording_url")])
    return data


@router.get("/analyses")
async def analyses(session_key: int, user=Depends(get_current_user)):
    """Every stored analysis for a session, so the page can show past results on load
    instead of starting blank and re-analyzing everything that was already done."""
    rows = await db.fetch(
        """SELECT recording_url, driver_number, text, mood, features, error, analyzed_at
           FROM radio_calls WHERE session_key=$1 AND analyzed_at IS NOT NULL""",
        session_key)
    return [{**dict(r), "features": json.loads(r["features"]) if r["features"] else None}
            for r in rows]


class IngestBody(BaseModel):
    recording_url: str
    # optional — when given, tone analysis calibrates against this driver's other radio
    # calls in the same session instead of a fixed global pitch/rate threshold
    session_key: int | None = None
    driver_number: int | None = None
    # skip the stored result and re-run (settings changed, or the stored one looks wrong)
    force: bool = False


@router.post("/ingest")
async def ingest(body: IngestBody, user=Depends(get_current_user)):
    cfg = await get_effective_settings()
    if not body.recording_url.startswith("https://livetiming.formula1.com/"):
        raise HTTPException(400, "recording_url must be an official F1 livetiming asset")

    row = await db.fetchrow(
        """INSERT INTO radio_calls (session_key, driver_number, recording_url)
           VALUES ($1,$2,$3)
           ON CONFLICT (recording_url) DO UPDATE SET
             session_key=COALESCE(radio_calls.session_key, EXCLUDED.session_key),
             driver_number=COALESCE(radio_calls.driver_number, EXCLUDED.driver_number)
           RETURNING id, text, mood, features, error, analyzed_at""",
        body.session_key, body.driver_number, body.recording_url)
    call_id = str(row["id"])

    # The payoff for persisting: re-analyzing a call we've already done is a database
    # read, not a download plus a GPU transcription. "Analyze all" over a session that
    # has been seen before returns immediately.
    if row["analyzed_at"] and not row["error"] and not body.force:
        return {"radio_call_id": call_id, "cached": True, "text": row["text"],
                "mood": row["mood"],
                "features": json.loads(row["features"]) if row["features"] else None}

    rel_path = f"tmp/f1/{call_id}.mp3"
    abs_path = Path(cfg.data_dir) / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(body.recording_url)
        r.raise_for_status()
        abs_path.write_bytes(r.content)

    await queue.enqueue_f1_radio(call_id, rel_path, body.session_key, body.driver_number)
    return {"radio_call_id": call_id, "cached": False,
            "job_id": call_id, "ws_url": f"/v1/ws/jobs/{call_id}"}
