"""Races: a user-created grouping over clips, plus a track-outline SVG for the header.

A race owns no pipeline of its own. Recordings dropped on it go through the exact same
ingest and processing as any other clip (see services/ingest.py) — transcription,
diarization, ECAPA fingerprinting against the global speaker_profiles, and the SENTIMENT
stage — and the race is just the race_id they carry. That's what makes a driver enrolled
in one race identified automatically in the next.
"""
import re
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from fastapi.responses import Response
from pydantic import BaseModel

from common.config import get_effective_settings, get_settings
from common import db, storage, audit as audit_mod, speaker as sc
from ..auth import get_current_user
from ..services import ingest

router = APIRouter(prefix="/v1/races", tags=["races"])

MAX_SVG_BYTES = 2 * 1024 * 1024

# An uploaded SVG is untrusted markup that we serve back from our own origin, which makes it
# a stored-XSS vector by default: SVG can carry <script>, event handlers, and <foreignObject>
# with arbitrary HTML. We keep only shape/structure elements and drop everything else. This
# is an allowlist on purpose — a denylist of "known bad tags" is exactly the thing that gets
# bypassed by the next tag nobody thought of.
ALLOWED_TAGS = {
    "svg", "g", "path", "polyline", "polygon", "line", "rect", "circle", "ellipse",
    "defs", "title", "desc", "style", "marker", "linearGradient", "radialGradient", "stop",
}
# Attributes are allowlisted too, which drops every on* handler without having to enumerate
# them, and drops href/xlink:href entirely (no external fetches, no javascript: URLs).
ALLOWED_ATTRS = {
    "d", "points", "x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry",
    "width", "height", "viewBox", "transform", "fill", "stroke", "stroke-width",
    "stroke-linecap", "stroke-linejoin", "stroke-dasharray", "fill-rule", "clip-rule",
    "opacity", "fill-opacity", "stroke-opacity", "class", "id", "offset", "stop-color",
    "preserveAspectRatio", "gradientUnits", "gradientTransform",
}
SVG_NS = "http://www.w3.org/2000/svg"


def _sanitize_svg(raw: bytes) -> bytes:
    try:
        # forbid_dtd equivalent: ET doesn't expand external entities by default in 3.8+,
        # but a DOCTYPE is still a billion-laughs vehicle, so refuse it outright.
        if re.search(rb"<!DOCTYPE|<!ENTITY", raw, re.I):
            raise HTTPException(400, "SVG must not contain a DOCTYPE or entity declarations")
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        raise HTTPException(400, f"not a parseable SVG: {e}")

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    if local(root.tag) != "svg":
        raise HTTPException(400, "root element must be <svg>")

    def clean(el):
        for child in list(el):
            if not isinstance(child.tag, str) or local(child.tag) not in ALLOWED_TAGS:
                el.remove(child)  # comments/PIs have non-str tags and go here too
                continue
            clean(child)
        for name in list(el.attrib):
            if local(name) not in ALLOWED_ATTRS:
                del el.attrib[name]

    clean(root)
    # a <style> block survives as an element but its *text* is still author-controlled CSS;
    # url() in CSS can fetch, so drop the contents and keep only presentation attributes
    for el in root.iter():
        if local(el.tag) == "style":
            el.text = None

    if not root.get("viewBox") and root.get("width") and root.get("height"):
        w = re.sub(r"[^0-9.]", "", root.get("width") or "")
        h = re.sub(r"[^0-9.]", "", root.get("height") or "")
        if w and h:
            root.set("viewBox", f"0 0 {w} {h}")
    # let the page size it; a baked-in px width can't be responsive
    root.attrib.pop("width", None)
    root.attrib.pop("height", None)

    ET.register_namespace("", SVG_NS)
    return ET.tostring(root, encoding="utf-8")


async def _get_race(race_id: str):
    row = await db.fetchrow("SELECT * FROM races WHERE id=$1", race_id)
    if not row:
        raise HTTPException(404, "race not found")
    return row


@router.post("")
async def create_race(name: str = Form(...), circuit: str = Form(""), race_date: str = Form(""),
                       session_key: str = Form(""), notes: str = Form(""),
                       svg: UploadFile | None = File(None), user=Depends(get_current_user)):
    cfg = await get_effective_settings()
    if not name.strip():
        raise HTTPException(400, "name is required")

    # asyncpg binds a DATE column from a datetime.date, never a string
    try:
        parsed_date = date.fromisoformat(race_date) if race_date.strip() else None
    except ValueError:
        raise HTTPException(400, "race_date must be YYYY-MM-DD")
    try:
        parsed_key = int(session_key) if session_key.strip() else None
    except ValueError:
        raise HTTPException(400, "session_key must be an integer")

    race_id = await db.insert("races", {
        "name": name.strip(), "circuit": circuit.strip() or None,
        "race_date": parsed_date, "session_key": parsed_key,
        "notes": notes.strip() or None,
    })

    if svg is not None and svg.filename:
        raw = await svg.read()
        if len(raw) > MAX_SVG_BYTES:
            raise HTTPException(413, f"SVG exceeds {MAX_SVG_BYTES // 1024}KB")
        cleaned = _sanitize_svg(raw)
        rel = f"races/{race_id}/track.svg"
        dest = Path(storage.resolve(cfg, rel))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(cleaned)
        await db.execute("UPDATE races SET svg_path=$2 WHERE id=$1", race_id, rel)

    await audit_mod.audit("race.create", "race", race_id, after={"name": name}, actor=user)
    return await get_race(race_id, user)


@router.get("")
async def list_races(user=Depends(get_current_user)):
    rows = await db.fetch(
        """SELECT r.*, count(c.id) AS n_clips,
                  count(*) FILTER (WHERE c.status = 'COMPLETE') AS n_complete,
                  avg(c.sentiment_score) AS avg_sentiment
             FROM races r LEFT JOIN clips c ON c.race_id = r.id
            GROUP BY r.id ORDER BY r.created_at DESC""")
    return {"items": [dict(r) for r in rows]}


@router.get("/{race_id}")
async def get_race(race_id: str, user=Depends(get_current_user)):
    race = dict(await _get_race(race_id))
    clips = await db.fetch(
        """SELECT c.id, c.filename, c.status, c.duration_s, c.language, c.n_speakers,
                  c.sentiment, c.sentiment_score, c.mood, c.needs_review, c.created_at,
                  t.text AS transcript
             FROM clips c
             LEFT JOIN LATERAL (
                  SELECT text FROM transcripts WHERE clip_id = c.id
                   ORDER BY created_at DESC LIMIT 1
             ) t ON true
            WHERE c.race_id = $1 ORDER BY c.created_at""", race_id)
    race["clips"] = [dict(r) for r in clips]
    # who spoke across the whole race, resolved to enrolled profiles where identification
    # was confident. This is the payoff of reusing the global profiles: a driver enrolled
    # once shows up by name in every race they appear in.
    speakers = await db.fetch(
        """SELECT p.id, p.display_name, count(DISTINCT cs.clip_id) AS n_clips,
                  sum(cs.speech_s) AS speech_s, avg(cs.match_score) AS avg_match
             FROM clip_speakers cs
             JOIN clips c ON c.id = cs.clip_id
             JOIN speaker_profiles p ON p.id = cs.profile_id
            WHERE c.race_id = $1
            GROUP BY p.id, p.display_name ORDER BY speech_s DESC NULLS LAST""", race_id)
    race["speakers"] = [dict(r) for r in speakers]
    return race


@router.get("/{race_id}/svg")
async def get_race_svg(race_id: str, user=Depends(get_current_user)):
    cfg = await get_effective_settings()
    race = await _get_race(race_id)
    if not race["svg_path"]:
        raise HTTPException(404, "no track outline uploaded")
    data = Path(storage.resolve(cfg, race["svg_path"])).read_bytes()
    # sanitized at upload; the CSP and nosniff are the second layer, so a bug in the
    # sanitizer still can't turn this route into script execution on our origin
    return Response(data, media_type="image/svg+xml", headers={
        "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "no-cache",
    })


@router.post("/{race_id}/clips")
async def upload_race_clip(race_id: str, file: UploadFile = File(...),
                            user=Depends(get_current_user)):
    """One recording per request. Bulk upload is the client firing these concurrently —
    that keeps per-file progress and per-file failure independent, which a single
    multi-file request can't do (one bad file would fail the whole batch)."""
    cfg = await get_effective_settings()
    await _get_race(race_id)
    return await ingest.ingest_clip(file, cfg, user, race_id=race_id)


@router.get("/{race_id}/utterances")
async def race_utterances(race_id: str, cluster_id: str | None = None, profile_id: str | None = None,
                           sentiment: str | None = None, mood: str | None = None,
                           q: str | None = None, user=Depends(get_current_user)):
    """Flat, time-ordered view across every recording in the race — the transcript with
    speaker tags and per-utterance sentiment, which is what the race page actually renders.

    All filters are optional and combinable; omitting them all is the unfiltered view.
    Note cluster_id/profile_id filter on the clip_speakers row, not on local_label — see
    the comment in the SELECT for why local_label is never a cross-clip identity."""
    where, args = ["c.race_id = $1"], [race_id]

    def param(v):  # numbered placeholder; user input never reaches the SQL text itself
        args.append(v)
        return f"${len(args)}"

    if cluster_id:
        where.append(f"cs.cluster_id = {param(cluster_id)}")
    if profile_id:
        where.append(f"cs.profile_id = {param(profile_id)}")
    if sentiment:
        where.append(f"u.sentiment = {param(sentiment)}")
    if mood:
        where.append(f"u.mood = {param(mood)}")
    if q:
        where.append(f"u.text ILIKE '%' || {param(q)} || '%'")

    rows = await db.fetch(
        f"""SELECT u.id, u.clip_id, c.filename, u.idx, u.start_s, u.end_s, u.text,
                  u.local_label, coalesce(u.profile_id, cs.profile_id) AS profile_id,
                  p.display_name,
                  u.sentiment, u.sentiment_score, u.text_sentiment, u.mood,
                  u.mean_word_conf, cs.match_result, cs.match_score,
                  -- the only cross-clip identity for an unenrolled voice. local_label is
                  -- pyannote's per-clip index: "SPEAKER_00" in two different clips is two
                  -- different people, so the UI must never present it as one identity.
                  cs.cluster_id
             FROM utterances u
             JOIN clips c ON c.id = u.clip_id
             LEFT JOIN clip_speakers cs
                    ON cs.clip_id = u.clip_id AND cs.local_label = u.local_label
             -- clip_speakers is the authoritative identity row; utterances.profile_id can
             -- lag it (speaker.promote_cluster clears cs.cluster_id before the utterances
             -- UPDATE that keys off it), so resolve the name from either.
             LEFT JOIN speaker_profiles p ON p.id = coalesce(u.profile_id, cs.profile_id)
            WHERE {' AND '.join(where)}
            ORDER BY c.created_at, u.idx""", *args)
    return {"items": [dict(r) for r in rows]}


@router.get("/{race_id}/stats")
async def race_stats(race_id: str, user=Depends(get_current_user)):
    """Everything the analysis panel needs in one call:

      totals             {clips, complete, rejected, utterances, speech_s, voices}
      voices             one row per CROSS-CLIP identity (profile_id if enrolled, else
                         cluster_id), speech_s desc: {key, kind, display_name, n_clips,
                         speech_s, talk_share, n_utterances, avg_sentiment, mood_counts}
                         talk_share is this voice's share of race speech, recomputed here —
                         clip_speakers.talk_share is per-clip and doesn't sum across clips.
      sentiment_timeline per-clip rollup ordered by clips.created_at
      languages          [{language, n}]   quality [{grade, n}]   mood_counts over utterances
    """
    await _get_race(race_id)

    totals = await db.fetchrow(
        """SELECT count(*) AS clips,
                  count(*) FILTER (WHERE status = 'COMPLETE') AS complete,
                  count(*) FILTER (WHERE status = 'REJECTED') AS rejected,
                  coalesce(sum(duration_s), 0) AS duration_s
             FROM clips WHERE race_id = $1""", race_id)

    voices = await db.fetch(
        # speech and utterance counts have to be aggregated separately: joining utterances
        # into the clip_speakers grouping would multiply speech_s by the turn count.
        """WITH sp AS (
              SELECT coalesce(cs.profile_id::text, cs.cluster_id::text) AS key,
                     CASE WHEN cs.profile_id IS NOT NULL THEN 'profile' ELSE 'cluster' END AS kind,
                     max(p.display_name) AS display_name,
                     count(DISTINCT cs.clip_id) AS n_clips,
                     coalesce(sum(cs.speech_s), 0) AS speech_s
                FROM clip_speakers cs
                JOIN clips c ON c.id = cs.clip_id
                LEFT JOIN speaker_profiles p ON p.id = cs.profile_id
               WHERE c.race_id = $1 AND (cs.profile_id IS NOT NULL OR cs.cluster_id IS NOT NULL)
               GROUP BY 1, 2
           ), utt AS (
              SELECT coalesce(cs.profile_id::text, cs.cluster_id::text) AS key,
                     count(*) AS n_utterances, avg(u.sentiment_score) AS avg_sentiment,
                     count(*) FILTER (WHERE u.mood = 'calm') AS calm,
                     count(*) FILTER (WHERE u.mood = 'stressed') AS stressed,
                     count(*) FILTER (WHERE u.mood = 'tired') AS tired
                FROM utterances u
                JOIN clips c ON c.id = u.clip_id
                JOIN clip_speakers cs ON cs.clip_id = u.clip_id AND cs.local_label = u.local_label
               WHERE c.race_id = $1 AND (cs.profile_id IS NOT NULL OR cs.cluster_id IS NOT NULL)
               GROUP BY 1
           )
           SELECT sp.key, sp.kind, sp.display_name, sp.n_clips, sp.speech_s,
                  sp.speech_s / nullif(sum(sp.speech_s) OVER (), 0) AS talk_share,
                  coalesce(utt.n_utterances, 0) AS n_utterances, utt.avg_sentiment,
                  coalesce(utt.calm, 0) AS calm, coalesce(utt.stressed, 0) AS stressed,
                  coalesce(utt.tired, 0) AS tired
             FROM sp LEFT JOIN utt USING (key)
            ORDER BY sp.speech_s DESC""", race_id)

    timeline = await db.fetch(
        """SELECT id AS clip_id, filename, created_at, sentiment, sentiment_score
             FROM clips WHERE race_id = $1 ORDER BY created_at""", race_id)
    languages = await db.fetch(
        """SELECT language, count(*) AS n FROM clips
            WHERE race_id = $1 AND language IS NOT NULL GROUP BY 1 ORDER BY n DESC""", race_id)
    quality = await db.fetch(
        """SELECT q.grade, count(*) AS n FROM quality_metrics q JOIN clips c ON c.id = q.clip_id
            WHERE c.race_id = $1 AND q.grade IS NOT NULL GROUP BY 1 ORDER BY n DESC""", race_id)
    moods = await db.fetchrow(
        """SELECT count(*) AS n_utterances,
                  count(*) FILTER (WHERE u.mood = 'calm') AS calm,
                  count(*) FILTER (WHERE u.mood = 'stressed') AS stressed,
                  count(*) FILTER (WHERE u.mood = 'tired') AS tired
             FROM utterances u JOIN clips c ON c.id = u.clip_id WHERE c.race_id = $1""", race_id)

    return {
        "totals": {
            "clips": totals["clips"], "complete": totals["complete"],
            "rejected": totals["rejected"], "utterances": moods["n_utterances"],
            "speech_s": float(sum(v["speech_s"] or 0 for v in voices)) or float(totals["duration_s"] or 0),
            "voices": len(voices),
        },
        "voices": [{
            "key": v["key"], "kind": v["kind"], "display_name": v["display_name"],
            "n_clips": v["n_clips"], "speech_s": float(v["speech_s"] or 0),
            "talk_share": float(v["talk_share"] or 0), "n_utterances": v["n_utterances"],
            "avg_sentiment": float(v["avg_sentiment"]) if v["avg_sentiment"] is not None else None,
            "mood_counts": {"calm": v["calm"], "stressed": v["stressed"], "tired": v["tired"]},
        } for v in voices],
        "sentiment_timeline": [dict(r) for r in timeline],
        "languages": [dict(r) for r in languages],
        "quality": [dict(r) for r in quality],
        "mood_counts": {"calm": moods["calm"], "stressed": moods["stressed"], "tired": moods["tired"]},
    }


class NameVoiceBody(BaseModel):
    display_name: str


@router.post("/{race_id}/voices/{cluster_id}/name")
async def name_race_voice(race_id: str, cluster_id: str, body: NameVoiceBody,
                           user=Depends(get_current_user)):
    """Give an unenrolled cluster a name. This promotes it to a real speaker_profile, so the
    voice is recognised in FUTURE races too — that's the whole point of the global profiles."""
    await _get_race(race_id)
    if not body.display_name.strip():
        raise HTTPException(400, "display_name is required")
    row = await db.fetchrow("SELECT promoted_to FROM speaker_clusters WHERE id=$1", cluster_id)
    if not row:
        raise HTTPException(404, "cluster not found")
    if row["promoted_to"]:
        raise HTTPException(404, "cluster already promoted")

    pid = await sc.promote_cluster(cluster_id, body.display_name.strip(), await get_effective_settings())
    await audit_mod.audit("race.voice.name", "speaker_cluster", cluster_id,
                           after={"profile_id": pid, "name": body.display_name, "race_id": race_id},
                           actor=user)
    return {"profile_id": pid, "display_name": body.display_name.strip()}


@router.delete("/{race_id}")
async def delete_race(race_id: str, delete_clips: bool = False, user=Depends(get_current_user)):
    """Delete a race, and optionally everything filed under it.

    Two genuinely different intentions, so the destructive one has to be asked for by name.
    By default clips.race_id is ON DELETE SET NULL: the recordings and their transcripts
    stay in the library and merely stop belonging to this race.

    With delete_clips, each recording goes through the same path as DELETE /v1/clips/{id} —
    files off disk, row deleted, deletion receipt written — rather than a bulk DELETE that
    would leave the audio orphaned on disk with nothing left pointing at it.
    """
    cfg = get_settings()
    await _get_race(race_id)

    deleted = 0
    if delete_clips:
        rows = await db.fetch("SELECT id, sha256 FROM clips WHERE race_id=$1", race_id)
        for row in rows:
            storage.delete_clip_files(cfg, str(row["id"]))
            await db.execute("DELETE FROM clips WHERE id=$1", row["id"])
            await db.insert("deletion_receipts", {
                "object_type": "clip", "object_id": str(row["id"]),
                "sha256": row["sha256"], "reason": "race_deleted",
            })
        deleted = len(rows)

    await db.execute("DELETE FROM races WHERE id=$1", race_id)
    await audit_mod.audit("race.delete", "race", race_id, actor=user,
                           after={"clips_deleted": deleted})
    return {"ok": True, "clips_deleted": deleted}
