from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from common import db, audit as audit_mod, speaker as sc, graph, graph_sync, tagging
from common.config import get_effective_settings
from ..auth import get_current_user
from ..services import queue, speaker_ops

router = APIRouter(prefix="/v1/speakers", tags=["speakers"])


PROFILE_STATUSES = ("provisional", "confirmed", "locked")


@router.get("")
async def list_speakers(q: str | None = None, user=Depends(get_current_user)):
    where = "WHERE p.display_name ILIKE $1 OR $1 ILIKE ANY(p.tags)" if q else ""
    args = [f"%{q}%"] if q else []
    # Activity rolled up per profile, so the directory can sort and show presence without
    # a request per row.
    rows = await db.fetch(
        f"""SELECT p.id, p.display_name, p.status, p.n_enrollments, p.total_speech_s,
                   p.mean_reliability, p.intra_cohesion, p.created_at, p.updated_at,
                   p.notes, p.tags,
                   coalesce(a.n_clips, 0) AS n_clips, a.last_seen, a.heard_s
              FROM speaker_profiles p
              LEFT JOIN (
                SELECT cs.profile_id, count(DISTINCT cs.clip_id)::int AS n_clips,
                       max(c.created_at) AS last_seen, sum(cs.speech_s) AS heard_s
                  FROM clip_speakers cs JOIN clips c ON c.id = cs.clip_id
                 WHERE c.status = 'COMPLETE' AND cs.profile_id IS NOT NULL
                 GROUP BY cs.profile_id
              ) a ON a.profile_id = p.id
            {where} ORDER BY p.display_name""", *args)
    return {"items": [dict(r) for r in rows]}


class CreateProfile(BaseModel):
    display_name: str
    notes: str | None = None
    tags: list[str] | None = None


def _clean_tags(tags: list[str] | None) -> list[str] | None:
    return None if tags is None else list(dict.fromkeys(t.strip() for t in tags if t.strip()))


async def _name_taken(name: str, except_id: str | None = None):
    row = await db.fetchrow(
        "SELECT id, display_name FROM speaker_profiles WHERE lower(display_name)=lower($1)", name)
    if row and str(row["id"]) != (except_id or ""):
        # 409 with the existing id, so the UI can offer a merge instead of a dead end.
        raise HTTPException(409, {"message": f"a speaker named {row['display_name']!r} "
                                              f"already exists", "existing_id": str(row["id"])})


@router.post("")
async def create_profile(body: CreateProfile, user=Depends(get_current_user)):
    name = body.display_name.strip()
    if not name:
        raise HTTPException(400, "display_name is required")
    await _name_taken(name)
    pid = await db.insert("speaker_profiles", {"display_name": name, "notes": body.notes,
                                                "tags": _clean_tags(body.tags) or [],
                                                "status": "provisional"})
    await audit_mod.audit("speaker.create", "speaker_profile", pid, after=body.model_dump(), actor=user)
    await graph_sync.resync(queue.redis())
    return {"id": pid}


@router.get("/merge-suggestions")
async def merge_suggestions(limit: int = 20, user=Depends(get_current_user)):
    """Pairs of profiles whose voiceprints are close enough that they are probably one
    person enrolled twice — the usual result of naming the same voice differently in two
    clips. Declared before /{profile_id} so the path is not read as an id."""
    cfg = await get_effective_settings()
    rows = await db.fetch(
        """SELECT a.id AS a_id, a.display_name AS a_name, a.n_enrollments AS a_n,
                  b.id AS b_id, b.display_name AS b_name, b.n_enrollments AS b_n,
                  1 - (a.centroid <=> b.centroid) AS sim
             FROM speaker_profiles a
             JOIN speaker_profiles b ON a.id < b.id
            WHERE a.centroid IS NOT NULL AND b.centroid IS NOT NULL
              AND 1 - (a.centroid <=> b.centroid) >= $1
            ORDER BY sim DESC LIMIT $2""",
        cfg.id_threshold - cfg.id_suggest_delta, min(max(limit, 1), 100))
    return {"items": [{**dict(r), "a_id": str(r["a_id"]), "b_id": str(r["b_id"]),
                       "band": "likely" if r["sim"] >= cfg.id_threshold else "possible"}
                      for r in rows],
            "id_threshold": cfg.id_threshold}


@router.get("/{profile_id}")
async def get_profile(profile_id: str, user=Depends(get_current_user)):
    row = await db.fetchrow("SELECT * FROM speaker_profiles WHERE id=$1", profile_id)
    if not row:
        raise HTTPException(404, "profile not found")
    enrollments = await db.fetch(
        """SELECT id, clip_id, duration_s, reliability, quality, source, is_outlier,
                  cos_to_centroid, created_at FROM speaker_enrollments
           WHERE profile_id=$1 ORDER BY created_at DESC""", profile_id)
    d = dict(row)
    d.pop("centroid", None)
    return {**d, "enrollments": [dict(e) for e in enrollments]}


class PatchProfile(BaseModel):
    display_name: str | None = None
    notes: str | None = None
    status: str | None = None
    tags: list[str] | None = None


@router.patch("/{profile_id}")
async def patch_profile(profile_id: str, body: PatchProfile, user=Depends(get_current_user)):
    if not await db.fetchval("SELECT 1 FROM speaker_profiles WHERE id=$1", profile_id):
        raise HTTPException(404, "profile not found")
    if body.display_name is not None:
        body.display_name = body.display_name.strip()
        if not body.display_name:
            raise HTTPException(400, "display_name cannot be empty")
        await _name_taken(body.display_name, profile_id)
    if body.status is not None and body.status not in PROFILE_STATUSES:
        raise HTTPException(400, f"status must be one of {', '.join(PROFILE_STATUSES)}")
    body.tags = _clean_tags(body.tags)
    sets, params = [], []
    for field, val in body.model_dump(exclude_none=True).items():
        params.append(val)
        sets.append(f"{field}=${len(params)}")
    if sets:
        params.append(profile_id)
        await db.execute(f"UPDATE speaker_profiles SET {', '.join(sets)}, updated_at=now() WHERE id=${len(params)}", *params)
        await audit_mod.audit("speaker.rename", "speaker_profile", profile_id,
                               after=body.model_dump(exclude_none=True), actor=user)
        if body.display_name is not None:
            await _refresh_tags(await tagging.clips_for_profile(profile_id))
        await graph_sync.resync(queue.redis())
    return {"ok": True}


async def _refresh_tags(clip_ids: list[str]):
    """Keep `speaker:` auto tags in step with an identity edit — they name the profile, so
    a rename, merge or delete makes the old ones false on every clip that voice is in."""
    await tagging.refresh_signal_tags(clip_ids, await get_effective_settings())


@router.delete("/{profile_id}")
async def delete_profile(profile_id: str, reassign_to: str | None = None, user=Depends(get_current_user)):
    if not await db.fetchval("SELECT 1 FROM speaker_profiles WHERE id=$1", profile_id):
        raise HTTPException(404, "profile not found")
    affected = await tagging.clips_for_profile(profile_id)  # before the links are cleared
    if reassign_to:
        await sc.merge_profiles(profile_id, reassign_to)
    else:
        await db.execute("UPDATE clip_speakers SET profile_id=NULL WHERE profile_id=$1", profile_id)
        await db.execute("UPDATE utterances SET profile_id=NULL WHERE profile_id=$1", profile_id)
        # Same FK as in merge_profiles; the cluster goes back to being an unnamed voice.
        await db.execute("UPDATE speaker_clusters SET promoted_to=NULL WHERE promoted_to=$1", profile_id)
        await db.execute("DELETE FROM speaker_profiles WHERE id=$1", profile_id)
    await audit_mod.audit("speaker.delete", "speaker_profile", profile_id, actor=user)
    await _refresh_tags(affected)
    # voices only this profile was claiming may now fall to another profile, or to unknown
    await queue.enqueue_reidentify()
    await graph_sync.resync(queue.redis())
    return {"ok": True}


@router.get("/{profile_id}/clips")
async def profile_clips(profile_id: str, user=Depends(get_current_user)):
    rows = await db.fetch(
        """SELECT c.id, c.filename, c.duration_s, c.created_at, cs.talk_share, cs.match_result
           FROM clip_speakers cs JOIN clips c ON c.id = cs.clip_id
           WHERE cs.profile_id=$1 ORDER BY c.created_at DESC""", profile_id)
    return {"items": [dict(r) for r in rows]}


class MergeBody(BaseModel):
    source_id: str | None = None
    # Several at once: "these four are all the same person" is one action, not four.
    source_ids: list[str] | None = None


@router.post("/{profile_id}/merge")
async def merge(profile_id: str, body: MergeBody, user=Depends(get_current_user)):
    sources = list(dict.fromkeys([*(body.source_ids or []), *([body.source_id] if body.source_id else [])]))
    sources = [s for s in sources if s != profile_id]
    if not sources:
        raise HTTPException(400, "name at least one other profile to merge in")
    if not await db.fetchval("SELECT 1 FROM speaker_profiles WHERE id=$1", profile_id):
        raise HTTPException(404, "profile not found")
    # Kept profile's own notes and tags survive; the merged ones' are folded in, so a merge
    # never silently drops something a person wrote.
    for source in sources:
        src = await db.fetchrow("SELECT display_name, notes, tags FROM speaker_profiles WHERE id=$1",
                                source)
        if not src:
            continue
        await db.execute(
            """UPDATE speaker_profiles
                  SET tags = ARRAY(SELECT DISTINCT unnest(tags || $2::text[])),
                      notes = CASE WHEN coalesce($3, '') = '' THEN notes
                                   WHEN coalesce(notes, '') = '' THEN $3
                                   ELSE notes || E'\n\n' || $3 END
                WHERE id=$1""",
            profile_id, list(src["tags"] or []),
            f"[from {src['display_name']}] {src['notes']}" if src["notes"] else None)
        await sc.merge_profiles(source, profile_id)
    await audit_mod.audit("speaker.merge", "speaker_profile", profile_id,
                           after={"merged_from": sources}, actor=user)
    # after the merge every clip of either voice is filed under profile_id
    await _refresh_tags(await tagging.clips_for_profile(profile_id))
    # the merged centroid is a different point from either original
    await queue.enqueue_reidentify()
    await graph_sync.resync(queue.redis())
    return {"ok": True}


@router.delete("/{profile_id}/enrollments/{eid}")
async def delete_enrollment(profile_id: str, eid: str, user=Depends(get_current_user)):
    await db.execute("DELETE FROM speaker_enrollments WHERE id=$1 AND profile_id=$2", eid, profile_id)
    await sc.recompute_centroid(profile_id)
    await audit_mod.audit("enrollment.remove", "speaker_profile", profile_id, after={"enrollment_id": eid}, actor=user)
    await queue.enqueue_reidentify()
    return {"ok": True}


@router.get("/{profile_id}/candidates")
async def profile_candidates(profile_id: str, limit: int = 30, user=Depends(get_current_user)):
    """Voices in other clips that sound like this person but are not attributed to them,
    best first — for confirming one voice everywhere it occurs in a single pass."""
    if not await db.fetchval("SELECT 1 FROM speaker_profiles WHERE id=$1", profile_id):
        raise HTTPException(404, "profile not found")
    cfg = await get_effective_settings()
    return {"items": await speaker_ops.candidates(profile_id, cfg, min(max(limit, 1), 100)),
            "id_threshold": cfg.id_threshold,
            "floor": cfg.id_threshold - cfg.id_suggest_delta}


class ClaimItem(BaseModel):
    clip_id: str
    local_label: str


class ClaimBody(BaseModel):
    items: list[ClaimItem]
    enroll: bool = True


@router.post("/{profile_id}/claim")
async def claim(profile_id: str, body: ClaimBody, user=Depends(get_current_user)):
    """Confirm a batch of candidate voices as this person — each one exactly as if it had
    been assigned from its own clip page, then one re-identification pass for the lot."""
    if not await db.fetchval("SELECT 1 FROM speaker_profiles WHERE id=$1", profile_id):
        raise HTTPException(404, "profile not found")
    cfg = await get_effective_settings()
    results = []
    for it in body.items:
        r = await speaker_ops.assign(it.clip_id, it.local_label, profile_id,
                                     enroll=body.enroll, cfg=cfg, actor=user)
        results.append({"clip_id": it.clip_id, "local_label": it.local_label, **r})
    await queue.enqueue_reidentify()
    await graph_sync.resync(queue.redis())
    return {"items": results,
            "enrolled": sum(1 for r in results if r.get("enrolled")),
            "assigned": sum(1 for r in results if r.get("ok"))}


@router.get("/{profile_id}/relations")
async def relations(profile_id: str, user=Depends(get_current_user)):
    """Who this person shares recordings with, what those recordings are about, and what
    they name when they talk. Co-speakers and topics come from Postgres directly; named
    entities exist only in the graph and are left empty when it is off or unsynced."""
    if not await db.fetchval("SELECT 1 FROM speaker_profiles WHERE id=$1", profile_id):
        raise HTTPException(404, "profile not found")
    co = await db.fetch(
        """SELECT p.id, p.display_name, count(DISTINCT a.clip_id)::int AS n_clips
             FROM clip_speakers a
             JOIN clip_speakers b ON b.clip_id = a.clip_id AND b.profile_id <> a.profile_id
             JOIN speaker_profiles p ON p.id = b.profile_id
             JOIN clips c ON c.id = a.clip_id
            WHERE a.profile_id = $1 AND c.status = 'COMPLETE'
            GROUP BY p.id, p.display_name ORDER BY n_clips DESC LIMIT 20""", profile_id)
    topics = await db.fetch(
        """SELECT substr(t, 7) AS topic, count(DISTINCT c.id)::int AS n_clips
             FROM clip_speakers cs
             JOIN clips c ON c.id = cs.clip_id
             CROSS JOIN LATERAL unnest(c.auto_tags) t
            WHERE cs.profile_id = $1 AND c.status = 'COMPLETE' AND t LIKE 'topic:%'
            GROUP BY 1 ORDER BY n_clips DESC LIMIT 20""", profile_id)
    unknown_voices = await db.fetchval(
        """SELECT count(*) FROM clip_speakers b
             JOIN clip_speakers a ON a.clip_id = b.clip_id
            WHERE a.profile_id = $1 AND b.profile_id IS NULL""", profile_id)
    entities = []
    try:
        entities = await graph.run(
            """MATCH (s:Speaker {profile_id: $pid})-[r:MENTIONED]->(e:Entity)
               RETURN e.name AS name, e.kind AS kind, r.n AS n ORDER BY n DESC LIMIT 20""",
            pid=profile_id)
    except Exception:
        pass
    return {"co_speakers": [{**dict(r), "id": str(r["id"])} for r in co],
            "topics": [dict(r) for r in topics],
            "entities": entities,
            "unidentified_co_voices": unknown_voices}


@router.get("/{profile_id}/utterances")
async def profile_utterances(profile_id: str, limit: int = 50, offset: int = 0,
                             user=Depends(get_current_user)):
    """What this person said, newest recording first."""
    rows = await db.fetch(
        """SELECT u.id, u.clip_id, u.start_s, u.end_s, u.text, c.filename, c.created_at
             FROM utterances u
             JOIN clips c ON c.id = u.clip_id
             LEFT JOIN clip_speakers cs ON cs.clip_id = u.clip_id AND cs.local_label = u.local_label
            WHERE coalesce(u.profile_id, cs.profile_id) = $1 AND c.status = 'COMPLETE'
            ORDER BY c.created_at DESC, u.idx
            LIMIT $2 OFFSET $3""", profile_id, min(max(limit, 1), 200), max(offset, 0))
    return {"items": [{**dict(r), "id": str(r["id"]), "clip_id": str(r["clip_id"])} for r in rows]}
