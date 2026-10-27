"""Attributing a clip voice to a profile — one implementation for every place a person does it.

The clip page's assign dialog and the Speakers page's "possible matches" confirm both end
in the same writes: the attribution, the transcript's speaker, an optional enrolment, the
audit entry, and then the follow-on passes (tags, re-identification, graph). Two copies of
that sequence would drift the way speaker.py's docstring warns about.
"""
from common import db, audit as audit_mod, speaker as sc, tagging


async def assign(clip_id: str, local_label: str, profile_id: str | None, *, enroll: bool,
                 cfg, actor: str) -> dict:
    cs = await db.fetchrow("SELECT * FROM clip_speakers WHERE clip_id=$1 AND local_label=$2",
                           clip_id, local_label)
    if not cs:
        return {"error": "speaker not found on this clip"}
    before = {"profile_id": str(cs["profile_id"]) if cs["profile_id"] else None,
              "match_result": cs["match_result"], "score": cs["match_score"]}

    # is_manual is what keeps re-identification from ever overriding a person's decision
    await db.execute(
        """UPDATE clip_speakers SET profile_id=$1, is_manual=true, match_result='confident',
               cluster_id = CASE WHEN $1::uuid IS NULL THEN cluster_id ELSE NULL END
           WHERE clip_id=$2 AND local_label=$3""", profile_id, clip_id, local_label)
    await db.execute("UPDATE utterances SET profile_id=$1 WHERE clip_id=$2 AND local_label=$3",
                     profile_id, clip_id, local_label)

    enrolled, skip_reason = False, None
    if enroll and profile_id:
        if cs["embedding"] is None:
            skip_reason = "not enough clean (non-overlapping) speech from this speaker to build a voiceprint"
        elif (cs["reliability"] or 0) < cfg.reliability_fair:
            skip_reason = f"voiceprint reliability too low ({cs['reliability'] or 0:.2f})"
        else:
            await sc.add_enrollment(profile_id, cs, clip_id, source="manual", actor=actor)
            enrolled = True

    await audit_mod.audit("identification.correct", "clip_speaker", f"{clip_id}:{local_label}",
                          before=before, after={"profile_id": str(profile_id) if profile_id else None},
                          actor=actor)
    await tagging.refresh_signal_tags([clip_id], cfg)
    return {"ok": True, "profile_id": profile_id, "enrolled": enrolled,
            "enroll_skip_reason": skip_reason, "was": before}


async def candidates(profile_id: str, cfg, limit: int = 30) -> list[dict]:
    """Voices elsewhere in the corpus that sound like this profile and are not attributed
    to it — what "name this voice everywhere" offers for confirmation.

    The floor is the bottom of the suggestion band (id_threshold - id_suggest_delta), the
    same line under which identify() calls a voice unknown: anything shown here would at
    least have been a suggestion. Voices a person attributed by hand are not offered; that
    decision was already made.
    """
    centroid = await db.fetchval("SELECT centroid FROM speaker_profiles WHERE id=$1", profile_id)
    if centroid is None:
        return []
    floor = cfg.id_threshold - cfg.id_suggest_delta
    rows = await db.fetch(
        """SELECT cs.clip_id, cs.local_label, cs.speech_s, cs.reliability, cs.match_result,
                  cs.profile_id, p.display_name AS current_name, c.filename,
                  1 - (cs.embedding <=> $1::vector) AS sim,
                  (SELECT min(t.start_s) FROM speaker_turns t
                    WHERE t.clip_id = cs.clip_id AND t.local_label = cs.local_label) AS first_s
             FROM clip_speakers cs
             JOIN clips c ON c.id = cs.clip_id
             LEFT JOIN speaker_profiles p ON p.id = cs.profile_id
            WHERE cs.embedding IS NOT NULL AND NOT cs.is_manual
              AND cs.profile_id IS DISTINCT FROM $2::uuid
              AND 1 - (cs.embedding <=> $1::vector) >= $3
            ORDER BY cs.embedding <=> $1::vector
            LIMIT $4""", centroid, profile_id, floor, limit)
    return [{**dict(r), "clip_id": str(r["clip_id"]),
             "profile_id": str(r["profile_id"]) if r["profile_id"] else None,
             "band": "likely" if r["sim"] >= cfg.id_threshold else "possible"} for r in rows]
