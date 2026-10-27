"""Automatic clip tags — the half that needs no model.

Tags are namespaced `family:value` so the UI can group and colour them, and so they can
never collide with an operator's free-text tags, which live in a different column
(clips.tags) and are never touched here.

Signal tags are a direct read of stored measurements: language, voice count, named
speakers, audio quality, length, cross-talk and ASR confidence. Topic tags need the text encoder and are
computed in the worker (worker/worker/stages/tag.py); everything here is shared so the api
can refresh signal tags the moment a human corrects a speaker, without a reprocess.

Same posture as identification: a tag is only written when the signal behind it is clear.
`speaker:` is confident matches only. A missing tag
means "not clearly so", never "clearly not".
"""
from . import db

LOW_ASR_CONF = 0.55
SHORT_CLIP_S = 30.0
LONG_CLIP_S = 600.0

TOPIC_PREFIX = "topic:"


def _dur(u: dict) -> float:
    return max(float(u["end_s"]) - float(u["start_s"]), 0.0)


def derive_tags(clip: dict, quality: dict | None, speakers: list[dict],
                utterances: list[dict], cfg) -> list[str]:
    """Signal tags from stored measurements. Pure, so it is testable without a database."""
    tags: list[str] = []

    if clip.get("language") and (clip.get("language_conf") or 0) >= 0.5:
        tags.append(f"lang:{clip['language']}")

    n = clip.get("n_speakers") or len(speakers)
    if n == 1:
        tags.append("voices:solo")
    elif n == 2:
        tags.append("voices:dialogue")
    elif n and n >= 3:
        tags.append("voices:group")

    # Confident only. A suggested match is a question for a curator, not a fact to file
    # a recording under. Manual assignments are recorded as confident, so they count.
    names = sorted({s["display_name"] for s in speakers
                    if s.get("match_result") == "confident" and s.get("display_name")})
    tags += [f"speaker:{name}" for name in names]

    grade = (quality or {}).get("grade")
    if grade == "poor":
        tags.append("audio:poor")
    elif grade == "good":
        tags.append("audio:clean")

    duration = clip.get("duration_s") or 0
    if duration and duration < SHORT_CLIP_S:
        tags.append("length:short")
    elif duration >= LONG_CLIP_S:
        tags.append("length:long")

    speech = sum(_dur(u) for u in utterances)
    if speech > 0:
        overlap = sum(_dur(u) * (u.get("overlap_ratio") or 0) for u in utterances)
        if overlap / speech > cfg.overlap_warn_ratio:
            tags.append("crosstalk")
        confs = [(u["mean_word_conf"], _dur(u)) for u in utterances
                 if u.get("mean_word_conf") is not None]
        weight = sum(d for _, d in confs)
        if weight > 0 and sum(c * d for c, d in confs) / weight < LOW_ASR_CONF:
            # the hallucination-risk flag: low-confidence ASR is where invented text lives
            tags.append("asr:low-confidence")

    return tags


async def load_inputs(clip_id: str, with_embeddings: bool = False):
    """Everything derive_tags (and the worker's topic pass) reads, from Postgres.
    None when the clip no longer exists."""
    clip = await db.fetchrow(
        """SELECT language, language_conf, n_speakers, duration_s, auto_tags
           FROM clips WHERE id=$1""", clip_id)
    if clip is None:
        return None
    quality = await db.fetchrow("SELECT grade FROM quality_metrics WHERE clip_id=$1", clip_id)
    speakers = await db.fetch(
        """SELECT cs.match_result, sp.display_name FROM clip_speakers cs
           LEFT JOIN speaker_profiles sp ON sp.id = cs.profile_id WHERE cs.clip_id=$1""",
        clip_id)
    utterances = await db.fetch(
        f"""SELECT start_s, end_s, text, mean_word_conf,
                   overlap_ratio{", embedding" if with_embeddings else ""}
            FROM utterances WHERE clip_id=$1 ORDER BY idx""", clip_id)
    return (dict(clip), dict(quality) if quality else None,
            [dict(s) for s in speakers], [dict(u) for u in utterances])


async def refresh_signal_tags(clip_ids, cfg) -> int:
    """Recompute the signal tags of already-tagged clips, keeping their topic tags.

    For the api's correction paths (speaker assign, rename, merge, delete): those change
    which `speaker:` tag is true, and nothing else re-derives tags until a reprocess. Topics
    depend on the transcript alone, which a correction does not change, so they are carried
    over rather than recomputed — which is also what lets this run without the encoder.
    Clips never tagged (auto_tags NULL) are left for the worker's backfill.
    """
    n = 0
    for clip_id in {str(c) for c in clip_ids}:
        inputs = await load_inputs(clip_id)
        if inputs is None or inputs[0].get("auto_tags") is None:
            continue
        clip, quality, speakers, utterances = inputs
        topics = [t for t in clip["auto_tags"] if t.startswith(TOPIC_PREFIX)]
        tags = derive_tags(clip, quality, speakers, utterances, cfg) + topics
        await db.execute("UPDATE clips SET auto_tags=$2, tagged_at=now() WHERE id=$1", clip_id, tags)
        n += 1
    return n


async def clips_for_profile(profile_id: str) -> list[str]:
    rows = await db.fetch(
        "SELECT DISTINCT clip_id FROM clip_speakers WHERE profile_id=$1", profile_id)
    return [str(r["clip_id"]) for r in rows]
