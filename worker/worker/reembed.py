"""Keep every stored voiceprint in the active embedding model's space.

A voiceprint is only comparable with voiceprints from the same model. Switching the
embedder (ECAPA -> ResNet293 was the reason this exists) without touching what is stored
would leave every profile centroid, enrolment, cluster and clip voice in the old space, and
every new clip would be compared against them — a cosine between two unrelated spaces,
reported as if it were a similarity. With different widths it would not even run.

So at worker start the active model is compared with the one stamped in embedding_spaces,
and on any difference the whole speaker store is rebuilt from the audio:

  1. the four vector columns are cleared and retyped to the new width (one transaction)
  2. every COMPLETE clip's voices are re-embedded from its stored audio and its stored
     diarization turns, with stages/embed.py's own compute_speakers — so a migrated
     voiceprint is exactly what processing that clip today would produce
  3. enrolments are rebuilt from the clip voice they came from; ones with no clip behind
     them (live-session tags, whose audio was never kept) cannot be, and are removed
  4. profile and cluster centroids are recomputed from the new vectors
  5. every machine-made attribution is re-identified under the new model

Interrupting it is safe: the stamp is written last, so the next start simply runs it again.
"""
import asyncio
import time

import numpy as np
import structlog

from common import db, storage, speaker as sc
from common.audit import audit
from .stages.embed import compute_speakers
from .stages.preprocess import normalized

log = structlog.get_logger()

SPACE = "speaker"

# (table, column, HNSW index or None). Index names are the ones db/init.sql creates.
VECTOR_COLUMNS = [
    ("clip_speakers", "embedding", "clipspk_emb"),
    ("speaker_enrollments", "embedding", None),
    ("speaker_profiles", "centroid", "profiles_centroid"),
    ("speaker_clusters", "centroid", "clusters_centroid"),
]


def needs_migration(stored: dict | None, column_dim: int | None, model: str, dim: int) -> bool:
    """Pure decision, kept apart so it is testable without a database.

    No stamp means the database predates stamping. Its vectors (if any) were made by
    whatever model was configured then, which this process cannot know — except that a
    width mismatch proves they came from a different model.
    """
    if column_dim is not None and column_dim != dim:
        return True
    if stored is None:
        # right width, no stamp: either empty (fresh install) or vectors from a same-width
        # model this process cannot identify. Stamping them as the active model is the only
        # option that does not throw away a store it has no evidence against.
        return False
    return stored["model"] != model or int(stored["dim"]) != dim


async def _retype(dim: int):
    async with db.pool().acquire() as conn:
        async with conn.transaction():
            for table, column, index in VECTOR_COLUMNS:
                if index:
                    await conn.execute(f"DROP INDEX IF EXISTS {index}")
                if table == "speaker_enrollments":
                    await conn.execute(f"ALTER TABLE {table} ALTER COLUMN {column} DROP NOT NULL")
                await conn.execute(f"UPDATE {table} SET {column} = NULL")
                await conn.execute(f"ALTER TABLE {table} ALTER COLUMN {column} TYPE vector({dim})")
                if index:
                    await conn.execute(
                        f"CREATE INDEX {index} ON {table} USING hnsw ({column} vector_cosine_ops)")


async def _reembed_clip(clip, pool, cfg) -> int:
    turns = [{"idx": r["idx"], "label": r["local_label"], "start": float(r["start_s"]),
              "end": float(r["end_s"]), "is_overlap": bool(r["is_overlap"]),
              # not stored, but defined by length alone: diarize.clean_turns marks exactly
              # the turns shorter than min_turn_s
              "too_short": float(r["end_s"]) - float(r["start_s"]) < cfg.min_turn_s}
             for r in await db.fetch(
                 "SELECT idx, local_label, start_s, end_s, is_overlap FROM speaker_turns "
                 "WHERE clip_id=$1 ORDER BY idx", clip["id"])]
    vad = [(float(r["start_s"]), float(r["end_s"])) for r in await db.fetch(
        "SELECT start_s, end_s FROM vad_regions WHERE clip_id=$1 ORDER BY idx", clip["id"])]
    if not turns:
        return 0
    audio_norm, _ = await asyncio.to_thread(normalized, storage.resolve(cfg, clip["raw_path"]), cfg)
    speakers = await asyncio.to_thread(
        compute_speakers, audio_norm, turns, vad, cfg, pool, lambda *a, **k: None)
    rows = [(clip["id"], label, list(map(float, sp.embedding)) if sp.embedding is not None else None,
             sp.reliability, sp.reliability_reason, sp.split_half_sim, sp.n_segments_used, sp.speech_s)
            for label, sp in speakers.items()]
    await db.pool().executemany(
        """UPDATE clip_speakers SET embedding=$3, reliability=$4, reliability_reason=$5,
               split_half_sim=$6, n_segments_used=$7, speech_s=$8
           WHERE clip_id=$1 AND local_label=$2""", rows)
    return sum(1 for r in rows if r[2] is not None)


async def _rebuild_enrollments() -> dict:
    # traced through the voice they were taken from; a promoted enrolment from before
    # promote_cluster recorded local_label is traced through the profile that clip voice
    # now belongs to, which is unambiguous when that profile holds one voice in the clip
    await db.execute(
        """UPDATE speaker_enrollments e SET embedding = cs.embedding
             FROM clip_speakers cs
            WHERE e.clip_id = cs.clip_id AND e.local_label = cs.local_label
              AND cs.embedding IS NOT NULL""")
    await db.execute(
        """UPDATE speaker_enrollments e SET embedding = cs.embedding, local_label = cs.local_label
             FROM clip_speakers cs
            WHERE e.embedding IS NULL AND e.local_label IS NULL AND e.clip_id = cs.clip_id
              AND cs.profile_id = e.profile_id AND cs.embedding IS NOT NULL
              AND (SELECT count(*) FROM clip_speakers c2
                    WHERE c2.clip_id = e.clip_id AND c2.profile_id = e.profile_id) = 1""")
    lost = await db.fetch(
        "SELECT id, profile_id, clip_id FROM speaker_enrollments WHERE embedding IS NULL")
    await db.execute("DELETE FROM speaker_enrollments WHERE embedding IS NULL")
    await db.execute("ALTER TABLE speaker_enrollments ALTER COLUMN embedding SET NOT NULL")
    return {"rebuilt": await db.fetchval("SELECT count(*) FROM speaker_enrollments"),
            "removed": [{"enrollment_id": str(r["id"]), "profile_id": str(r["profile_id"]),
                         "clip_id": str(r["clip_id"]) if r["clip_id"] else None} for r in lost]}


async def _rebuild_clusters():
    for c in await db.fetch("SELECT id FROM speaker_clusters WHERE promoted_to IS NULL"):
        members = await db.fetch(
            """SELECT embedding, reliability, speech_s FROM clip_speakers
               WHERE cluster_id=$1 AND embedding IS NOT NULL""", c["id"])
        if not members:
            continue
        E = np.stack([np.asarray(m["embedding"], dtype=np.float32) for m in members])
        w = np.array([max(float(m["reliability"] or 0) * float(m["speech_s"] or 0), 1e-6)
                      for m in members], dtype=np.float32)
        await db.execute(
            "UPDATE speaker_clusters SET centroid=$2, weight_sum=$3 WHERE id=$1",
            c["id"], list(map(float, sc.weighted_centroid(E, w))), float(w.sum()))


async def ensure_space(pool, cfg, model: str, dim: int) -> dict | None:
    """Migrate the speaker store to `model` if it is not already in its space. Caller holds
    the model lane (this runs at worker start, before any job)."""
    stored = await db.fetchrow("SELECT model, dim FROM embedding_spaces WHERE name=$1", SPACE)
    col = await sc.embedding_dim()
    if not needs_migration(dict(stored) if stored else None, col, model, dim):
        if stored is None:
            await db.execute(
                "INSERT INTO embedding_spaces (name, model, dim) VALUES ($1,$2,$3)", SPACE, model, dim)
        return None

    t0 = time.perf_counter()
    log.warning("speaker_space_migrating", from_model=stored["model"] if stored else None,
                from_dim=col, to_model=model, to_dim=dim)
    await _retype(dim)

    clips = await db.fetch("SELECT id, raw_path FROM clips WHERE status='COMPLETE' ORDER BY created_at")
    voices, failed = 0, []
    for i, clip in enumerate(clips, 1):
        try:
            voices += await _reembed_clip(clip, pool, cfg)
        except Exception as e:
            # a clip whose audio is gone keeps NULL voiceprints: it simply stops taking part
            # in identification, which is the honest state for a voice nobody can re-measure
            failed.append(str(clip["id"]))
            log.warning("reembed_clip_failed", clip_id=str(clip["id"]), error=str(e)[:200])
        if i % 25 == 0:
            log.info("reembed_progress", done=i, of=len(clips))

    enrol = await _rebuild_enrollments()
    for p in await db.fetch("SELECT id FROM speaker_profiles"):
        await sc.recompute_centroid(str(p["id"]))
    await _rebuild_clusters()
    reid = await sc.reidentify(cfg)

    await db.execute(
        """INSERT INTO embedding_spaces (name, model, dim, updated_at) VALUES ($1,$2,$3, now())
           ON CONFLICT (name) DO UPDATE SET model=$2, dim=$3, updated_at=now()""", SPACE, model, dim)
    summary = {"from_model": stored["model"] if stored else None, "from_dim": col,
               "to_model": model, "to_dim": dim, "clips": len(clips), "voices": voices,
               "failed_clips": failed, "enrollments": enrol,
               "reidentified": {"checked": reid["checked"], "changed": len(reid["changed"])},
               "seconds": round(time.perf_counter() - t0, 1)}
    await audit("speaker.reembed", "embedding_space", SPACE, actor="system:worker", after=summary)
    log.warning("speaker_space_migrated", **{k: v for k, v in summary.items() if k != "enrollments"})
    return summary
