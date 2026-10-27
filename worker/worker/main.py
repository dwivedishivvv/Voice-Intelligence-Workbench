import asyncio
import json
import os
import time

import structlog
from arq import cron, func
from arq.connections import RedisSettings

import numpy as np

from common.config import get_settings, get_effective_settings
from common import db, storage, speaker as sc, schema, graph_sync
from common.audit import audit
from . import maintenance, reembed
from .ctx import SR
from .speaker_models import embedding_dim
from .lane import ModelLane, INTERACTIVE
from .audio.decode import decode_to_array
from .audio.enhance import highpass, loudness_normalize
from .audio.vad import detect_vad
from .pool import ModelPool
from .pipeline import process_clip, _sync_graph
from common.events import init_events, emit_live
from .stages.transcribe import compression_ratio, BOILERPLATE
from .stages import embed as embed_mod, tag as tag_mod
from .audio.quality import rolloff_99

log = structlog.get_logger()

# how often (in chunks) a locked live session re-opens language detection instead of
# forcing the pinned language — lets genuine code-switching catch up within ~40s
# (6 chunks * 7s) instead of being stuck on whatever chunk 0 happened to detect
LIVE_LANG_RECHECK_EVERY = 6


async def startup(ctx):
    cfg = get_settings()
    await db.init_pool(cfg)
    # The api replays these too, but the worker may come up first, and its very next
    # status write can name a stage (TAGGING) the enum on an older volume does not have yet.
    await schema.apply_replayable()
    # Startup is the *only* moment the restart-required settings can take effect, so read
    # the DB overrides here rather than the env-only defaults. Without this, both the
    # Settings page's device toggle and the Models tab's Activate button write a
    # settings_overrides row that nothing ever reads — they'd appear to work and change
    # nothing. (Connection settings above are deliberately still env-only: the pool has to
    # exist before there's a DB to read overrides from.)
    cfg = await get_effective_settings()
    ctx["cfg"] = cfg
    await init_events(cfg.redis_url)
    ctx["pool"] = ModelPool(cfg)
    load_ms = await ctx["pool"].load()
    # Before any job: if the stored voiceprints came from a different embedding model they
    # are not comparable with anything this worker will produce, so rebuild them first.
    await reembed.ensure_space(ctx["pool"], cfg, cfg.embed_model,
                               embedding_dim(ctx["pool"].embedder))
    # ONE lane through the model pool, shared by every job type and every device.
    #
    # Every job shares the single ModelPool, and the models in it are not thread-safe:
    # silero's JIT module, the CTranslate2 Whisper model (built with num_workers=1), and
    # pyannote's SpeakerDiarization pipeline, which carries per-call internal state. Two
    # jobs calling the same module from two threadpool threads corrupts the process heap
    # and takes it down with a bare 0xC0000374 — no traceback, clips stuck mid-status.
    #
    # This was previously two independent semaphores, both sized to worker_concurrency on
    # GPU. That was wrong twice over: the corruption is not CPU-specific (it was first
    # caught on CUDA, mid-batch, at worker_concurrency=2), and two separate semaphores let
    # a live chunk and a batch clip overlap inside the same models even when each semaphore
    # was individually satisfied.
    #
    # Serializing costs little — on CPU the models already saturate every core, and on GPU
    # the device is the bottleneck, so concurrent jobs were taking turns regardless. To
    # process clips genuinely in parallel, run N worker processes (N ModelPools), not N
    # threads sharing one pool.
    #
    # Still one lane, but no longer first-come-first-served: see lane.py. A live chunk or a
    # voice query waiting for it goes ahead of the next batch pipeline stage.
    ctx["gpu_sem"] = ModelLane()
    # per-session language lock + periodic recheck state (see LIVE_LANG_RECHECK_EVERY) —
    # a fresh chunk re-guessing language from scratch every time is unstable on that little
    # audio, and detecting it costs a second encoder pass (~250ms of a ~350ms chunk)
    ctx["live_lang"] = {}
    log.info("worker_ready", device=ctx["pool"].device, models=ctx["pool"].versions, load_ms=load_ms)


async def shutdown(ctx):
    await db.close_pool()


async def process_clip_job(ctx, clip_id: str):
    try:
        # fresh per job (not the startup-cached ctx["cfg"]) so Settings-page edits to
        # TUNABLE_FIELDS apply to the next job without a worker restart
        cfg = await get_effective_settings()
        # the lane is taken per stage inside process_clip, not for the whole pipeline, so a
        # live chunk can be served between stages (lane.py)
        await process_clip(clip_id, ctx["pool"], cfg, lane=ctx["gpu_sem"])
    except Exception:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    # Outside the model lane: scheduling is a Redis write, and the rebuild itself runs as
    # its own job that touches no model.
    if cfg.graph_enabled:
        try:
            await graph_sync.schedule(ctx["redis"], cfg.graph_sync_debounce_s)
        except Exception as e:
            log.warning("graph_sync_schedule_failed", error=str(e)[:200])


GRAPH_SYNC_LOCK = "graph-sync:lock"


async def graph_sync_job(ctx):
    """One coalesced graph rebuild (see graph_sync.schedule).

    Locked, because a rebuild is a wipe-then-reproject and two interleaved ones can each
    wipe what the other just wrote. If a rebuild is already running — a slow one outlasting
    its window, or another worker process's — this request is pushed to the next window
    rather than dropped, so the change that triggered it still lands.
    """
    cfg = await get_effective_settings()
    redis = ctx["redis"]
    if not await redis.set(GRAPH_SYNC_LOCK, "1", nx=True, ex=900):
        await graph_sync.schedule(redis, cfg.graph_sync_debounce_s)
        return
    try:
        await _sync_graph(cfg)
    finally:
        await redis.delete(GRAPH_SYNC_LOCK)


async def retag_clips_job(ctx, clip_ids: list[str] | None = None):
    """Re-derive auto tags without reprocessing audio. None means every COMPLETE clip.

    Taken through the model lane in small batches rather than all at once, so a corpus-wide
    retag never holds up a newly uploaded clip for more than one batch."""
    cfg = await get_effective_settings()
    if clip_ids is None:
        clip_ids = [str(r["id"]) for r in await db.fetch(
            "SELECT id FROM clips WHERE status='COMPLETE' ORDER BY created_at DESC")]
    done = 0
    for i in range(0, len(clip_ids), maintenance.BACKFILL_BATCH):
        async with ctx["gpu_sem"]:
            for cid in clip_ids[i:i + maintenance.BACKFILL_BATCH]:
                try:
                    if await tag_mod.tag_clip(cid, ctx["pool"], cfg) is not None:
                        done += 1
                except Exception as e:
                    log.warning("retag_failed", clip_id=cid, error=str(e)[:200])
    return {"tagged": done, "requested": len(clip_ids)}


async def reidentify_job(ctx, clip_ids: list[str] | None = None):
    """Re-run identification over stored voices after anything that changes what a match
    could be — see common/speaker.py reidentify(). Database-only, so it does not take the
    model lane. Clips whose attributions moved get their speaker: tags refreshed and the
    graph rebuild scheduled, so every surface agrees with the new names."""
    cfg = await get_effective_settings()
    result = await sc.reidentify(cfg, clip_ids)
    changed = result["changed"]
    if changed:
        from common import tagging
        await tagging.refresh_signal_tags([c["clip_id"] for c in changed], cfg)
        await audit("speaker.reidentify", "clip_speaker", "batch", actor="system:worker",
                    after={"checked": result["checked"], "changed": changed[:200]})
        if cfg.graph_enabled:
            await graph_sync.schedule(ctx["redis"], cfg.graph_sync_debounce_s)
    log.info("reidentify_done", checked=result["checked"], changed=len(changed))
    return {"checked": result["checked"], "changed": len(changed)}


# A spoken query is a question, not a recording: past this it is trimmed rather than
# transcribed whole, which keeps one voice search from holding the model lane for long.
VOICE_QUERY_MAX_S = 30.0


def _voice_query(pool, cfg, path: str) -> dict:
    audio = decode_to_array(path, SR)[: int(VOICE_QUERY_MAX_S * SR)]
    audio = _precondition(audio, cfg)
    vad = detect_vad(audio, SR, pool.vad, cfg)
    speech = (np.concatenate([audio[int(s * SR):int(e * SR)] for s, e in vad])
              if vad else np.zeros(0, dtype=np.float32))

    text, language, lang_p = "", None, 0.0
    if speech.size >= int(0.3 * SR):
        try:
            segments, info = pool.asr.transcribe(
                audio, beam_size=cfg.asr_beam_size, word_timestamps=False,
                language=None if cfg.asr_language == "auto" else cfg.asr_language,
                condition_on_previous_text=False, temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
                no_speech_threshold=0.6, vad_filter=True)
            text = " ".join(s.text.strip() for s in segments
                            if not _is_hallucination(s.text, s.avg_logprob, s.no_speech_prob)).strip()
            language, lang_p = info.language, float(info.language_probability)
        except ValueError as e:
            if "empty sequence" not in str(e):
                raise

    # The voiceprint is taken from speech only (VAD regions), the same signal the pipeline
    # fingerprints, with the same reliability measure — so a voice search is judged by the
    # rules identification uses, including abstaining on too little audio.
    embedding, reliability = None, 0.0
    speech_s = speech.size / SR
    if speech_s >= LIVE_EMBED_MIN_S:
        embedding = embed_mod.encode_many(pool, [speech])[0]
        noise = _live_noise_power(audio, [{"start": s, "end": e} for s, e in vad])
        reliability = sc.reliability_score(min(speech_s, cfg.embed_target_s), {
            "snr_db": embed_mod._snr_db(speech, noise),
            "clipping_ratio": float(np.mean(np.abs(speech) >= cfg.clipping_threshold)),
            "bandwidth_hz": rolloff_99(speech, SR),
        }, cfg)
    return {"text": text, "language": language, "language_probability": lang_p,
            "duration_s": round(audio.size / SR, 2), "speech_s": round(speech_s, 2),
            "embedding": embedding.tolist() if embedding is not None else None,
            "reliability": reliability}


async def voice_query_job(ctx, rel_path: str):
    """What a spoken (or uploaded) search query says, and whose voice it is.

    Returned to the api, which waits on this job and runs the actual searches — the api
    holds no models. Through the model lane like everything else that touches the pool."""
    cfg = await get_effective_settings()
    path = storage.resolve(cfg, rel_path)
    try:
        # interactive on both lanes: someone is holding a microphone waiting for this
        async with ctx["gpu_sem"].hold(INTERACTIVE), ctx["pool"].asr_lane.hold(INTERACTIVE):
            return await asyncio.to_thread(_voice_query, ctx["pool"], cfg, path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


async def maintenance_job(ctx, manual: bool = False):
    cfg = await get_effective_settings()
    if not cfg.auto_maintenance and not manual:
        return {"skipped": "auto_maintenance is off"}
    summary = await maintenance.run(ctx, cfg)
    log.info("maintenance_done", **{k: (len(v) if isinstance(v, list) else v)
                                    for k, v in summary.items()})
    return summary


def _precondition(audio: np.ndarray, cfg) -> np.ndarray:
    # matches the batch pipeline's cheap (non-model) preprocessing step-for-step, minus
    # denoise (noisereduce) — that one alone was ~260ms/s of audio in the batch run, too
    # much of the chunk budget for what raw mic input needs to become intelligible
    audio = audio - float(np.mean(audio))
    audio = highpass(audio, SR, cfg.highpass_hz)
    audio, _ = loudness_normalize(audio, SR, cfg.target_lufs)
    return audio


# Lower than cfg.embed_min_s (batch default 1.5s): live chunks are naturally shorter than
# a whole clip's pooled best segments (see stages/embed.py's note on this), and
# reliability_score already discounts a short embedding on its own — this only gates
# whether an attempt is made at all, not how much weight it gets once identify() sees it.
LIVE_EMBED_MIN_S = 1.0


def _live_noise_power(audio: np.ndarray, turns: list[dict]) -> float:
    """Power of everything in this chunk that diarization did not put inside a turn — the
    floor each turn's SNR is measured against. A chunk with no gaps falls back to a fixed
    estimate rather than reporting an absurd SNR off an empty slice."""
    if audio is None or not audio.size:
        return 1e-12
    mask = np.zeros(audio.size, dtype=bool)
    for t in turns:
        mask[int(t["start"] * SR):int(t["end"] * SR)] = True
    ns = audio[~mask]
    return float(np.mean(ns ** 2)) if ns.size > SR * 0.1 else 1e-12


def _is_hallucination(text: str, avg_logprob: float, no_speech_prob: float) -> bool:
    # same heuristics the batch pipeline uses to drop dead air/repetition/boilerplate —
    # a live chunk with no real speech in it should render as nothing, not ghost text
    t = text.lower().strip()
    if no_speech_prob > 0.85 and avg_logprob < -0.9:
        return True
    if len(t) > 12 and compression_ratio(t) > 2.4:
        return True
    return t in BOILERPLATE


def _decode_chunk(path: str) -> np.ndarray:
    # ffmpeg, not faster-whisper's in-process PyAV decoder: PyAV was tried and measured at
    # ~180ms per 7s opus chunk (its frame-by-frame resampler) against ~40ms for ffmpeg.
    return decode_to_array(path, SR)


def _transcribe_live(pool, cfg, path: str, language: str | None):
    """Decode, condition, transcribe one chunk — all in one worker-thread hop, including
    consuming faster-whisper's lazy segment generator (the actual decoding happens as it is
    iterated, and doing that on the event loop blocked the worker for its whole duration).

    Settings measured on pause-cut chunks of continuous speech (RTX 4050, large-v3-turbo):
    the language passed in rather than detected (detection is a second encoder pass, ~250ms
    of a ~350ms chunk — see LIVE_LANG_RECHECK_EVERY), no word timestamps (nothing on the
    live path uses them any more), beam 5 kept (beam 1 saved ~25ms and nothing measurable
    in accuracy, so accuracy wins the tie). Smaller models were measured too: medium was no
    faster and less accurate; small was 2x faster but 7.0% -> 10.3% WER on noisy audio.
    Feeding the previous chunk as a prompt helped clean speech and hurt noisy speech, so it
    is not done.
    """
    t0 = time.perf_counter()
    audio = _decode_chunk(path)
    t1 = time.perf_counter()
    audio = _precondition(audio, cfg)
    t2 = time.perf_counter()
    timing = {"decode_ms": int((t1 - t0) * 1000), "precondition_ms": int((t2 - t1) * 1000)}
    try:
        segments, info = pool.asr.transcribe(
            audio, language=language, task="transcribe", beam_size=cfg.asr_beam_size,
            word_timestamps=False, condition_on_previous_text=False,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
            no_speech_threshold=0.6, vad_filter=True)
        lines = [{"text": s.text.strip(), "start_s": round(s.start, 2), "end_s": round(s.end, 2)}
                 for s in segments
                 if s.text.strip() and not _is_hallucination(s.text, s.avg_logprob, s.no_speech_prob)]
    except ValueError as e:
        # vad_filter drops 100% of a pure-silence chunk and faster-whisper's internal
        # duration calc does max() on the (now empty) speech-chunk list — not a real error
        if "empty sequence" not in str(e):
            raise
        return [], None, 0.0, timing
    timing["asr_ms"] = int((time.perf_counter() - t2) * 1000)
    return lines, info.language, float(info.language_probability), timing


async def transcribe_live_chunk_job(ctx, session_id: str, chunk_rel_path: str, seq: int):
    """One microphone chunk -> its transcript lines. Transcription only, as fast and as
    accurate as this hardware allows; nothing else runs on the live path.

    Takes the model lane at interactive priority (lane.py), so it is served ahead of the
    next stage of any clip being processed instead of waiting out the whole pipeline.

    get_effective_settings, not the startup-cached get_settings: Settings-page edits
    (asr_beam_size, asr_language) reach the next chunk without a worker restart.
    """
    cfg = await get_effective_settings()
    path = storage.resolve(cfg, chunk_rel_path)
    lines: list[dict] = []
    error = None

    # The live channel's whole point is a same-second turnaround; on CPU a chunk takes long
    # enough to make the transcript feel broken rather than slow, so this fails the chunk
    # loudly instead of resolve_device()'s silent fallback quietly degrading every result.
    if ctx["pool"].device != "cuda":
        error = "live transcription requires a GPU; this worker resolved to cpu"
        log.error("live_chunk_requires_gpu", session_id=session_id, seq=seq)
        try:
            os.remove(path)
        except OSError:
            pass
        await emit_live(session_id, seq, lines, error)
        return

    pinned = None if cfg.asr_language == "auto" else cfg.asr_language
    lang_state = None
    lang_arg = pinned
    if pinned is None:
        lang_state = ctx["live_lang"].setdefault(session_id, {"lang": None, "n": 0})
        lang_state["n"] += 1
        # detect on the first chunk (nothing locked yet) and periodically after, so
        # genuine code-switching can catch up; otherwise the locked language is passed in
        recheck = lang_state["lang"] is None or lang_state["n"] % LIVE_LANG_RECHECK_EVERY == 0
        lang_arg = None if recheck else lang_state["lang"]

    t0 = time.perf_counter()
    timing: dict = {}
    try:
        # Whisper's lane only (lane.py): nothing on this path touches another model, so it
        # never waits behind diarization or voiceprints of a clip being processed
        async with ctx["pool"].asr_lane.hold(INTERACTIVE):
            t_lane = time.perf_counter()
            lines, lang, lang_p, timing = await asyncio.to_thread(
                _transcribe_live, ctx["pool"], cfg, path, lang_arg)
        timing["lane_wait_ms"] = int((t_lane - t0) * 1000)
        if lang_state is not None and lang and lang_p > 0.6:
            lang_state["lang"] = lang
    except Exception as e:
        error = str(e)[:200]
        log.warning("live_chunk_failed", session_id=session_id, seq=seq, error=error)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    timing.update(total_ms=int((time.perf_counter() - t0) * 1000), detected_language=lang_arg is None)
    log.info("live_chunk", seq=seq, lines=len(lines), **timing)
    # timing rides along so the page (and anyone measuring) can see where a chunk's time went
    await emit_live(session_id, seq, lines, error, timing=timing)


class WorkerSettings:
    functions = [process_clip_job, transcribe_live_chunk_job,
                 # a corpus-wide retag is many short batches, not one clip, and would trip
                 # the per-clip job_timeout_s long before it finished
                 func(retag_clips_job, timeout=4 * 3600),
                 maintenance_job, reidentify_job, voice_query_job,
                 # every rebuild gets a fresh per-window job id, so there is nothing for a
                 # kept result to dedupe against — keeping it would only litter Redis
                 func(graph_sync_job, keep_result=0)]
    # Every five minutes. arq gives each tick its own job id, so with several worker
    # processes exactly one of them runs it.
    cron_jobs = [cron(maintenance_job, minute=set(range(0, 60, 5)), run_at_startup=False)]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    # Model work is serialized by the lane regardless; the extra slots exist so a live chunk
    # (or graph sync / reidentify) can be picked up while clip jobs occupy the others.
    max_jobs = get_settings().worker_concurrency + 2
    # How often an idle worker checks the queue. arq's 0.5s default was measured as ~480ms
    # of dead time on every live chunk (upload ~10ms, then sitting in the queue until the
    # next poll). A sorted-set range query every 50ms is nothing for Redis.
    poll_delay = 0.05
    job_timeout = get_settings().job_timeout_s
    max_tries = get_settings().job_max_attempts
    retry_delay = 5
    keep_result = 3600
