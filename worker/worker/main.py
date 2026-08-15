import asyncio
import json
import os

import structlog
from arq.connections import RedisSettings

import numpy as np

from common.config import get_settings, get_effective_settings
from common import db, storage
from .ctx import SR
from .audio.decode import decode_to_array
from .audio.enhance import highpass, loudness_normalize
from .pool import ModelPool
from .pipeline import process_clip
from common.events import init_events, emit_live, emit_f1_result
from .stages.transcribe import compression_ratio, BOILERPLATE
from .audio import tone as tone_mod

log = structlog.get_logger()

# how often (in chunks) a locked live session re-opens language detection instead of
# forcing the pinned language — lets genuine code-switching catch up within ~40s
# (6 chunks * 7s) instead of being stuck on whatever chunk 0 happened to detect
LIVE_LANG_RECHECK_EVERY = 6


async def startup(ctx):
    cfg = get_settings()
    await db.init_pool(cfg)
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
    ctx["gpu_sem"] = ctx["live_sem"] = asyncio.Semaphore(1)
    # per-session language lock + periodic recheck state (see LIVE_LANG_RECHECK_EVERY) —
    # a fresh chunk re-guessing language from scratch every time is unstable on that little audio
    ctx["live_lang"] = {}
    # per-session tone baseline (see tone.new_baseline) — keyed by live session_id or, for
    # F1 radio, f"{session_key}:{driver_number}" so a driver's calls across a race calibrate
    # against each other instead of a fixed global pitch/rate cutoff
    ctx["tone_baseline"] = {}
    log.info("worker_ready", device=ctx["pool"].device, models=ctx["pool"].versions, load_ms=load_ms)


async def shutdown(ctx):
    await db.close_pool()


async def process_clip_job(ctx, clip_id: str):
    async with ctx["gpu_sem"]:
        try:
            # fresh per job (not the startup-cached ctx["cfg"]) so Settings-page edits to
            # TUNABLE_FIELDS apply to the next job without a worker restart
            cfg = await get_effective_settings()
            await process_clip(clip_id, ctx["pool"], cfg)
        except Exception:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise


def _precondition(audio: np.ndarray, cfg) -> np.ndarray:
    # matches the batch pipeline's cheap (non-model) preprocessing step-for-step, minus
    # denoise (noisereduce) — that one alone was ~260ms/s of audio in the batch run, too
    # much of the chunk budget for what raw mic input needs to become intelligible
    audio = audio - float(np.mean(audio))
    audio = highpass(audio, SR, cfg.highpass_hz)
    audio, _ = loudness_normalize(audio, SR, cfg.target_lufs)
    return audio


def _is_hallucination(text: str, avg_logprob: float, no_speech_prob: float) -> bool:
    # same heuristics the batch pipeline uses to drop dead air/repetition/boilerplate —
    # a live chunk with no real speech in it should render as nothing, not ghost text
    t = text.lower().strip()
    if no_speech_prob > 0.85 and avg_logprob < -0.9:
        return True
    if len(t) > 12 and compression_ratio(t) > 2.4:
        return True
    return t in BOILERPLATE


async def transcribe_live_chunk_job(ctx, session_id: str, chunk_rel_path: str, seq: int):
    # transcript + tone only. Denoise is skipped (260ms/s of audio in the batch run), and so
    # is diarization/speaker-ID — per-chunk speaker reads were never stable enough on a few
    # seconds of audio to be worth their latency. Upload the recording for real speaker work.
    cfg = get_settings()
    path = storage.resolve(cfg, chunk_rel_path)
    text, mood, features, error = "", "calm", tone_mod.NEUTRAL_FEATURES, None

    pinned = None if cfg.asr_language == "auto" else cfg.asr_language
    lang_state = None
    lang_arg = pinned
    if pinned is None:
        lang_state = ctx["live_lang"].setdefault(session_id, {"lang": None, "n": 0})
        lang_state["n"] += 1
        # re-open detection on chunk 1 (nothing locked yet) and periodically after —
        # otherwise force the locked language for decode stability on typical short chunks
        recheck = lang_state["lang"] is None or lang_state["n"] % LIVE_LANG_RECHECK_EVERY == 0
        lang_arg = None if recheck else lang_state["lang"]

    try:
        async with ctx["live_sem"]:
            audio = await asyncio.to_thread(decode_to_array, path, SR)
            audio = await asyncio.to_thread(_precondition, audio, cfg)
            duration_s = len(audio) / SR

            # ASR and tone analysis are independent signals from the same audio — a whisper
            # hiccup (vad_filter finding 0 speech in a short/quiet chunk) shouldn't also
            # blank out the tone read, and vice versa (see analyze_f1_radio_job, same pattern)
            try:
                asr_segments, info = await asyncio.to_thread(
                    ctx["pool"].asr.transcribe, audio, language=lang_arg, beam_size=cfg.asr_beam_size,
                    word_timestamps=False, condition_on_previous_text=False,
                    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
                    no_speech_threshold=0.6, vad_filter=True)
                segs = [{"text": s.text.strip(), "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob}
                        for s in asr_segments]
                text = " ".join(s["text"] for s in segs
                                 if not _is_hallucination(s["text"], s["avg_logprob"], s["no_speech_prob"])).strip()
                if lang_state is not None and info.language_probability > 0.6:
                    lang_state["lang"] = info.language
            except ValueError as e:
                # vad_filter drops 100% of a pure-silence chunk and faster-whisper's internal
                # duration calc does max() on the (now empty) speech-chunk list — not a real error
                if "empty sequence" not in str(e):
                    raise

            # a mood reading needs actual words behind it — acoustic-only features (pitch,
            # energy) still "detect" tone on background noise/breathing when ASR found
            # nothing to transcribe, which is how silent chunks were coming back "stressed"
            if text:
                word_count = len(text.split())
                features = await asyncio.to_thread(tone_mod.extract_features, audio, SR, word_count, duration_s)
                baseline = ctx["tone_baseline"].setdefault(session_id, tone_mod.new_baseline())
                mood = tone_mod.classify(features, baseline)
                if mood == "calm":
                    tone_mod.update_baseline(baseline, features)
    except Exception as e:
        error = str(e)[:200]
        log.warning("live_chunk_failed", session_id=session_id, seq=seq, error=error)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    await emit_live(session_id, seq, text, mood, features, error)


async def analyze_f1_radio_job(ctx, radio_call_id: str, rel_path: str, session_key: int | None = None,
                                driver_number: int | None = None):
    cfg = get_settings()
    path = storage.resolve(cfg, rel_path)
    text, mood, features, error = "", "calm", tone_mod.NEUTRAL_FEATURES, None
    baseline_key = f"{session_key}:{driver_number}" if session_key is not None else None
    try:
        async with ctx["live_sem"]:
            audio = await asyncio.to_thread(decode_to_array, path, SR)
            audio = await asyncio.to_thread(_precondition, audio, cfg)
            duration_s = len(audio) / SR

            # ASR and tone analysis are independent signals from the same audio — a
            # whisper hiccup (e.g. vad_filter finding literally 0 speech in a short
            # "copy" acknowledgement) shouldn't also blank out the tone read, and vice versa
            try:
                segments, _ = await asyncio.to_thread(
                    ctx["pool"].asr.transcribe, audio, beam_size=cfg.asr_beam_size, word_timestamps=False,
                    condition_on_previous_text=False, temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
                    no_speech_threshold=0.6, vad_filter=True)
                segs = [{"text": s.text.strip(), "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob}
                        for s in segments]
                text = " ".join(s["text"] for s in segs
                                 if not _is_hallucination(s["text"], s["avg_logprob"], s["no_speech_prob"])).strip()
            except ValueError as e:
                if "empty sequence" not in str(e):
                    raise

            # same reasoning as transcribe_live_chunk_job: no transcribed words means no
            # trustworthy mood read, whatever the acoustic features happen to say
            if text:
                word_count = len(text.split())
                features = await asyncio.to_thread(tone_mod.extract_features, audio, SR, word_count, duration_s)
                baseline = ctx["tone_baseline"].setdefault(baseline_key, tone_mod.new_baseline()) if baseline_key else None
                mood = tone_mod.classify(features, baseline)
                if mood == "calm" and baseline is not None:
                    tone_mod.update_baseline(baseline, features)
    except Exception as e:
        error = str(e)[:200]
        log.warning("f1_radio_failed", radio_call_id=radio_call_id, error=error)
    finally:
        # The audio is not retained: the F1 page plays each call straight from its
        # livetiming.formula1.com URL, so a local copy would serve nothing today. Keep
        # it here if radio ever goes through the full clip pipeline, which needs a file.
        try:
            os.remove(path)
        except OSError:
            pass

    # Persist before emitting, so a client that reconnects after the websocket message
    # has already gone out can still read the result back via GET /v1/f1/analyses.
    await db.execute(
        """UPDATE radio_calls SET text=$2, mood=$3, features=$4, error=$5, analyzed_at=now()
           WHERE id=$1""",
        radio_call_id, text, mood, json.dumps(features) if features else None, error)
    await emit_f1_result(radio_call_id, text, mood, features, error)


class WorkerSettings:
    functions = [process_clip_job, transcribe_live_chunk_job, analyze_f1_radio_job]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = get_settings().worker_concurrency
    job_timeout = get_settings().job_timeout_s
    max_tries = get_settings().job_max_attempts
    retry_delay = 5
    keep_result = 3600
