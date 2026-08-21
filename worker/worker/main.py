import asyncio
import json
import os

import structlog
from arq.connections import RedisSettings

import numpy as np

from common.config import get_settings, get_effective_settings
from common import db, storage, speaker as sc
from .ctx import SR
from .audio.decode import decode_to_array
from .audio.enhance import highpass, loudness_normalize
from .audio.vad import detect_vad
from .pool import ModelPool
from .pipeline import process_clip, _sync_graph
from common.events import init_events, emit_live, emit_f1_result
from .stages.transcribe import compression_ratio, BOILERPLATE
from .stages import (diarize as diarize_mod, reconcile as reconcile_mod,
                      sentiment as sentiment_mod, embed as embed_mod)
from .audio import tone as tone_mod
from .audio.quality import rolloff_99

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
    # per-session voice pooling (see _pool_voice) — keyed by live session_id. Neither this
    # nor the two dicts above is ever evicted; a worker that runs for a very long time
    # without restarting accumulates one small entry per session forever.
    ctx["live_voices"] = {}
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


def _live_turn_prep(pool, cfg, audio: np.ndarray, utterances: list[dict], noise_power: float):
    """All the numpy/GPU work for every turn in one chunk, in a single thread hop.

    Everything here used to run per turn from the async loop: one `to_thread` for tone
    features, another for the embedding, and a separate encoder round trip each. On a chunk
    that diarized into three turns that was six hops and three GPU calls; it is now one hop
    and one call, which is the difference the live path actually feels.
    """
    prepped: list[dict | None] = []
    segs: list[np.ndarray] = []
    for u in utterances:
        a, b = int(u["start"] * SR), int(u["end"] * SR)
        seg = audio[a:b]
        dur = seg.size / SR
        if dur < 0.2:
            prepped.append(None)
            continue
        entry: dict = {"dur": dur, "features": None, "quality": None, "row": None}
        text = u.get("text", "")
        if text:
            entry["features"] = tone_mod.extract_features(seg, SR, len(text.split()), dur)
        if dur >= LIVE_EMBED_MIN_S:
            # the same per-segment quality the batch pipeline measures, instead of the
            # hardcoded mid-range defaults reliability_score falls back to on an empty dict
            entry["quality"] = {
                "snr_db": embed_mod._snr_db(seg, noise_power),
                "clipping_ratio": float(np.mean(np.abs(seg) >= cfg.clipping_threshold)),
                "bandwidth_hz": rolloff_99(seg, SR),
            }
            entry["row"] = len(segs)
            segs.append(seg)
        prepped.append(entry)

    embs = embed_mod.encode_many(pool, segs) if segs else None
    return prepped, embs


def _is_hallucination(text: str, avg_logprob: float, no_speech_prob: float) -> bool:
    # same heuristics the batch pipeline uses to drop dead air/repetition/boilerplate —
    # a live chunk with no real speech in it should render as nothing, not ghost text
    t = text.lower().strip()
    if no_speech_prob > 0.85 and avg_logprob < -0.9:
        return True
    if len(t) > 12 and compression_ratio(t) > 2.4:
        return True
    return t in BOILERPLATE


def _filter_hallucinated_words(words: list[dict], segs: list[dict]) -> list[dict]:
    # word-level version of stages/transcribe.py's drop_hallucinations, minus the ctx.warn
    # (nothing here reviews live chunks the way a rejected clip gets reviewed)
    bad_spans = [(s["start"], s["end"]) for s in segs
                 if _is_hallucination(s["text"], s["avg_logprob"], s["no_speech_prob"])]
    if not bad_spans:
        return words
    return [w for w in words if not any(a <= w["start"] < b for a, b in bad_spans)]


def _diarize_chunk(pool, audio: np.ndarray, cfg) -> list[dict]:
    import torch
    ann = pool.diar({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SR},
                     min_speakers=cfg.diar_min_speakers, max_speakers=cfg.diar_max_speakers)
    return [{"start": seg.start, "end": seg.end, "label": lbl}
            for seg, _, lbl in ann.itertracks(yield_label=True)]


def _pool_voice(voices: list[dict], emb: np.ndarray, dur: float, threshold: float) -> tuple[np.ndarray, float, int]:
    """Online, session-local speaker clustering — the same nearest-centroid geometry as
    common.speaker.assign_cluster, but in-process and never written to the DB.

    Every live turn used to call identify() on its own single embedding, in complete
    isolation. That is a much thinner signal than batch identification ever sees:
    stages/embed.py pools every turn of one speaker in a clip, up to embed_target_s
    (8s default) of audio, into ONE embedding before identify() runs on it at all — a
    two-word turn ("Hallo.", "but") never gets identified alone in batch mode either. Live
    mode has no clip to pool within, so this pools across chunks instead: each new turn's
    embedding is compared against the voices already heard this session, and folded into
    the nearest one (running weighted average) if it's a close enough match, or seeds a new
    one otherwise. identify() then runs on the pooled centroid — the more a voice has
    talked this session, the more of stages/embed.py's pooling advantage it gets back.

    The returned index is this session's own running voice count, position-stable for the
    life of the session (voices are only ever appended, never reordered or dropped) — the
    "Grouped" tab in the UI uses it to bucket turns nobody has named yet.
    """
    best_idx, best_sim = -1, -1.0
    for i, v in enumerate(voices):
        sim = float(emb @ v["centroid"])
        if sim > best_sim:
            best_idx, best_sim = i, sim
    if best_idx >= 0 and best_sim >= threshold:
        v = voices[best_idx]
        n = v["n"]
        c = (v["centroid"] * n + emb) / (n + 1)
        v["centroid"] = c / max(float(np.linalg.norm(c)), 1e-6)
        v["n"] += 1
        v["total_s"] += dur
        return v["centroid"], v["total_s"], best_idx
    voices.append({"centroid": emb, "n": 1, "total_s": dur})
    return emb, dur, len(voices) - 1


async def _process_live_turns(ctx, cfg, session_id: str, audio: np.ndarray,
                               utterances: list[dict], baseline: dict,
                               noise_power: float) -> list[dict]:
    """Every diarized turn in one live chunk: tone, sentiment and a best-effort speaker
    match — the same signals the batch pipeline derives, on the little context a live
    chunk has.

    Batched deliberately rather than looped: the tone/quality/embedding work happens in one
    thread hop for the whole chunk, and the sentiment classifier sees every turn in a single
    forward pass. Only the parts that genuinely have to be sequential stay so — the tone
    baseline updates turn by turn, and identify() hits the database.
    """
    prepped, embs = await asyncio.to_thread(
        _live_turn_prep, ctx["pool"], cfg, audio, utterances, noise_power)

    moods: list[str] = []
    for u, p in zip(utterances, prepped):
        if p is None or not u.get("text") or p["features"] is None:
            moods.append("calm")
            continue
        mood = tone_mod.classify(p["features"], baseline)
        if mood == "calm":
            tone_mod.update_baseline(baseline, p["features"])
        moods.append(mood)

    # Same text-dominant fusion stages/sentiment.py runs over a whole clip: the trained
    # text model sets the score and acoustic delivery only nudges it by MOOD_SHIFT, never
    # far enough to manufacture a verdict the words do not support.
    fused: list[dict | None] = [None] * len(utterances)
    if getattr(ctx["pool"], "sentiment", None) is not None:
        idxs = [i for i, u in enumerate(utterances) if u.get("text") and prepped[i] is not None]
        if idxs:
            results = await asyncio.to_thread(
                sentiment_mod.fuse_many, ctx["pool"],
                [(utterances[i]["text"], moods[i]) for i in idxs])
            for i, r in zip(idxs, results):
                fused[i] = r

    voices = ctx["live_voices"].setdefault(session_id, [])
    out: list[dict] = []
    for i, (u, p) in enumerate(zip(utterances, prepped)):
        if p is None:
            continue
        text = u.get("text", "")
        speaker = speaker_result = speaker_score = None
        embedding = reliability = voice_cluster = None

        if p["row"] is not None and embs is not None:
            pooled_emb, pooled_s, voice_cluster = _pool_voice(
                voices, embs[p["row"]], p["dur"], cfg.cluster_threshold)
            reliability = sc.reliability_score(
                min(pooled_s, cfg.embed_target_s), p["quality"] or {}, cfg)
            match = await sc.identify(pooled_emb, reliability, cfg)
            if match.result in ("confident", "suggested") and match.profile_id:
                speaker = await db.fetchval(
                    "SELECT display_name FROM speaker_profiles WHERE id=$1", match.profile_id)
                speaker_result, speaker_score = match.result, match.score
            # The pooled centroid, not this turn's raw embedding: it is the stronger of the
            # two, and it is carried out even on a miss so an unmatched (or only-suggested)
            # turn can be named from the UI without re-recording — POST /v1/live/tag
            # enrolls exactly this vector.
            embedding = pooled_emb.tolist()

        if not text and speaker is None:
            continue

        f = fused[i] or {}
        out.append({"text": text, "mood": moods[i], "speaker": speaker,
                    "speaker_result": speaker_result, "speaker_score": speaker_score,
                    "start_s": u["start"], "end_s": u["end"],
                    "embedding": embedding, "reliability": reliability,
                    "voice_cluster": voice_cluster,
                    "sentiment": f.get("sentiment"), "sentiment_score": f.get("sentiment_score"),
                    "text_sentiment": f.get("text_sentiment"), "text_score": f.get("text_score")})
    return out


async def transcribe_live_chunk_job(ctx, session_id: str, chunk_rel_path: str, seq: int):
    # Full diarize -> assign_words -> regroup pipeline (stages/diarize.py,
    # stages/reconcile.py) scoped to one chunk instead of a whole clip, so a chunk with more
    # than one voice in it (two people sharing a mic, cross-talk on a radio channel) comes
    # back as more than one turn — matching upload/batch mode instead of the flat
    # one-speaker-per-chunk assumption the live path used before this. Each turn gets its
    # own tone read and speaker match, same as before, just per turn now instead of per chunk.
    #
    # get_effective_settings, not the startup-cached get_settings: without this, Settings-page
    # edits to any TUNABLE_FIELDS (asr_beam_size, id_threshold, ...) never reached the live
    # path — process_clip_job already re-reads per job for exactly this reason.
    cfg = await get_effective_settings()
    path = storage.resolve(cfg, chunk_rel_path)
    turns_out: list[dict] = []
    error = None

    # The live channel's whole point is a same-second turnaround; on CPU a chunk this size
    # takes long enough to make the chunked-transcript UI feel broken rather than slow, so
    # this fails the chunk loudly instead of resolve_device()'s silent fallback quietly
    # degrading every "live" result. Batch clip processing is unaffected — this only gates
    # the live path.
    if ctx["pool"].device != "cuda":
        error = "live transcription requires a GPU; this worker resolved to cpu"
        log.error("live_chunk_requires_gpu", session_id=session_id, seq=seq)
        try:
            os.remove(path)
        except OSError:
            pass
        await emit_live(session_id, seq, turns_out, error)
        return

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

            words: list[dict] = []
            try:
                asr_segments, info = await asyncio.to_thread(
                    ctx["pool"].asr.transcribe, audio, language=lang_arg, task="transcribe",
                    beam_size=cfg.asr_beam_size, word_timestamps=True,
                    condition_on_previous_text=False,
                    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
                    no_speech_threshold=0.6, vad_filter=True)
                segs = []
                for si, s in enumerate(asr_segments):
                    segs.append({"idx": si, "start": s.start, "end": s.end, "text": s.text.strip(),
                                 "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob})
                    for w in (s.words or []):
                        words.append({"word": w.word.strip(), "start": w.start, "end": w.end,
                                      "probability": w.probability})
                words = _filter_hallucinated_words(words, segs)
                if lang_state is not None and info.language_probability > 0.6:
                    lang_state["lang"] = info.language
            except ValueError as e:
                # vad_filter drops 100% of a pure-silence chunk and faster-whisper's internal
                # duration calc does max() on the (now empty) speech-chunk list — not a real error
                if "empty sequence" not in str(e):
                    raise

            if words:
                # Deliberately sequential, not gathered: these touch two different models
                # in the one shared ModelPool, and startup() documents that concurrent use
                # of that pool took the process down with an untraceable heap corruption.
                # A little latency is the cheaper side of that trade.
                raw_turns = await asyncio.to_thread(_diarize_chunk, ctx["pool"], audio, cfg)
                vad = await asyncio.to_thread(detect_vad, audio, SR, ctx["pool"].vad, cfg)
                turns = diarize_mod.clean_turns(raw_turns, vad, cfg)
                if not turns:
                    # diarization found nothing worth splitting (the common case on a short
                    # VAD-bounded phrase) -- one speaker across the whole chunk, the same
                    # fallback the pre-diarization live path always assumed
                    turns = [{"start": 0.0, "end": duration_s, "label": "spk0",
                              "is_overlap": False, "too_short": False}]

                reconcile_mod.assign_words(words, turns)
                reconcile_mod.smooth(words, cfg.smooth_min_conf)
                utterances = reconcile_mod.regroup(words)

                baseline = ctx["tone_baseline"].setdefault(session_id, tone_mod.new_baseline())
                turns_out = await _process_live_turns(
                    ctx, cfg, session_id, audio, utterances, baseline,
                    _live_noise_power(audio, turns))
    except Exception as e:
        error = str(e)[:200]
        log.warning("live_chunk_failed", session_id=session_id, seq=seq, error=error)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    await emit_live(session_id, seq, turns_out, error)


async def analyze_f1_radio_job(ctx, radio_call_id: str, rel_path: str, session_key: int | None = None,
                                driver_number: int | None = None):
    cfg = await get_effective_settings()  # see transcribe_live_chunk_job — same reasoning
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
    if not error:
        await _sync_graph(cfg)


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
