"""Every threshold used anywhere in the pipeline comes from here — no literals in stage code."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings

# Absolute, not the bare ".env" — pydantic-settings resolves a relative env_file against the
# process's cwd, and the worker runs with cwd=worker/ (see start.ps1), so a relative path
# here silently finds nothing and every setting falls back to its class default instead of
# .env. That was masked for years because most defaults happen to mirror .env's dev
# values — except GRAPH_PASSWORD, which is how this got caught.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

# Fields the Settings page is allowed to override live, via settings_overrides. Deliberately
# excludes model/device/connection fields — those are baked into ModelPool at worker startup
# (or are infra wiring), so changing them without a restart would silently do nothing, which
# is worse than not exposing them at all.
TUNABLE_FIELDS = {
    "max_upload_mb", "max_duration_s", "target_duration_s", "min_duration_s",
    "highpass_hz", "target_lufs", "denoise_enabled", "denoise_prop_decrease",
    "clipping_threshold", "vad_min_speech_ms", "vad_min_silence_ms", "vad_speech_pad_ms",
    "min_total_speech_s",
    "quality_good_snr_db", "quality_fair_snr_db", "quality_max_clipping", "quality_min_bandwidth_hz",
    "diar_min_speakers", "diar_max_speakers", "min_turn_s", "merge_gap_s", "vad_snap_tol_s",
    "overlap_warn_ratio", "smooth_min_conf", "diar_recluster_threshold",
    "embed_min_s", "embed_target_s", "reliability_good", "reliability_fair", "reliability_poor",
    "id_threshold", "id_suggest_delta", "id_min_margin", "id_threshold_penalty", "id_min_z",
    "verify_threshold", "cluster_threshold", "cluster_min_margin", "cluster_merge_threshold",
    "auto_enroll", "auto_enroll_min_sim",
    "auto_enroll_min_reliability", "retention_days", "asr_beam_size", "asr_language",
    "asr_batch_size", "asr_batched_min_s",
    "job_timeout_s", "job_max_attempts",
    "graph_entity_min_mentions", "graph_utterance_nodes",
    "denoise_skip_snr_db", "auto_tag_enabled", "auto_tag_max_topics", "auto_tag_min_topic_sim",
    "auto_maintenance", "auto_consolidate_clusters", "stalled_after_s", "graph_sync_debounce_s",
}

# The agent's own configuration: provider, model, key, and the switch that decides whether
# anything leaves the box at all. Deliberately NOT in TUNABLE_FIELDS — the generic settings
# PATCH refuses every one of these, so they cannot be changed by a caller sweeping the
# settings surface. They have their own endpoint, their own section in the UI, and in the
# case of the keys a write-only contract: a key can be set and replaced, never read back.
#
# This is a weaker posture than .env-only, and the trade is explicit: the operator gets to
# configure the agent without a redeploy, and in exchange anyone holding the app's API key
# can point it at a provider of their choosing. The keys sit in settings_overrides in
# plaintext, like every other override, so the database is now a secret store.
AGENT_TUNABLE_FIELDS = {"llm_enabled", "llm_provider", "llm_model"}
AGENT_SECRET_FIELDS = {"anthropic_api_key", "nvidia_api_key", "groq_api_key"}

# Offered in the UI as a toggle per provider. Not a validation list — llm_model accepts any
# string, because a local vLLM serves whatever it was started with — just the shortlist that
# saves the operator from typing a model id from memory.
LLM_MODELS = {
    "anthropic": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001"],
    # Groq serves these at hundreds of tokens a second, which is the difference between a
    # readable answer and a minute of staring at a spinner.
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-120b"],
    "nvidia": ["meta/llama-3.1-70b-instruct", "meta/llama-3.1-8b-instruct",
               "meta/llama-3.3-70b-instruct"],
}


# Editable from the Settings page too, but unlike TUNABLE_FIELDS these are read once when
# ModelPool is built, so an edit is stored as pending and only bites on the next worker
# start. Kept separate so the "applies to the next job" promise TUNABLE_FIELDS makes stays
# true — the System tab renders these with the restart caveat attached.
RESTART_TUNABLE_FIELDS = {"device"}

# Groups TUNABLE_FIELDS for the Settings page — purely presentational.
SETTINGS_CATEGORIES = {
    "Ingest": ["max_upload_mb", "max_duration_s", "target_duration_s", "min_duration_s"],
    "Pre-processing": ["highpass_hz", "target_lufs", "denoise_enabled", "denoise_prop_decrease",
                        "denoise_skip_snr_db", "clipping_threshold", "vad_min_speech_ms", "vad_min_silence_ms",
                        "vad_speech_pad_ms", "min_total_speech_s"],
    "Quality grading": ["quality_good_snr_db", "quality_fair_snr_db", "quality_max_clipping",
                         "quality_min_bandwidth_hz"],
    "Diarization": ["diar_min_speakers", "diar_max_speakers", "diar_recluster_threshold",
                     "min_turn_s", "merge_gap_s", "vad_snap_tol_s", "overlap_warn_ratio",
                     "smooth_min_conf"],
    "Speaker identification": ["embed_min_s", "embed_target_s", "reliability_good",
                                "reliability_fair", "reliability_poor", "id_threshold",
                                "id_suggest_delta", "id_min_margin", "id_threshold_penalty",
                                "id_min_z", "verify_threshold", "cluster_threshold",
                                "cluster_min_margin", "cluster_merge_threshold", "auto_enroll",
                                "auto_enroll_min_sim", "auto_enroll_min_reliability"],
    "Transcription": ["asr_beam_size", "asr_language", "asr_batch_size", "asr_batched_min_s"],
    "Jobs & retention": ["job_timeout_s", "job_max_attempts", "retention_days"],
    "Tagging & automation": ["auto_tag_enabled", "auto_tag_max_topics", "auto_tag_min_topic_sim",
                              "auto_maintenance", "auto_consolidate_clusters", "stalled_after_s",
                              "graph_sync_debounce_s"],
    # Unlike the rest, this one applies to the next *graph sync* rather than the next job.
    "Graph": ["graph_entity_min_mentions", "graph_utterance_nodes"],
}


class Settings(BaseSettings):
    app_env: str = "production"
    log_level: str = "info"
    log_transcripts: bool = False
    api_key: str = "change-me"

    data_dir: str = "/data"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "workbench"
    postgres_user: str = "workbench"
    postgres_password: str = "change-me"
    redis_url: str = "redis://localhost:6379/0"

    # Graph projection (GRAPH_RAG_PLAN.md). On by default. Deliberately absent from
    # TUNABLE_FIELDS for the same reason the other connection fields are: it is deployment
    # wiring, not a threshold, and Neo4j holds a *derived* read model — with it off, search,
    # the pipeline and every existing page carry on exactly as before.
    graph_enabled: bool = True
    # A named thing (person, company, place, product) is projected as an :Entity only once
    # the transcripts name it at least this many times across the corpus. Capitalised words
    # in ASR output are noisy -- a sentence-initial word, a misheard name -- and one stray
    # mention is far more often that noise than a real subject of the corpus.
    graph_entity_min_mentions: int = 2
    # Project every utterance as its own node. The agent's expand_speech walks these for
    # context; switching it off keeps the graph to people, recordings, topics and entities,
    # which is all the map needs on a corpus too large to draw line by line.
    graph_utterance_nodes: bool = True
    # GRAPH_* rather than NEO4J_*: the neo4j container reads every NEO4J_-prefixed
    # variable in its environment as a server config setting and refuses to start on one
    # it doesn't recognise. Sharing that prefix with the client's own connection settings
    # makes .env un-shareable between the two.
    graph_uri: str = "bolt://localhost:7687"
    graph_user: str = "neo4j"
    graph_password: str = "change-me"

    # Agent layer (AGENT_LAYER_PLAN.md). OFF BY DEFAULT, and not a TUNABLE_FIELD: this is
    # the one setting in the system that changes where data goes. With it false the agent
    # routes 404 and the product's "nothing leaves the box" property holds exactly as
    # before; with it true, transcript excerpts are sent to the Anthropic API as tool
    # results. That is a deployment decision, not a threshold to tweak from a web page.
    llm_enabled: bool = False
    # anthropic | nvidia | groq. The latter two are the OpenAI-compatible path, which also
    # serves any other OpenAI-shaped endpoint (a local vLLM or Ollama server) by pointing
    # that provider's base URL elsewhere — what makes keeping the seam worth it rather than
    # one provider plus a speculative abstraction.
    llm_provider: str = "anthropic"
    anthropic_api_key: str = ""
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    # Empty means "the default for the selected provider" (see agent.DEFAULT_MODELS), so
    # switching provider does not silently keep pointing at the other one's model id.
    llm_model: str = ""
    # Bounds a runaway loop. Cheaper and simpler than a token budget for a first cut; swap
    # in output_config.task_budget if real questions start hitting this legitimately.
    llm_max_iterations: int = 12
    # Per-response cap on the OpenAI-compatible path. Reasoning models need this several
    # times larger, because their chain of thought is billed against the same budget as
    # the answer.
    llm_max_tokens: int = 4096
    llm_effort: str = "high"

    # 90s (a couple of radio calls) was too tight for an upload that's a real recording
    # session rather than a single call — raised to an hour, the top of the range
    # SETTINGS.md already documented as valid. Still a real ceiling (stages/validate.py
    # rejects TOO_LONG above it), not an unbounded accept; still overridable per-deployment
    # from Settings > Ingest without a restart.
    max_upload_mb: int = 500
    max_duration_s: float = 3600.0
    target_duration_s: float = 60.0
    min_duration_s: float = 0.5

    # Literal, not str: this is settable from the Settings page, so a typo would otherwise
    # be stored happily and only surface as a crashed worker on the next restart.
    # default cuda, not auto: resolve_device already falls back to cpu when there's no
    # usable GPU, so this only changes which one you have to say out loud.
    device: Literal["auto", "cpu", "cuda"] = "cuda"
    precision: str = "int8"
    worker_concurrency: int = 2
    job_timeout_s: int = 300
    job_max_attempts: int = 3

    model_dir: str = "/models"
    asr_model: str = "large-v3-turbo"
    asr_beam_size: int = 5
    asr_language: str = "auto"
    # Batched decoding (worker/stages/transcribe.py _batched) for recordings at least
    # asr_batched_min_s long, on GPU: windows are decoded asr_batch_size at a time, and any
    # window the sequential decoder would have retried is re-decoded sequentially. Measured
    # ~2x on 10 minutes of audio with unchanged WER. Short clips gain nothing (one window)
    # and keep the sequential path exactly. 0 turns batching off. Larger batches need more
    # VRAM; an OOM falls back to sequential for that clip rather than failing it.
    asr_batch_size: int = 8
    asr_batched_min_s: float = 60.0
    vad_model: str = "silero-vad"
    diar_model: str = "pyannote-3.1"
    # WeSpeaker ResNet293: ~3x fewer cross-session errors than the previous ECAPA default on
    # degraded and channel-mismatched speech (measurements in worker/speaker_models.py).
    # Switching this re-embeds every stored voiceprint at the next worker start
    # (worker/reembed.py) — vectors from two models are not comparable.
    embed_model: str = "wespeaker-resnet293"
    text_embed_model: str = "minilm"
    hf_hub_offline: int = 1
    transformers_offline: int = 1

    highpass_hz: float = 70.0
    target_lufs: float = -23.0
    denoise_enabled: bool = True
    denoise_prop_decrease: float = 0.75
    # Non-stationary spectral gating is the most expensive non-model step in the pipeline
    # (~260ms per second of audio on CPU), and on a recording that is already clean it buys
    # nothing — there is no noise floor to gate. Above this rough pre-denoise SNR estimate
    # (worker/audio/enhance.py quick_snr_db) the step is skipped. 0 never skips.
    denoise_skip_snr_db: float = 30.0
    clipping_threshold: float = 0.99
    vad_min_speech_ms: int = 200
    vad_min_silence_ms: int = 150
    vad_speech_pad_ms: int = 100
    # A whole radio call is often shorter than a second ("Box, box.", "Copy that."). At a
    # 1.0s floor those were rejected as NO_SPEECH_DETECTED despite VAD finding real speech
    # in them — one test clip had 0.96s of detected speech and was thrown away. 0.5s still
    # rejects the genuinely empty ones (VAD returns 0.00s on those, not 0.6s).
    min_total_speech_s: float = 0.5

    quality_good_snr_db: float = 18.0
    quality_fair_snr_db: float = 10.0
    quality_max_clipping: float = 0.02
    quality_min_bandwidth_hz: float = 3400.0

    diar_min_speakers: int = 1
    # pyannote takes None here to mean "no cap, cluster freely" -- but this field can't
    # actually be Optional: get_settings_view()/_coerce() (api/app/routers/admin.py) infer
    # the Settings-page field type from `f.annotation is int`, and an int|None annotation
    # falls through to "str", so a PATCH from Settings would store this as a raw string and
    # hand pyannote "20" instead of 20 on the next job. 50 is the practical stand-in for
    # unlimited instead -- no real recording has 50 simultaneous speakers.
    diar_max_speakers: int = 50
    # Re-cluster pyannote's turns with the speaker embedder (worker/stages/recluster.py):
    # average-linkage cosine similarity at which two groups of speech are the same person.
    # 0 keeps pyannote's own clustering. Measured on 18 synthetic conversations of known
    # speakers (LibriSpeech, 2-4 people, 25-150s), clean and telephone+noise:
    #
    #                          DER clean  DER degraded  speaker count right (clean/degr.)
    #   pyannote 3.1 alone       10.35%      15.62%        56% / 44%
    #   re-clustered @ 0.20       8.28%       9.54%        78% / 78%
    #   re-clustered @ 0.25       4.40%       7.07%        94% / 83%
    #   re-clustered @ 0.30       4.40%       7.64%        94% / 67%
    #
    # Chosen on that same set, so real recordings may prefer a nearby value: higher splits
    # people more readily, lower merges them.
    diar_recluster_threshold: float = 0.25
    min_turn_s: float = 0.4
    merge_gap_s: float = 0.6
    vad_snap_tol_s: float = 0.15
    overlap_warn_ratio: float = 0.30

    smooth_min_conf: float = 0.60

    embed_min_s: float = 1.5
    embed_target_s: float = 8.0
    reliability_good: float = 0.75
    reliability_fair: float = 0.45
    reliability_poor: float = 0.20

    # Calibrated on 70 F1 team-radio clips (median 3.9s of speech per speaker, heavily
    # compressed) against ground-truth driver labels. ECAPA separates these voices well
    # — same-speaker cosine averaged 0.52, different-speaker 0.15 — but a single short,
    # band-limited embedding never reaches the ~0.75 a clean long-form recording does.
    # At the previous cluster_threshold of 0.76 the same driver re-clustered as a new
    # person 95% of the time (59 clusters out of 65 assignments), which makes cross-clip
    # identity useless. 0.52 recovers 59% of same-speaker pairs while merging different
    # speakers 0.3% of the time — deliberately biased toward "unknown" over a wrong name.
    #
    # These are the right numbers for short radio-style audio, not universal constants.
    # POST /v1/admin/calibrate re-derives them from your own enrolled profiles (EER) once
    # you have >=3 profiles with >=2 enrollments each; prefer that over these defaults.
    #
    # Measured under ECAPA. When the embedder became ResNet293, id_threshold (0.55) and
    # cluster_threshold (0.52) were moved to the ResNet293 scores that admit the same share
    # of impostors on degraded and channel-mismatched trials: 0.55 -> 0.578, 0.52 -> 0.545,
    # rounded down a hair. Same false-accept posture, and fewer genuine misses at it (32.6%
    # -> 29.5% at the identification cutoff). The stricter cutoffs (merge, verify,
    # auto-enrol) sit in a tail with too few impostor scores to map, so they were left alone.
    id_threshold: float = 0.57
    id_suggest_delta: float = 0.08
    id_min_margin: float = 0.04
    id_threshold_penalty: float = 0.10
    # AS-Norm gate (common/speaker.py identify()): how many standard deviations above this
    # embedding's own impostor cohort the winning score has to sit before a name is
    # written. id_min_margin only inspects rank 2, so it cannot separate "clearly the best
    # of a well-spread field" from "top of a crowd where everything scores high" — the
    # second being what a blended or poor-quality embedding looks like. Tightening only:
    # a failed z-test downgrades confident to suggested, it never promotes anything.
    id_min_z: float = 1.5
    verify_threshold: float = 0.62
    cluster_threshold: float = 0.54
    # Same abstain-on-ambiguity posture as id_min_margin, for the clustering path that
    # previously had none. Below this gap between the best and second-best cluster the
    # segment is still filed under the nearer one (a third cluster would only fragment the
    # identity further) but is not allowed to move either centroid.
    cluster_min_margin: float = 0.04
    # Cutoff for consolidate_clusters(). Stricter than cluster_threshold on purpose:
    # declaring two whole clusters to be one person asserts far more than admitting one
    # more segment to a cluster.
    cluster_merge_threshold: float = 0.62
    auto_enroll: bool = False
    auto_enroll_min_sim: float = 0.85
    auto_enroll_min_reliability: float = 0.75

    retention_days: int = 0

    # Automatic clip tags (worker/worker/stages/tag.py). Kept in clips.auto_tags, apart from
    # the operator's own clips.tags, so a re-run can replace them without touching anything a
    # person typed. Topic tags are keyphrases ranked against the clip's own utterance
    # embeddings; below auto_tag_min_topic_sim a phrase is not tagged at all.
    auto_tag_enabled: bool = True
    auto_tag_max_topics: int = 3
    auto_tag_min_topic_sim: float = 0.30

    # Background upkeep the worker runs on a timer (worker/worker/maintenance.py): retention,
    # stalled-clip recovery and tag backfill. Cluster consolidation is switchable on its own
    # and OFF by default: it rewrites which cluster past segments belong to, which the admin
    # endpoint deliberately leaves to a person pressing a button. Turn it on to let the
    # worker run that same pass hourly, at the same strict cluster_merge_threshold.
    auto_maintenance: bool = True
    auto_consolidate_clusters: bool = False
    # A clip whose latest run started this long ago and never finished was lost with a
    # worker (crash, OOM kill, restart mid-job) rather than still being processed.
    stalled_after_s: int = 1800
    # Graph rebuilds requested within this window are coalesced into one. Every rebuild is
    # a full wipe and re-projection, so a 50-clip upload used to run 50 of them back to back.
    graph_sync_debounce_s: int = 20

    class Config:
        env_file = _ENV_FILE
        case_sensitive = False
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def config_snapshot(s: Settings) -> dict:
    """Settings with every secret removed.

    Served by GET /v1/admin/config and stored on every processing run, so anything left in
    here is both readable over the API and durable in the database. The provider keys were
    not on this list until they could be set from the UI; they are secrets wherever they
    came from, and a snapshot is for reproducing a run, which never needs one."""
    d = s.model_dump()
    for k in {"postgres_password", "api_key", "graph_password"} | AGENT_SECRET_FIELDS:
        d.pop(k, None)
    return d


async def get_effective_settings() -> Settings:
    """Env defaults with DB overrides layered on top — call this (not get_settings) anywhere
    a TUNABLE_FIELDS value is read at request/job time, so Settings-page edits take effect
    without a restart."""
    from . import db
    base = get_settings()
    rows = await db.fetch("SELECT key, value FROM settings_overrides")
    if not rows:
        return base
    overrides = {}
    for r in rows:
        field = Settings.model_fields.get(r["key"])
        if field is None:
            continue
        overrides[r["key"]] = _coerce(r["value"], field.annotation)
    return base.model_copy(update=overrides)


def _coerce(raw: str, annotation):
    if annotation is bool:
        return raw.lower() in ("1", "true", "yes", "on")
    if annotation is int:
        return int(raw)
    if annotation is float:
        return float(raw)
    return raw
