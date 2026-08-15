"""Every threshold used anywhere in the pipeline comes from here — no literals in stage code."""
from functools import lru_cache
from pydantic_settings import BaseSettings

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
    "overlap_warn_ratio", "smooth_min_conf",
    "embed_min_s", "embed_target_s", "reliability_good", "reliability_fair", "reliability_poor",
    "id_threshold", "id_suggest_delta", "id_min_margin", "id_threshold_penalty",
    "verify_threshold", "cluster_threshold", "auto_enroll", "auto_enroll_min_sim",
    "auto_enroll_min_reliability", "retention_days", "asr_beam_size", "asr_language",
    "job_timeout_s", "job_max_attempts",
}

# Groups TUNABLE_FIELDS for the Settings page — purely presentational.
SETTINGS_CATEGORIES = {
    "Ingest": ["max_upload_mb", "max_duration_s", "target_duration_s", "min_duration_s"],
    "Pre-processing": ["highpass_hz", "target_lufs", "denoise_enabled", "denoise_prop_decrease",
                        "clipping_threshold", "vad_min_speech_ms", "vad_min_silence_ms",
                        "vad_speech_pad_ms", "min_total_speech_s"],
    "Quality grading": ["quality_good_snr_db", "quality_fair_snr_db", "quality_max_clipping",
                         "quality_min_bandwidth_hz"],
    "Diarization": ["diar_min_speakers", "diar_max_speakers", "min_turn_s", "merge_gap_s",
                     "vad_snap_tol_s", "overlap_warn_ratio", "smooth_min_conf"],
    "Speaker identification": ["embed_min_s", "embed_target_s", "reliability_good",
                                "reliability_fair", "reliability_poor", "id_threshold",
                                "id_suggest_delta", "id_min_margin", "id_threshold_penalty",
                                "verify_threshold", "cluster_threshold", "auto_enroll",
                                "auto_enroll_min_sim", "auto_enroll_min_reliability"],
    "Transcription": ["asr_beam_size", "asr_language"],
    "Jobs & retention": ["job_timeout_s", "job_max_attempts", "retention_days"],
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

    max_upload_mb: int = 50
    max_duration_s: float = 90.0
    target_duration_s: float = 60.0
    min_duration_s: float = 0.5

    device: str = "auto"
    precision: str = "int8"
    worker_concurrency: int = 2
    job_timeout_s: int = 300
    job_max_attempts: int = 3

    model_dir: str = "/models"
    asr_model: str = "large-v3-turbo"
    asr_beam_size: int = 5
    asr_language: str = "auto"
    vad_model: str = "silero-vad"
    diar_model: str = "pyannote-3.1"
    embed_model: str = "ecapa"
    text_embed_model: str = "minilm"
    sentiment_model: str = "xlmr-sentiment"
    hf_hub_offline: int = 1
    transformers_offline: int = 1

    highpass_hz: float = 70.0
    target_lufs: float = -23.0
    denoise_enabled: bool = True
    denoise_prop_decrease: float = 0.75
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
    diar_max_speakers: int = 4
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
    id_threshold: float = 0.55
    id_suggest_delta: float = 0.08
    id_min_margin: float = 0.04
    id_threshold_penalty: float = 0.10
    verify_threshold: float = 0.62
    cluster_threshold: float = 0.52
    auto_enroll: bool = False
    auto_enroll_min_sim: float = 0.85
    auto_enroll_min_reliability: float = 0.75

    retention_days: int = 0

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def config_snapshot(s: Settings) -> dict:
    d = s.model_dump()
    for k in ("postgres_password", "api_key"):
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
