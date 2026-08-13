"""Every threshold used anywhere in the pipeline comes from here — no literals in stage code."""
from functools import lru_cache
from typing import Literal

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

# Editable from the Settings page too, but unlike TUNABLE_FIELDS these are read once when
# ModelPool is built, so an edit is stored as pending and only bites on the next worker
# start. Kept separate so the "applies to the next job" promise TUNABLE_FIELDS makes stays
# true — the System tab renders these with the restart caveat attached.
RESTART_TUNABLE_FIELDS = {"device"}

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

    # Literal, not str: this is settable from the Settings page, so a typo would otherwise
    # be stored happily and only surface as a crashed worker on the next restart.
    device: Literal["auto", "cpu", "cuda"] = "auto"
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
    min_total_speech_s: float = 1.0

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

    id_threshold: float = 0.68
    id_suggest_delta: float = 0.08
    id_min_margin: float = 0.04
    id_threshold_penalty: float = 0.10
    verify_threshold: float = 0.74
    cluster_threshold: float = 0.76
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
