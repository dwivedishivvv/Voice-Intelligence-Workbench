from dataclasses import dataclass, field
import numpy as np

SR = 16000


@dataclass
class SpeakerResult:
    local_label: str
    embedding: np.ndarray | None = None
    speech_s: float = 0.0
    n_turns: int = 0
    n_segments_used: int = 0
    reliability: float = 0.0
    reliability_reason: str | None = None
    profile_id: str | None = None
    cluster_id: str | None = None
    match_score: float | None = None
    match_margin: float | None = None
    match_result: str = "unknown"
    runner_up_id: str | None = None
    runner_up_score: float | None = None
    talk_share: float | None = None
    longest_turn_s: float | None = None
    interruptions: int | None = None
    overlap_s: float | None = None


@dataclass
class Ctx:
    clip_id: str
    run_id: str
    corr_id: str
    cfg: object
    pool: object

    raw_path: str | None = None
    audio_clean: np.ndarray | None = None
    audio_norm: np.ndarray | None = None
    duration_s: float = 0.0
    probe: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    vad: list = field(default_factory=list)
    speech_s: float = 0.0
    words: list = field(default_factory=list)
    segments: list = field(default_factory=list)
    language: str | None = None
    language_conf: float = 0.0
    turns: list = field(default_factory=list)
    utterances: list = field(default_factory=list)
    speakers: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    transcript_id: str | None = None
    needs_review: bool = False

    def warn(self, code: str, **kw):
        self.warnings.append({"code": code, **kw})
