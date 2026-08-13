import time
import uuid
from contextvars import ContextVar
import structlog
from prometheus_client import Counter, Histogram, Gauge

_corr: ContextVar[str] = ContextVar("corr_id", default="-")

STAGE_SECONDS = Histogram("clip_stage_seconds", "Per-stage duration", ["stage"])
CLIP_TOTAL = Counter("clip_processing_total", "Clips processed", ["outcome"])
ID_RESULT = Counter("speaker_identification_total", "Identification outcomes", ["result"])


def new_corr_id() -> str:
    return uuid.uuid4().hex[:12]


def bind_correlation(corr_id: str):
    _corr.set(corr_id)


def get_correlation() -> str:
    return _corr.get()


def _add_corr(_, __, ev):
    ev["corr"] = _corr.get()
    return ev


def configure_logging(level: str = "info"):
    structlog.configure(
        processors=[
            _add_corr,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
    )


log = structlog.get_logger()
