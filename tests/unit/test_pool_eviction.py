"""ModelPool loads models lazily and, on CUDA OOM, evicts the lowest-priority cached model
and retries — this is the one piece of real branching logic in pool.py, so it gets a check
independent of actually having a GPU or the real model weights."""
from worker.pool import ModelPool, _PRIORITY


def _bare_pool(device="cuda"):
    pool = ModelPool.__new__(ModelPool)
    pool.cfg = None
    pool.device = device
    pool._cache = {}
    return pool


def test_evicts_lowest_priority_first():
    pool = _bare_pool()
    pool._cache = {name: object() for name in _PRIORITY}
    assert pool._evict_lowest_priority(exclude="asr") == _PRIORITY[0]
    assert _PRIORITY[0] not in pool._cache


def test_evict_skips_excluded_and_uncached():
    pool = _bare_pool()
    pool._cache = {"asr": object()}
    assert pool._evict_lowest_priority(exclude="asr") is None


def test_load_retries_after_evicting_on_oom(monkeypatch):
    pool = _bare_pool()
    pool._cache = {"sentiment": object()}  # occupying the slot that should get evicted
    calls = []

    def fake_load_diar():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory.")
        return "diar-model"

    monkeypatch.setattr(pool, "_load_diar", fake_load_diar)
    monkeypatch.setattr(pool, "_warm", lambda name, obj: None)

    result = pool._get("diar")
    assert result == "diar-model"
    assert "sentiment" not in pool._cache  # evicted to make room
    assert len(calls) == 2  # first attempt OOM'd, retry succeeded


def test_non_oom_error_propagates_without_evicting(monkeypatch):
    pool = _bare_pool()
    pool._cache = {"sentiment": object()}
    monkeypatch.setattr(pool, "_load_diar", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    try:
        pool._get("diar")
        assert False, "expected RuntimeError to propagate"
    except RuntimeError as e:
        assert "boom" in str(e)
    assert "sentiment" in pool._cache  # untouched — this wasn't an OOM


def test_cpu_never_evicts(monkeypatch):
    pool = _bare_pool(device="cpu")
    pool._cache = {"sentiment": object()}
    monkeypatch.setattr(pool, "_load_diar", lambda: (_ for _ in ()).throw(RuntimeError("out of memory")))

    try:
        pool._get("diar")
        assert False, "CPU OOM should propagate, not trigger eviction"
    except RuntimeError:
        pass
    assert "sentiment" in pool._cache
