from types import SimpleNamespace

import pytest

from common.config import Settings, TUNABLE_FIELDS, RESTART_TUNABLE_FIELDS


def test_device_rejects_unknown_value():
    """device is settable from the Settings page, so a typo has to fail at the PATCH rather
    than being stored and only surfacing as a worker that won't start."""
    with pytest.raises(Exception):
        Settings.model_validate({**Settings().model_dump(), "device": "gpu"})


@pytest.mark.parametrize("value", ["auto", "cpu", "cuda"])
def test_device_accepts_supported_values(value):
    assert Settings.model_validate({**Settings().model_dump(), "device": value}).device == value


def test_restart_tunable_is_disjoint_from_live_tunable():
    """The two sets promise different things to the user ("next job" vs "after restart"),
    so a field in both would make one of those promises a lie."""
    assert not (TUNABLE_FIELDS & RESTART_TUNABLE_FIELDS)


def test_cuda_falls_back_when_unavailable(monkeypatch):
    """Picking cuda from the UI on a box without a GPU must degrade, not take out every
    model load in the pool at once."""
    import torch

    from worker.pool import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device(SimpleNamespace(device="cuda")) == "cpu"
    assert resolve_device(SimpleNamespace(device="auto")) == "cpu"
    assert resolve_device(SimpleNamespace(device="cpu")) == "cpu"


def test_model_dir_without_weights_is_not_downloaded(tmp_path):
    """A gated repo that 401s still leaves a directory behind holding whatever was public
    (pyannote's leaves README.md), so "non-empty" would report it ready to a worker that
    then can't load it."""
    pytest.importorskip("fastapi")
    from api.app.routers.models import _has_weights

    scraps = tmp_path / "gated-401"
    (scraps / "reproducible_research").mkdir(parents=True)
    (scraps / "README.md").write_text("just the public bits")
    assert not _has_weights(scraps)

    (scraps / "config.yaml").write_text("pipeline: ...")
    assert _has_weights(scraps)

    real = tmp_path / "real"
    real.mkdir()
    (real / "model.safetensors").write_bytes(b"\x00")
    assert _has_weights(real)
