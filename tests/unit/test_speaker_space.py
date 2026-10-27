"""Cross-recording identity: the migration decision (worker/reembed.py) and retroactive
re-identification (common/speaker.py reidentify). Both decide, without a person watching,
whether stored voice attributions change — so the rules are pinned here."""
import asyncio
from types import SimpleNamespace

import numpy as np

from common import speaker as sc
from worker.reembed import needs_migration

# ---- migration decision -----------------------------------------------------------------


def test_width_mismatch_always_migrates():
    """192-dim ECAPA vectors next to a 256-dim model cannot even be compared."""
    assert needs_migration(None, 192, "wespeaker-resnet293", 256)
    assert needs_migration({"model": "wespeaker-resnet293", "dim": 256}, 192, "wespeaker-resnet293", 256)


def test_same_width_different_model_migrates():
    """Two 256-dim models are unrelated spaces; the stamp, not the width, catches it."""
    assert needs_migration({"model": "wespeaker-resnet34", "dim": 256}, 256, "wespeaker-resnet293", 256)


def test_matching_stamp_is_left_alone():
    assert not needs_migration({"model": "wespeaker-resnet293", "dim": 256}, 256, "wespeaker-resnet293", 256)


def test_unstamped_store_at_the_right_width_is_not_thrown_away():
    """A fresh install (init.sql already at the model's width) has nothing to migrate."""
    assert not needs_migration(None, 256, "wespeaker-resnet293", 256)


# ---- re-identification --------------------------------------------------------------------


class FakeDB:
    def __init__(self, rows):
        self.rows = rows
        self.updates = []

    async def fetch(self, sql, *args):
        assert "NOT is_manual" in sql, "manual attributions must never be re-identified"
        return self.rows

    async def execute(self, sql, *args):
        self.updates.append((" ".join(sql.split()), args))


def _row(label, profile=None, result="unknown", cluster=None):
    return {"clip_id": "c1", "local_label": label, "embedding": np.ones(4, dtype=np.float32) / 2,
            "reliability": 0.9, "profile_id": profile, "match_result": result, "cluster_id": cluster}


def _run(monkeypatch, rows, match):
    fake = FakeDB(rows)
    monkeypatch.setattr(sc, "db", fake)

    async def fake_identify(emb, rel, cfg):
        return match

    monkeypatch.setattr(sc, "identify", fake_identify)
    return fake, asyncio.run(sc.reidentify(SimpleNamespace()))


def test_a_newly_matching_voice_is_named_and_leaves_the_unnamed_pool(monkeypatch):
    m = sc.Match("p-ana", 0.81, 0.3, "confident", "p-ben", 0.2, 4.0)
    fake, out = _run(monkeypatch, [_row("SPEAKER_00", cluster="k1")], m)
    assert out["changed"] == [{"clip_id": "c1", "local_label": "SPEAKER_00", "from": None,
                               "to": "p-ana", "was": "unknown", "now": "confident"}]
    cs_update = next(a for s, a in fake.updates if s.startswith("UPDATE clip_speakers"))
    assert cs_update[2] == "p-ana" and cs_update[-1] is None          # cluster cleared
    assert any(s.startswith("UPDATE utterances") for s, _ in fake.updates)


def test_a_suggestion_keeps_its_cluster(monkeypatch):
    """Only a confident claim takes a voice out of the unnamed pool; a suggestion is a
    question for a person, and the voice should still be findable as unclaimed."""
    m = sc.Match("p-ana", 0.52, 0.1, "suggested", None, None, None)
    fake, _ = _run(monkeypatch, [_row("SPEAKER_00", cluster="k1")], m)
    cs_update = next(a for s, a in fake.updates if s.startswith("UPDATE clip_speakers"))
    assert cs_update[-1] == "k1"


def test_unchanged_attribution_does_not_touch_the_transcript(monkeypatch):
    m = sc.Match("p-ana", 0.8, 0.3, "confident", None, None, None)
    fake, out = _run(monkeypatch, [_row("SPEAKER_00", profile="p-ana", result="confident")], m)
    assert out["changed"] == []
    assert not any(s.startswith("UPDATE utterances") for s, _ in fake.updates)


def test_a_voice_no_profile_claims_any_more_is_unnamed(monkeypatch):
    """After a profile is deleted or loses its samples, a machine attribution that no
    longer holds goes back to unknown rather than keeping a name nothing supports."""
    m = sc.Match(None, 0.3, 0.0, "unknown")
    _, out = _run(monkeypatch, [_row("SPEAKER_01", profile="p-gone", result="confident")], m)
    assert out["changed"][0]["to"] is None and out["changed"][0]["now"] == "unknown"
