"""Placeholder numbering in the search SQL.

The filters are optional and the fixed parameters ahead of them differ per mode, so the
$-indices shift depending on which filters are set. Getting that wrong does not raise: it
binds the mood string to the profile_id comparison, or vice versa, and returns an empty
result that reads as "the corpus has none of those" — the exact failure mode the mood
filter exists to fix.

Nothing here touches Postgres; db.fetch is captured so the built SQL and its arguments can
be asserted directly.
"""
import asyncio

import pytest

from api.app.services import search as search_svc


@pytest.fixture
def captured(monkeypatch):
    """Record every query the module issues, and hand back no rows."""
    calls = []

    async def fake_fetch(sql, *args):
        calls.append({"sql": " ".join(sql.split()), "args": list(args)})
        return []

    monkeypatch.setattr(search_svc.db, "fetch", fake_fetch)
    monkeypatch.setattr(search_svc.text_embed, "encode", lambda q: [0.1, 0.2, 0.3])
    return calls


def test_filters_are_numbered_from_the_first_free_placeholder():
    frag, args = search_svc._utterance_filters(3, "spk", "stressed")
    assert frag == "AND u.profile_id = $3 AND u.mood = $4"
    assert args == ["spk", "stressed"]


def test_a_single_filter_takes_the_first_slot_whichever_one_it_is():
    """The second filter must not reserve a placeholder when it is absent — that would
    leave a hole and bind every later argument one position off."""
    assert search_svc._utterance_filters(2, None, "calm") == ("AND u.mood = $2", ["calm"])
    assert search_svc._utterance_filters(2, "spk", None) == ("AND u.profile_id = $2", ["spk"])
    assert search_svc._utterance_filters(2, None, None) == ("", [])


@pytest.mark.parametrize("mode,fixed", [("fts", 1), ("semantic", 1), ("hybrid", 2)])
def test_every_mode_binds_filters_after_its_own_fixed_parameters(captured, mode, fixed):
    """fts and semantic take one fixed parameter, hybrid takes two. The filters follow."""
    asyncio.run(search_svc.hybrid_search("tyres", mode=mode, speaker_id="spk",
                                          mood="stressed", limit=5))
    q = captured[0]
    assert f"AND u.profile_id = ${fixed + 1}" in q["sql"]
    assert f"AND u.mood = ${fixed + 2}" in q["sql"]
    assert q["args"][fixed:] == ["spk", "stressed"]


def test_hybrid_applies_the_filters_to_both_legs(captured):
    """RRF fuses two rankings. Filtering only the FTS leg would let the vector leg
    reintroduce exactly the rows the caller asked to exclude."""
    asyncio.run(search_svc.hybrid_search("tyres", speaker_id="spk", mood="calm"))
    assert captured[0]["sql"].count("AND u.profile_id = $3 AND u.mood = $4") == 2


def test_mood_reaches_the_radio_leg_too(captured):
    """Radio calls are most of the F1 corpus and carry their own mood column. A tone filter
    that only narrowed utterances would quietly return unfiltered radio alongside it."""
    asyncio.run(search_svc.anchor_speech("tyres", limit=5, mood="tired"))
    radio = [c for c in captured if "radio_calls" in c["sql"]]
    assert radio, "anchor_speech did not search radio at all"
    assert "AND rc.mood = $3" in radio[0]["sql"]
    assert radio[0]["args"][-1] == "tired"


def test_tone_browse_reads_both_sources_and_interleaves_them(monkeypatch):
    """Concatenating and truncating would return one source only whenever it has enough
    rows, hiding the other half of the corpus behind a limit."""
    async def fake_fetch(sql, *args):
        if "utterances" in sql:
            return [{"utterance_id": f"u{i}", "text": "u", "mood": "stressed"} for i in range(5)]
        return [{"speech_id": f"r{i}", "text": "r", "mood": "stressed"} for i in range(5)]

    monkeypatch.setattr(search_svc.db, "fetch", fake_fetch)
    rows = asyncio.run(search_svc.speech_by_mood("stressed", limit=4))
    assert [r["kind"] for r in rows] == ["utterance", "radio_call", "utterance", "radio_call"]
    assert all(r["mood"] == "stressed" for r in rows)


def test_tone_browse_drops_radio_when_scoped_to_a_speaker(monkeypatch):
    """Radio calls carry a driver, not a speaker profile. Including them under a speaker
    filter would attribute speech to someone the corpus never placed there."""
    seen = []

    async def fake_fetch(sql, *args):
        seen.append(sql)
        return []

    monkeypatch.setattr(search_svc.db, "fetch", fake_fetch)
    asyncio.run(search_svc.speech_by_mood("calm", limit=4, speaker_id="spk"))
    assert not any("radio_calls" in s for s in seen)
