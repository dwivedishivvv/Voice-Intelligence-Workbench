"""Placeholder numbering in the search SQL.

The filters are optional and the fixed parameters ahead of them differ per mode, so the
$-indices shift depending on which filters are set. Getting that wrong does not raise: it
binds the clip id to the profile_id comparison, or vice versa, and returns an empty result
that reads as "the corpus has none of those".

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
    frag, args = search_svc._utterance_filters(3, "spk", None, "clip")
    assert frag == "AND u.profile_id = $3 AND u.clip_id = $4"
    assert args == ["spk", "clip"]


def test_a_single_filter_takes_the_first_slot_whichever_one_it_is():
    """An absent filter must not reserve a placeholder — that would leave a hole and bind
    every later argument one position off."""
    assert search_svc._utterance_filters(2, None, None, "clip") == ("AND u.clip_id = $2", ["clip"])
    assert search_svc._utterance_filters(2, "spk") == ("AND u.profile_id = $2", ["spk"])
    assert search_svc._utterance_filters(2, None) == ("", [])


def test_tags_follow_the_scalar_filters():
    frag, args = search_svc._utterance_filters(2, "spk", ["lang:en"])
    assert frag.startswith("AND u.profile_id = $2 AND u.clip_id IN")
    assert "$3 = ANY(tags)" in frag and args == ["spk", "lang:en"]


@pytest.mark.parametrize("mode,fixed", [("fts", 1), ("semantic", 1), ("hybrid", 2)])
def test_every_mode_binds_filters_after_its_own_fixed_parameters(captured, mode, fixed):
    """fts and semantic take one fixed parameter, hybrid takes two. The filters follow."""
    asyncio.run(search_svc.hybrid_search("tyres", mode=mode, speaker_id="spk",
                                          clip_id="clip", limit=5))
    q = captured[0]
    assert f"AND u.profile_id = ${fixed + 1}" in q["sql"]
    assert f"AND u.clip_id = ${fixed + 2}" in q["sql"]
    assert q["args"][fixed:] == ["spk", "clip"]


def test_hybrid_applies_the_filters_to_both_legs(captured):
    """RRF fuses two rankings. Filtering only the FTS leg would let the vector leg
    reintroduce exactly the rows the caller asked to exclude."""
    asyncio.run(search_svc.hybrid_search("tyres", speaker_id="spk", clip_id="clip"))
    assert captured[0]["sql"].count("AND u.profile_id = $3 AND u.clip_id = $4") == 2


def test_anchor_speech_keys_hits_by_speech_id(monkeypatch):
    async def fake_hybrid(q, **kw):
        return [{"utterance_id": "u1", "text": "hi", "score": 0.5}]

    monkeypatch.setattr(search_svc, "hybrid_search", fake_hybrid)
    rows = asyncio.run(search_svc.anchor_speech("hi"))
    assert rows == [{"speech_id": "u1", "kind": "utterance", "text": "hi", "score": 0.5}]
