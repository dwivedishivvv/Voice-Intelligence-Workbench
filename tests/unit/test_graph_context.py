"""render_context turns a graph expansion into the text a person or a model actually
reads. It is the last step before a claim gets made about someone's audio, so the things
asserted here are mostly about honesty rather than formatting:

  - every block carries its speech id, or a downstream claim cannot be cited back to the
    recording it came from;
  - tone is phrased as a reading, never as a fact, because the classifier behind it is a
    threshold heuristic;
  - text/acoustic disagreement is surfaced rather than averaged away;
  - a missing neighbourhood renders as absence, not as a guess.

Pure function, so none of this needs Postgres or Neo4j.
"""
import asyncio

import pytest

from api.app.services.graph_context import render_context

FULL = {
    "speech_id": "abc-123",
    "text": "I can't get past, the tyres are gone",
    "mood": "stressed", "sentiment": "negative", "sentiment_score": -0.62,
    "text_sentiment": "negative",
    "speaker": None, "driver": "VER", "team": "Red Bull Racing",
    "session": "Monaco Grand Prix", "year": 2024, "circuit": "Monaco",
    "clip_id": None, "filename": None,
    "prev_text": "Box this lap", "next_text": "Copy that",
    "laps": [{"number": 42, "duration_s": 74.8, "prev_s": 73.9, "next_s": 75.1}],
    "mentions": [{"kind": "Driver", "name": "LEC"}],
}

BARE = {"speech_id": "bare-1", "text": "Copy that", "mood": None, "sentiment": None,
        "sentiment_score": None, "text_sentiment": None, "speaker": None, "driver": None,
        "team": None, "session": None, "year": None, "circuit": None,
        "prev_text": None, "next_text": None, "laps": [], "mentions": []}


def test_every_block_leads_with_its_speech_id():
    """The citation contract: an id in the text is what lets a claim be traced back to
    source audio, and what the UI turns into a link."""
    out = render_context([FULL, BARE])
    assert out.startswith("[abc-123]")
    assert "[bare-1]" in out


def test_tone_is_reported_as_a_reading_not_a_fact():
    out = render_context([FULL])
    assert "voice reads stressed" in out
    # the assertion that must never appear: tone stated as something that was true
    assert "was stressed" not in out
    assert "is stressed" not in out


def test_voice_and_words_are_reported_separately():
    """A calm-sounding delivery of negative words is a signal. Both reads are shown so the
    reader sees the tension rather than a single averaged verdict."""
    row = {**FULL, "mood": "calm", "sentiment": "negative", "text_sentiment": "negative"}
    out = render_context([row])
    assert "voice reads calm" in out and "words read negative" in out


def test_text_read_is_surfaced_when_it_disagrees_with_the_fused_read():
    """The fusion can land somewhere the text alone did not. When those differ the text
    read is called out explicitly — otherwise the fused label silently overwrites it and
    the disagreement the pipeline went to the trouble of storing is lost here."""
    row = {**FULL, "mood": "calm", "sentiment": "neutral", "text_sentiment": "negative"}
    assert "text alone reads negative" in render_context([row])


def test_no_extra_text_line_when_the_reads_agree():
    row = {**FULL, "sentiment": "negative", "text_sentiment": "negative"}
    assert "text alone reads" not in render_context([row])


def test_agreeing_reads_are_not_repeated():
    """When the text read matches the fused read there is nothing extra to say, and
    repeating it would pad every block for no information."""
    assert render_context([FULL]).count("negative") == 1


def test_lap_time_carries_its_delta_to_the_previous_lap():
    """A bare lap time means nothing without a comparison; the delta is the whole point of
    lining speech up against laps."""
    out = render_context([FULL])
    assert "lap 42: 74.8s" in out and "+0.9s vs previous" in out


def test_lap_without_a_previous_lap_renders_without_a_delta():
    row = {**FULL, "laps": [{"number": 1, "duration_s": 80.0, "prev_s": None, "next_s": None}]}
    out = render_context([row])
    assert "lap 1: 80.0s" in out and "vs previous" not in out


def test_missing_neighbourhood_is_absent_not_invented():
    """Most of the corpus has no session, no lap and no enrolled speaker. Those blocks must
    render as short, not as a block full of None."""
    out = render_context([BARE])
    assert "None" not in out
    assert "unidentified speaker" in out
    assert "lap" not in out and "mentions" not in out


def test_unknown_speaker_is_named_as_unknown():
    """Speaker identification abstains by design; the renderer inherits that posture rather
    than quietly dropping the attribution line and letting the reader assume."""
    assert "unidentified speaker" in render_context([BARE])


def test_speaker_is_used_when_there_is_no_driver():
    row = {**BARE, "speaker": "Race Engineer"}
    out = render_context([row])
    assert "Race Engineer" in out and "unidentified speaker" not in out


def test_row_order_is_preserved():
    """Blocks come out in the order the anchor step ranked them."""
    out = render_context([FULL, BARE])
    assert out.index("[abc-123]") < out.index("[bare-1]")


def test_empty_input_renders_empty():
    assert render_context([]) == ""


@pytest.mark.parametrize("fraction", [0.05, 0.3, 0.5, 0.8, 0.95])
def test_truncation_never_emits_a_partial_block(fraction):
    """Truncating mid-block can drop the speech id, leaving a quote nothing can cite, or
    keep a quote while dropping the hedge on the tone line above it. Cutting on a block
    boundary loses a whole result instead, which is recoverable.

    Asserting that each surviving block is byte-identical to a block of the *untruncated*
    render is what makes this bite: a raw prefix slice still starts with '[', so merely
    checking the first character passes on exactly the implementation this forbids."""
    whole = render_context([FULL, BARE])
    intact = set(whole.split("\n\n"))

    out = render_context([FULL, BARE], max_chars=int(len(whole) * fraction))
    assert len(out) <= int(len(whole) * fraction)
    for block in filter(None, out.split("\n\n")):
        assert block in intact, f"partial block emitted: {block[:60]!r}"


def test_truncation_keeps_the_highest_ranked_blocks_first():
    full_len = len(render_context([FULL]))
    out = render_context([FULL, BARE], max_chars=full_len + 2)
    assert "[abc-123]" in out and "[bare-1]" not in out


# --- resolving citations without the graph -----------------------------------
#
# The graph is a derived read model rebuilt by an explicit sync, so it is routinely behind
# the corpus. Resolving a cited id only through it meant an answer whose citations went
# nowhere: the UI renders one card per resolved id, so an empty resolution is a chip with
# nothing to click. Measured at 1 of 8 ids on a live corpus before this existed.

def test_resolve_falls_back_to_postgres_for_ids_the_graph_has_not_seen(monkeypatch):
    from api.app.services import graph_context as gc

    async def fake_graph_run(*a, **kw):
        return [{"speech_id": "in-graph", "text": "from neo4j", "laps": [{"number": 3}]}]

    async def fake_fetch(sql, ids):
        if "utterances" in sql:
            return [{"speech_id": "in-pg", "text": "from postgres", "clip_id": "c1",
                     "start_s": 4.2, "mood": "stressed"}]
        return []

    monkeypatch.setattr(gc.graph, "run", fake_graph_run)
    monkeypatch.setattr(gc.db, "fetch", fake_fetch)

    rows = asyncio.run(gc.resolve(["in-graph", "in-pg"]))
    assert [r["speech_id"] for r in rows] == ["in-graph", "in-pg"], "anchor order not preserved"
    # The graph row keeps the edges only it has; the Postgres row carries what the UI needs
    # to make the citation clickable.
    assert rows[0]["laps"] == [{"number": 3}]
    assert rows[1]["clip_id"] == "c1" and rows[1]["start_s"] == 4.2
    assert rows[1]["laps"] == [] and rows[1]["mentions"] == []


def test_resolve_works_with_the_graph_switched_off(monkeypatch):
    """Citations must keep working when the graph is switched off. Citations are not an optional feature."""
    from api.app.services import graph_context as gc
    from common.graph import GraphUnavailable

    async def fake_graph_run(*a, **kw):
        raise GraphUnavailable("graph disabled")

    async def fake_fetch(sql, ids):
        return ([{"speech_id": "u1", "text": "hi", "clip_id": "c1", "start_s": 1.0}]
                if "utterances" in sql else [])

    monkeypatch.setattr(gc.graph, "run", fake_graph_run)
    monkeypatch.setattr(gc.db, "fetch", fake_fetch)

    rows = asyncio.run(gc.resolve(["u1"]))
    assert rows and rows[0]["clip_id"] == "c1"


def test_resolve_reads_radio_calls_as_well_as_utterances(monkeypatch):
    """Radio calls are most of the F1 corpus and never reach the utterances table unless
    they go through the full pipeline."""
    from api.app.services import graph_context as gc

    async def fake_graph_run(*a, **kw):
        return []

    async def fake_fetch(sql, ids):
        return ([] if "utterances" in sql
                else [{"speech_id": "r1", "text": "box box", "driver": "NOR"}])

    monkeypatch.setattr(gc.graph, "run", fake_graph_run)
    monkeypatch.setattr(gc.db, "fetch", fake_fetch)

    rows = asyncio.run(gc.resolve(["r1"]))
    assert rows and rows[0]["driver"] == "NOR"
