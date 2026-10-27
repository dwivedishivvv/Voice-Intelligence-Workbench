"""render_context turns a graph expansion into the text a person or a model actually
reads. It is the last step before a claim gets made about someone's audio, so the things
asserted here are mostly about honesty rather than formatting:

  - every block carries its speech id, or a downstream claim cannot be cited back to the
    recording it came from;
  - a missing neighbourhood renders as absence, not as a guess;
  - the recording and offset are named, so a quote can be found in the audio.

Pure function, so none of this needs Postgres or Neo4j.
"""
import asyncio

import pytest

from api.app.services.graph_context import render_context

FULL = {
    "speech_id": "abc-123",
    "text": "Reliance led the round and Dunzo needed their sign-off",
    "speaker": "Marta", "clip_id": "c1", "filename": "board_call.wav", "start_s": 75.0,
    "prev_text": "What happened with the funding?", "prev_speaker": "Dan",
    "next_text": "That slowed everything down", "next_speaker": None,
    "mentions": [{"kind": "name", "name": "Reliance"}, {"kind": "person", "name": "Kabir Biswas"}],
    "topics": ["quick commerce"],
}

BARE = {"speech_id": "bare-1", "text": "Copy that", "speaker": None, "clip_id": None,
        "filename": None, "start_s": None, "prev_text": None, "next_text": None,
        "mentions": [], "topics": []}


def test_every_block_leads_with_its_speech_id():
    """The citation contract: an id in the text is what lets a claim be traced back to
    source audio, and what the UI turns into a link."""
    out = render_context([FULL, BARE])
    assert out.startswith("[abc-123]")
    assert "[bare-1]" in out


def test_missing_neighbourhood_is_absent_not_invented():
    """Most of the corpus has no enrolled speaker and names nothing. Those blocks must
    render as short, not as a block full of None."""
    out = render_context([BARE])
    assert "None" not in out
    assert "unidentified speaker" in out
    assert "names:" not in out and "topics" not in out


def test_unknown_speaker_is_named_as_unknown():
    """Speaker identification abstains by design; the renderer inherits that posture rather
    than quietly dropping the attribution line and letting the reader assume."""
    assert "unidentified speaker" in render_context([BARE])


def test_speaker_is_named_when_known():
    row = {**BARE, "speaker": "Marta"}
    out = render_context([row])
    assert "Marta" in out and "unidentified speaker" not in out


def test_context_carries_names_topics_and_who_spoke_either_side():
    out = render_context([FULL])
    assert "names: Reliance, Kabir Biswas (person)" in out
    assert "recording topics: quick commerce" in out
    assert 'before (Dan): "What happened with the funding?"' in out
    assert 'after: "That slowed everything down"' in out


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
        return [{"speech_id": "in-graph", "text": "from neo4j",
                 "mentions": [{"kind": "name", "name": "Dunzo"}]}]

    async def fake_fetch(sql, ids):
        if "utterances" in sql:
            return [{"speech_id": "in-pg", "text": "from postgres", "clip_id": "c1",
                     "start_s": 4.2}]
        return []

    monkeypatch.setattr(gc.graph, "run", fake_graph_run)
    monkeypatch.setattr(gc.db, "fetch", fake_fetch)

    rows = asyncio.run(gc.resolve(["in-graph", "in-pg"]))
    assert [r["speech_id"] for r in rows] == ["in-graph", "in-pg"], "anchor order not preserved"
    # The graph row keeps the edges only it has; the Postgres row carries what the UI needs
    # to make the citation clickable.
    assert rows[0]["mentions"] == [{"kind": "name", "name": "Dunzo"}]
    assert rows[1]["clip_id"] == "c1" and rows[1]["start_s"] == 4.2
    assert rows[1]["mentions"] == [] and rows[1]["topics"] == []


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


# --- uploaded clips: which recording, and the part that matched -------------------
#
# Most of a corpus is uploaded recordings with usually no enrolled speaker. Every hit from one used to render as "unidentified speaker: <first 160
# characters>", which named neither the recording nor, in long podcast turns, the words
# that matched.

LONG = ("In 2014 Zomato was still a restaurant listing app and Swiggy was still figuring out "
        "the first version of its app. Kabir Biswas was 30 years old at this time and had "
        "seen little success with his first startup Hopper.")


def test_brief_names_the_clip_and_offset():
    from api.app.services.graph_context import render_brief
    row = {**BARE, "filename": "meeting_100s.wav", "start_s": 62.4}
    assert 'clip "meeting_100s.wav" at 1:02' in render_brief([row])


def test_context_block_names_the_clip():
    out = render_context([{**BARE, "filename": "meeting_100s.wav", "start_s": 9.0}])
    assert out.startswith('[bare-1] clip "meeting_100s.wav" at 0:09')


def test_long_quote_is_windowed_on_the_query_term():
    """Observed: the search for "Kabir" returned this line clipped before the surname, and
    the model supplied a surname of its own."""
    from api.app.services.graph_context import render_brief
    out = render_brief([{**BARE, "text": LONG}], max_text=120, terms=["kabir"])
    assert "Kabir Biswas" in out
    quote = out.split(': "', 1)[1].rstrip('"')
    assert len(quote) <= 120 and quote.startswith("…")


def test_long_quote_without_terms_is_cut_from_the_start_as_before():
    from api.app.services.graph_context import render_brief
    out = render_brief([{**BARE, "text": LONG}], max_text=60)
    quote = out.split(': "', 1)[1].rstrip('"')
    assert quote.startswith("In 2014") and quote.endswith("…") and len(quote) <= 60


def test_short_quote_is_untouched_by_windowing():
    from api.app.services.graph_context import render_brief
    out = render_brief([{**BARE, "text": "Copy that"}], terms=["copy"])
    assert out.endswith(': "Copy that"')
