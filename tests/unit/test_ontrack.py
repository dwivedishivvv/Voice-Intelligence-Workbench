"""On-track event classification.

The scoring is a few lines, and every one of them exists to refuse. That is the part worth
asserting: an event becomes a graph edge, and a model reading the graph will state it, so a
wrong edge is worse than a missing one and each abstention rule is a promise about what
does not get written.

Pure — no database, no model. The similarities are supplied directly, which is also how
the failure cases stay legible: each test names the shape of confusion it forbids.
"""
import pytest

from common import ontrack


# --- the taxonomy -------------------------------------------------------------

def test_every_event_type_is_in_a_known_family():
    for e in ontrack.EVENT_TYPES:
        assert e.family in ontrack.FAMILIES, e.name


def test_event_names_are_unique():
    names = [e.name for e in ontrack.EVENT_TYPES]
    assert len(names) == len(set(names))


def test_every_type_carries_exemplars_and_a_reading():
    """An exemplar set is the classifier for that type — an empty one is a type that can
    never fire, and a missing `reads_as` is an event nothing can word honestly."""
    for e in ontrack.EVENT_TYPES:
        assert len(e.exemplars) >= 4, f"{e.name}: too few exemplars to anchor a meaning"
        assert e.reads_as and not e.reads_as.endswith("."), e.name


def test_the_none_pseudo_type_is_not_an_event_type():
    """It exists to be beaten, never to be stored. A NONE that leaked into EVENT_TYPES
    would be projected into the graph as a real edge."""
    assert ontrack.NONE not in ontrack.BY_NAME
    assert len(ontrack.NONE_ANCHORS) >= 10


# --- refusing ------------------------------------------------------------------

def test_a_clear_winner_is_read():
    reading = ontrack.classify(
        {"overtake_attempt": 0.80, "closing": 0.40, ontrack.NONE: 0.30},
        floor=0.60, min_margin=0.05)
    assert reading is not None
    assert reading.event_type == "overtake_attempt"
    assert reading.family == ontrack.OVERTAKING
    assert reading.score == pytest.approx(0.80)


def test_nothing_clearing_the_floor_reads_as_nothing():
    """Most radio is not about anything on track. Silence is the common answer."""
    assert ontrack.classify(
        {"overtake_attempt": 0.55, "closing": 0.20}, floor=0.60, min_margin=0.05) is None


def test_two_event_types_a_hair_apart_read_as_nothing():
    """"he's right behind me" is defending and closing at once. Breaking that tie by the
    larger float manufactures a distinction the sentence does not carry — the same reason
    speaker identification abstains on a thin margin instead of picking the top score."""
    assert ontrack.classify(
        {"defending": 0.81, "closing": 0.80}, floor=0.60, min_margin=0.05) is None


def test_ordinary_radio_beats_a_weak_event():
    """The none-anchors are the third test and the one that does most of the work: without
    them the nearest of eighteen event types always wins, because "closest" says nothing
    about whether any of them fit. On this corpus that read "this car is so hot to drive,
    it's snappy like crazy" as dirty air."""
    assert ontrack.classify(
        {"dirty_air": 0.68, ontrack.NONE: 0.66}, floor=0.60, min_margin=0.05) is None


def test_a_strong_event_still_wins_against_ordinary_radio():
    """The anchors are competition, not a veto — a line that really is about a pass must
    still get through."""
    reading = ontrack.classify(
        {"overtake_attempt": 0.90, ontrack.NONE: 0.50}, floor=0.60, min_margin=0.05)
    assert reading and reading.event_type == "overtake_attempt"


def test_empty_scores_read_as_nothing():
    assert ontrack.classify({}, floor=0.6, min_margin=0.05) is None
    assert ontrack.classify({ontrack.NONE: 0.9}, floor=0.6, min_margin=0.05) is None


def test_margin_is_measured_against_the_stronger_of_runner_up_and_none():
    """Taking the runner-up alone would let a line that is 0.61 event / 0.60 ordinary
    through on a wide margin over the *third* type, which is the wrong comparison."""
    assert ontrack.classify(
        {"contact": 0.66, "spin": 0.20, ontrack.NONE: 0.64},
        floor=0.60, min_margin=0.05) is None


# --- the length guard ----------------------------------------------------------

@pytest.mark.parametrize("text", ["Head down.", "Done, Ali.", "Difficult, Oscar.", "", None])
def test_fragments_are_too_short_to_read(text):
    """Measured, not assumed: each of these was stored as a confident on-track event at a
    0.60 floor — "Head down." as a completed overtake at 0.74. A sentence embedding of two
    words encodes the words, so its nearest neighbour among eighteen types is arbitrary."""
    assert ontrack.too_short(text, 6) is True


def test_a_full_sentence_is_long_enough_to_read():
    assert ontrack.too_short("can you get closer to the car ahead", 6) is False


# --- wording -------------------------------------------------------------------

def test_events_are_worded_as_readings_never_as_facts():
    """The same rule tone is held to. A classifier over one line of ASR output from a
    compressed radio channel cannot report what happened on track, and the wording is
    where that distinction survives into the graph and the answer."""
    for e in ontrack.EVENT_TYPES:
        text = ontrack.phrase(ontrack.Reading(e.name, e.family, 0.7, 0.1))
        assert text.startswith("reads as ")
        assert " was " not in text and " did " not in text
