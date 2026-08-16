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


# --- the LLM pass's contract ---------------------------------------------------
#
# The model is asked to label, never to invent. Everything below is a way of not trusting
# its reply: a label outside the vocabulary, an index pointing past the batch, a confidence
# below the bar and "none" all have to drop the row rather than be coerced into an edge.

def test_prompt_offers_only_the_real_vocabulary():
    """A prompt that drifts from the taxonomy produces labels the database discards
    silently — the model looks like it answered and nothing is stored."""
    prompt = ontrack.build_prompt(["anything"])
    for e in ontrack.EVENT_TYPES:
        assert e.name in prompt


def test_prompt_numbers_lines_so_replies_match_by_position():
    prompt = ontrack.build_prompt(["first line", "second line"])
    assert "1. first line" in prompt and "2. second line" in prompt


def test_prompt_teaches_the_failure_cases_measured_on_this_corpus():
    """These three are not hypothetical: the embedding pass read "Head down." as a
    completed overtake, "the car is snappy" as dirty air, and a sentence about pitting as
    a pass. The prompt names them because they are what the model is there to catch."""
    prompt = ontrack.build_prompt(["x"]).lower()
    assert "head down" in prompt and "snappy" in prompt
    assert "none" in prompt


def test_a_confirmed_verdict_is_read():
    out = ontrack.parse_verdicts(
        '[{"n": 1, "event": "contact", "other_car": "Norris", "confidence": 0.9, "why": "hit"}]',
        batch_size=1, min_confidence=0.6)
    assert out == [ontrack.Verdict(1, "contact", 0.9, "Norris", "hit")]


def test_none_is_a_real_answer_not_a_row():
    assert ontrack.parse_verdicts(
        '[{"n": 1, "event": "none", "confidence": 0.9}]', 1, 0.6) == []


def test_an_invented_event_type_is_discarded():
    """A model asked for a closed vocabulary will still occasionally return something
    plausible-sounding from outside it. Storing that would put a type in the graph that no
    query knows and no renderer can word."""
    assert ontrack.parse_verdicts(
        '[{"n": 1, "event": "wheel_to_wheel_battle", "confidence": 0.99}]', 1, 0.6) == []


def test_an_index_outside_the_batch_is_discarded():
    """Verdicts are matched to lines by position. An index past the end would attach a
    label to the wrong utterance — or crash on the lookup, which is the better of two bad
    outcomes and still not one to allow."""
    assert ontrack.parse_verdicts('[{"n": 7, "event": "spin", "confidence": 0.9}]', 3, 0.6) == []
    assert ontrack.parse_verdicts('[{"n": 0, "event": "spin", "confidence": 0.9}]', 3, 0.6) == []


def test_low_confidence_is_discarded():
    assert ontrack.parse_verdicts(
        '[{"n": 1, "event": "spin", "confidence": 0.3}]', 1, 0.6) == []


def test_prose_around_the_json_is_tolerated():
    """Models wrap JSON in explanation and fences whatever the instruction says."""
    out = ontrack.parse_verdicts(
        'Sure! Here are the labels:\n```json\n'
        '[{"n": 1, "event": "traffic", "confidence": 0.8}]\n```\nHope that helps.',
        1, 0.6)
    assert len(out) == 1 and out[0].event_type == "traffic"


@pytest.mark.parametrize("reply", ["", "no json here", "[", '{"n": 1}', "[1, 2, 3]"])
def test_an_unreadable_reply_labels_nothing(reply):
    """An unusable reply leaves the batch unlabelled, which is the same outcome as the
    model saying "none" — no edge, rather than a guess."""
    assert ontrack.parse_verdicts(reply, 5, 0.6) == []
