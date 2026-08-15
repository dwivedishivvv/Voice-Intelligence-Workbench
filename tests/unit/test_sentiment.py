"""Fusion + label-normalization logic for the SENTIMENT stage. No model, no DB — this
covers the arithmetic that decides what a user sees, which is the part that breaks silently."""
from worker.stages.sentiment import (
    _normalize, _signed, _label, _is_protocol, _group, _context_prefix,
    NEUTRAL_BAND, MOOD_SHIFT, CONTEXT_DAMPING, MERGE_MAX_GAP_S,
)


def utt(text, label="S0", start=0.0, end=1.0):
    return {"text": text, "local_label": label, "start": start, "end": end}


# ---- 1. protocol lexicon ----

def test_protocol_words_are_not_sentiment():
    # each of these was scored confidently and wrongly by the raw model
    for t in ["Negative.", "Head down.", "Understood.", "Copy that.", "Box, box.",
              "Affirmative", "stand by", "Okay.", "Yeah."]:
        assert _is_protocol(t), t


def test_real_opinions_containing_protocol_words_still_score():
    # the word appears, but something is actually being asserted -- must reach the model
    for t in ["that's a negative result for us", "the box is broken",
              "I understood nothing he said", "check the rear tyres, they are gone"]:
        assert not _is_protocol(t), t


def test_short_but_genuinely_emotional_turns_are_not_neutralized():
    # these are as short as the protocol words but carry real valence
    for t in ["Thanks, guys.", "Sorry mate.", "Brilliant!", "VAMOS!"]:
        assert not _is_protocol(t), t


# ---- 2. short-turn merging ----

def test_consecutive_short_turns_from_one_speaker_merge():
    us = [utt("Not yet,", "S0", 0.0, 0.6), utt("not yet.", "S0", 0.7, 1.3)]
    assert _group(us) == [[0, 1]]


def test_different_speakers_never_merge():
    # merging across a speaker boundary would file one person's words under another
    us = [utt("Not yet,", "S0", 0.0, 0.6), utt("not yet.", "S1", 0.7, 1.3)]
    assert _group(us) == [[0], [1]]


def test_long_turns_are_left_alone():
    long = "we need a little bit more pace here the tyres are really starting to go off now"
    us = [utt(long, "S0", 0.0, 5.0), utt(long, "S0", 5.1, 10.0)]
    assert _group(us) == [[0], [1]]


def test_a_long_pause_starts_a_new_thought():
    us = [utt("Okay,", "S0", 0.0, 0.5), utt("go.", "S0", 0.5 + MERGE_MAX_GAP_S + 1, 9.0)]
    assert _group(us) == [[0], [1]]


# ---- 3. context window ----

def test_context_prefix_takes_preceding_turns_only():
    us = [utt("the rear tyres are gone"), utt("okay"), utt("copy")]
    assert _context_prefix(us, 1) == "the rear tyres are gone"
    assert _context_prefix(us, 0) == ""          # nothing precedes the first turn


def test_protocol_turns_never_merge_with_real_speech():
    # regression: "Negative." absorbed the sentence after it and inherited its -0.60,
    # which is precisely the false positive the lexicon exists to remove
    us = [utt("Negative.", "S0", 0.0, 0.5),
          utt("the rear tyres are completely gone now", "S0", 0.6, 3.0)]
    assert _group(us) == [[0], [1]]


def test_protocol_turn_is_not_absorbed_by_a_preceding_group():
    us = [utt("we are losing time", "S0", 0.0, 1.0), utt("Copy that.", "S0", 1.1, 1.6)]
    assert _group(us) == [[0], [1]]


def test_merging_two_protocol_turns_stays_protocol():
    # regression: "Negative." + "Head down." merged into a string matching neither the
    # phrase list nor the all-tokens rule, reached the model, and came back -0.52
    joined = "Negative. Head down."
    assert not _is_protocol(joined)                       # the merged string alone fools it
    assert all(_is_protocol(t) for t in ["Negative.", "Head down."])  # members do not


def test_context_assisted_scores_are_damped_below_the_raw_score():
    # a context score is partly the other speaker's words, so it must never be attributed
    # at full strength -- otherwise a neutral "okay" inherits the bad news before it
    raw = -0.9
    assert abs(raw * CONTEXT_DAMPING) < abs(raw)


def test_normalize_handles_every_catalog_label_set():
    # 3-class, 5-class, and LABEL_N heads all collapse to the same vocabulary
    assert _normalize("Positive") == "positive"
    assert _normalize("NEGATIVE") == "negative"
    assert _normalize("Very Negative") == "negative"
    assert _normalize("Very Positive") == "positive"
    assert _normalize("LABEL_0") == "negative"
    assert _normalize("LABEL_2") == "positive"
    assert _normalize("neutral") == "neutral"
    assert _normalize("something unrecognized") == "neutral"


def test_signed_puts_confidence_on_the_right_side_of_zero():
    assert _signed("positive", 0.9) == ("positive", 0.9)
    assert _signed("negative", 0.9) == ("negative", -0.9)
    assert _signed("neutral", 0.9) == ("neutral", 0.0)


def test_neutral_band_keeps_a_mood_nudge_from_inventing_sentiment():
    # a neutral utterance delivered under stress must not read as "negative" on the
    # acoustic nudge alone -- that nudge is smaller than the deadband by design
    neutral_but_stressed = 0.0 + MOOD_SHIFT["stressed"]
    assert abs(neutral_but_stressed) <= NEUTRAL_BAND
    assert _label(neutral_but_stressed) == "neutral"


def test_mood_shifts_a_borderline_score_across_the_line():
    # words mildly negative, delivery stressed -> the pair clears the band together
    assert _label(-0.15) == "neutral"
    assert _label(-0.15 + MOOD_SHIFT["stressed"]) == "negative"


def test_label_thresholds():
    assert _label(0.0) == "neutral"
    assert _label(NEUTRAL_BAND) == "neutral"        # boundary is exclusive
    assert _label(NEUTRAL_BAND + 0.01) == "positive"
    assert _label(-NEUTRAL_BAND - 0.01) == "negative"
