"""Fusion + label-normalization logic for the SENTIMENT stage. No model, no DB — this
covers the arithmetic that decides what a user sees, which is the part that breaks silently."""
from worker.stages.sentiment import _normalize, _signed, _label, NEUTRAL_BAND, MOOD_SHIFT


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
