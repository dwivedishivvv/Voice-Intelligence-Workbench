"""Turn cleanup and word attribution: the changes that decide who a word is filed under.

Two of these guard optimizations (the phantom scan and word assignment both stopped being
all-pairs), so they assert equivalence against the implementations they replaced rather
than against hand-picked expectations — a faster function that quietly answers differently
is worse than the slow one it replaced.
"""
import numpy as np
import pytest

from worker.stages.diarize import drop_onset_phantoms, clean_turns
from worker.stages.reconcile import assign_words, regroup
from worker.stages.postprocess import count_interruptions_all


class Cfg:
    min_turn_s = 0.4
    merge_gap_s = 0.6
    vad_snap_tol_s = 0.15


def turn(start, end, label, **kw):
    return {"start": start, "end": end, "label": label, **kw}


# ---- drop_onset_phantoms: windowed scan must equal the all-pairs scan ----

def _phantoms_reference(turns, start_tol=0.3, dur_ratio=0.6):
    drop = set()
    for i, a in enumerate(turns):
        a_dur = a["end"] - a["start"]
        for j, b in enumerate(turns):
            if i == j or a["label"] == b["label"]:
                continue
            b_dur = b["end"] - b["start"]
            if b_dur <= a_dur:
                continue
            if abs(a["start"] - b["start"]) <= start_tol and a_dur < dur_ratio * b_dur:
                drop.add(i)
                break
    return [t for i, t in enumerate(turns) if i not in drop]


@pytest.mark.parametrize("seed", range(6))
def test_phantom_scan_matches_the_all_pairs_version(seed):
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.uniform(0, 30, 40))
    turns = [turn(float(s), float(s + rng.uniform(0.1, 3.0)), f"S{rng.integers(0, 3)}")
             for s in starts]
    assert drop_onset_phantoms(turns) == _phantoms_reference(turns)


def test_phantom_at_the_onset_of_a_longer_turn_is_dropped():
    turns = [turn(10.0, 10.2, "S1"), turn(10.05, 14.0, "S0")]
    assert [t["label"] for t in drop_onset_phantoms(turns)] == ["S0"]


# ---- clean_turns: short turns survive, flagged, instead of vanishing ----

def test_short_turns_are_kept_and_flagged_not_deleted():
    """A one-word interjection is real speech. Deleting it did not delete its words — it
    handed them to whoever was talking around it (see assign_words), which is how a second
    speaker's 'Copy' was attributed to the first."""
    # the two S0 turns are deliberately further apart than merge_gap_s, so this asserts
    # the short turn surviving rather than accidentally asserting the merge
    raw = [turn(0.0, 5.0, "S0"), turn(2.0, 2.2, "S1"), turn(7.0, 9.0, "S0")]
    out = clean_turns(raw, [], Cfg)
    assert [t["label"] for t in out] == ["S0", "S1", "S0"]
    short = [t for t in out if t["label"] == "S1"][0]
    assert short["too_short"] is True
    assert all(t["too_short"] is False for t in out if t["label"] == "S0")


def test_a_short_interjection_does_not_block_the_merge_around_it():
    """merge_adjacent walks consecutive entries and stops at a label change, so leaving the
    interjection in the list would silently keep the two S0 turns from joining."""
    raw = [turn(0.0, 2.0, "S0"), turn(2.1, 2.3, "S1"), turn(2.4, 5.0, "S0")]
    out = clean_turns(raw, [], Cfg)
    s0 = [t for t in out if t["label"] == "S0"]
    assert len(s0) == 1 and s0[0]["start"] == 0.0 and s0[0]["end"] == 5.0


def test_a_short_turn_never_marks_the_long_turn_it_interrupts_as_overlapped():
    """An 8s turn disqualified from fingerprinting by a 0.2s 'yeah' over the top of it
    would cost that speaker their best audio for the sake of a rounding error."""
    raw = [turn(0.0, 8.0, "S0"), turn(3.0, 3.2, "S1")]
    out = clean_turns(raw, [], Cfg)
    long_turn = [t for t in out if t["label"] == "S0"][0]
    short_turn = [t for t in out if t["label"] == "S1"][0]
    assert long_turn["is_overlap"] is False
    assert short_turn["is_overlap"] is True


# ---- assign_words: moving window must equal the restart-every-word scan ----

def _assign_reference(words, turns):
    for w in words:
        best, best_ov = None, 0.0
        for t in turns:
            if t["start"] >= w["end"]:
                break
            ov = min(w["end"], t["end"]) - max(w["start"], t["start"])
            if ov > best_ov:
                best, best_ov = t["label"], ov
        dur = max(w["end"] - w["start"], 1e-6)
        w["ref_label"] = best
        w["ref_conf"] = 0.0 if best is None else min(1.0, best_ov / dur)


@pytest.mark.parametrize("seed", range(6))
def test_word_assignment_matches_the_rescanning_version(seed):
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.uniform(0, 30, 25))
    turns = [turn(float(s), float(s + rng.uniform(0.3, 2.0)), f"S{rng.integers(0, 3)}",
                  is_overlap=False)
             for s in starts]
    wstarts = np.sort(rng.uniform(0, 32, 200))
    words = [{"word": "x", "start": float(s), "end": float(s + 0.25), "probability": 0.9}
             for s in wstarts]

    _assign_reference(words, turns)
    assign_words(words, turns)
    for w in words:
        if w["ref_label"] is not None:          # the fallback path is nearest-turn, untouched
            assert w["local_label"] == w["ref_label"]
            assert w["speaker_conf"] == pytest.approx(w["ref_conf"])


def test_words_over_cross_talk_are_marked_and_reach_the_utterance():
    """The overlap flag was computed per turn, used to keep overlapped audio out of
    fingerprints, then dropped — nothing downstream could tell an attribution made over
    clean speech from one made while two people talked at once."""
    turns = [turn(0.0, 2.0, "S0", is_overlap=True), turn(2.0, 4.0, "S0", is_overlap=False)]
    words = [{"word": "a", "start": 0.1, "end": 0.4, "probability": 0.9},
             {"word": "b", "start": 1.0, "end": 1.3, "probability": 0.9},
             {"word": "c", "start": 2.5, "end": 2.8, "probability": 0.9},
             {"word": "d", "start": 3.0, "end": 3.3, "probability": 0.9}]
    assign_words(words, turns)
    assert [w["in_overlap"] for w in words] == [True, True, False, False]

    utts = regroup(words)
    assert len(utts) == 1
    assert utts[0]["overlap_ratio"] == pytest.approx(0.5)


def test_clean_speech_reports_no_overlap():
    turns = [turn(0.0, 4.0, "S0", is_overlap=False)]
    words = [{"word": "a", "start": 0.1, "end": 0.4, "probability": 0.9}]
    assign_words(words, turns)
    assert regroup(words)[0]["overlap_ratio"] == 0.0


# ---- interruptions: one sweep must equal the all-pairs count ----

def _interruptions_reference(label, turns):
    n = 0
    for t in turns:
        if t["label"] != label:
            continue
        for o in turns:
            if o["label"] != label and o["start"] < t["start"] < o["end"]:
                n += 1
                break
    return n


@pytest.mark.parametrize("seed", range(6))
def test_interruption_sweep_matches_the_all_pairs_count(seed):
    rng = np.random.default_rng(seed)
    starts = np.sort(rng.uniform(0, 30, 40))
    turns = [turn(float(s), float(s + rng.uniform(0.2, 3.0)), f"S{rng.integers(0, 3)}")
             for s in starts]
    got = count_interruptions_all(turns)
    for label in {t["label"] for t in turns}:
        assert got[label] == _interruptions_reference(label, turns), label


def test_turns_sharing_a_start_do_not_interrupt_each_other():
    """snap_to_vad can pull two turns onto the same VAD edge; the all-pairs version this
    replaced required a strictly earlier start, so neither counted."""
    turns = [turn(1.0, 3.0, "S0"), turn(1.0, 2.0, "S1")]
    assert count_interruptions_all(turns) == {"S0": 0, "S1": 0}
