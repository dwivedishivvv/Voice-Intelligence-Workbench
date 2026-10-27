"""Re-clustering pyannote's turns with the speaker embedder (worker/stages/recluster.py).
Pure helpers only — the clustering itself is measured on real audio, not unit-tested."""
from types import SimpleNamespace

import numpy as np

from worker.stages import recluster as rc
from worker.stages.diarize import tidy_relabelled

CFG = SimpleNamespace(merge_gap_s=0.6)


def test_pieces_cover_the_turn_with_no_unembeddable_remainder():
    ps = rc.pieces(0.0, 6.2)
    assert ps[0] == (0.0, 2.5) and ps[-1][1] == 6.2
    assert all(e - s >= rc.MIN_PIECE_S for s, e in ps)
    assert all(e - s <= rc.PIECE_S + rc.MIN_PIECE_S for s, e in ps)
    assert rc.pieces(0.0, 0.8) == []                 # too short to embed at all


def test_a_cluster_with_a_few_seconds_of_speech_folds_into_its_nearest_neighbour():
    """Two seconds is not evidence of an extra person; it joins whoever it sounds closest to."""
    E = np.array([[1, 0], [1, 0], [0, 1], [0.9, 0.1]], dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    lab = np.array([1, 1, 2, 3])
    durs = np.array([2.5, 2.5, 2.5, 1.0])
    out = rc.fold_small(E, lab, durs, min_s=3.0)
    assert out[3] == 1                                # 1s cluster -> the near one
    assert len(set(out.tolist())) <= 2


def test_fold_never_removes_the_last_cluster():
    E = np.array([[1, 0]], dtype=np.float32)
    assert rc.fold_small(E, np.array([5]), np.array([1.0]), min_s=3.0).tolist() == [5]


def _t(start, end, label, too_short=False, overlap=False):
    return {"start": start, "end": end, "label": label, "too_short": too_short,
            "is_overlap": overlap, "snapped": False}


def test_turns_that_now_share_a_speaker_merge_and_lose_their_overlap_flag():
    turns = [_t(0, 4, "SPEAKER_00", overlap=True), _t(3.8, 7, "SPEAKER_00", overlap=True),
             _t(9, 12, "SPEAKER_01")]
    out = tidy_relabelled(turns, CFG)
    assert [(t["start"], t["end"], t["label"]) for t in out] == [(0, 7, "SPEAKER_00"), (9, 12, "SPEAKER_01")]
    assert not any(t["is_overlap"] for t in out)
    assert [t["idx"] for t in out] == [0, 1]


def test_short_interjections_keep_their_own_label_and_are_reflagged():
    turns = [_t(0, 6, "SPEAKER_00"), _t(2.0, 2.3, "SPEAKER_01", too_short=True)]
    out = tidy_relabelled(turns, CFG)
    short = next(t for t in out if t["too_short"])
    assert short["label"] == "SPEAKER_01" and short["is_overlap"]
