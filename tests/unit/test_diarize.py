from worker.stages.diarize import snap_to_vad, merge_adjacent, annotate_overlaps


def test_snap_to_vad_within_tolerance():
    turns = [{"start": 1.02, "end": 3.48, "label": "S0"}]
    vad = [(1.00, 3.50)]
    out = snap_to_vad(turns, vad, tol=0.15)
    assert out[0]["start"] == 1.00 and out[0]["end"] == 3.50
    assert out[0]["snapped"]


def test_merge_adjacent_same_speaker():
    turns = [{"start": 0, "end": 1, "label": "A"}, {"start": 1.2, "end": 2, "label": "A"}]
    out = merge_adjacent(turns, max_gap=0.6)
    assert len(out) == 1 and out[0]["end"] == 2


def test_overlap_annotation():
    turns = [{"start": 0, "end": 2, "label": "A"}, {"start": 1.5, "end": 3, "label": "B"}]
    out = annotate_overlaps(turns)
    assert out[0]["is_overlap"] and out[1]["is_overlap"]


if __name__ == "__main__":
    test_snap_to_vad_within_tolerance()
    test_merge_adjacent_same_speaker()
    test_overlap_annotation()
    print("ok")
