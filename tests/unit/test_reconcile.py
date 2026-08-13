from worker.stages.reconcile import assign_words, smooth, regroup


def mk(word, label_hint_start, label_hint_end, prob=0.9):
    return {"word": word, "start": label_hint_start, "end": label_hint_end, "probability": prob}


def test_word_assignment_at_turn_boundary():
    words = [{"word": "hello", "start": 2.9, "end": 3.3, "probability": 0.9}]
    turns = [{"start": 0, "end": 3.0, "label": "A"}, {"start": 3.0, "end": 6, "label": "B"}]
    assign_words(words, turns)
    assert words[0]["local_label"] == "B"          # 0.3s overlap with B vs 0.1s with A
    assert 0.7 < words[0]["speaker_conf"] <= 1.0


def test_smoothing_fixes_single_stray_word():
    words = [mk("a", 0, 1), mk("b", 1, 2), mk("c", 2, 3)]
    words[0]["local_label"], words[0]["speaker_conf"] = "A", 0.9
    words[1]["local_label"], words[1]["speaker_conf"] = "B", 0.3
    words[2]["local_label"], words[2]["speaker_conf"] = "A", 0.9
    smooth(words, min_conf=0.6)
    assert words[1]["local_label"] == "A"
    assert words[1]["smoothed"]


def test_smoothing_respects_confident_disagreement():
    words = [mk("a", 0, 1), mk("b", 1, 2), mk("c", 2, 3)]
    words[0]["local_label"], words[0]["speaker_conf"] = "A", 0.9
    words[1]["local_label"], words[1]["speaker_conf"] = "B", 0.95
    words[2]["local_label"], words[2]["speaker_conf"] = "A", 0.9
    smooth(words, min_conf=0.6)
    assert words[1]["local_label"] == "B"


def test_regroup_splits_on_speaker_change():
    words = [mk("hi.", 0, 0.5), mk("bye.", 0.6, 1.0)]
    words[0]["local_label"], words[0]["speaker_conf"] = "A", 1.0
    words[1]["local_label"], words[1]["speaker_conf"] = "B", 1.0
    utts = regroup(words)
    assert len(utts) == 2
    assert utts[0]["local_label"] == "A" and utts[1]["local_label"] == "B"


if __name__ == "__main__":
    test_word_assignment_at_turn_boundary()
    test_smoothing_fixes_single_stray_word()
    test_smoothing_respects_confident_disagreement()
    test_regroup_splits_on_speaker_change()
    print("ok")
