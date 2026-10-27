"""Auto tags are read by people filtering a library, so a wrong one is a quiet lie about a
recording. These pin the abstain-unless-clear rules (common/tagging.py) and the topic
selection (worker/stages/tag.py) without a database or a model."""
from types import SimpleNamespace

import numpy as np

from common import tagging
from worker.stages import tag

CFG = SimpleNamespace(overlap_warn_ratio=0.30)


def _u(start, end, text="hello there", conf=0.9, overlap=0.0):
    return {"start_s": start, "end_s": end, "text": text, "mean_word_conf": conf,
            "overlap_ratio": overlap}


def _tags(clip=None, quality=None, speakers=(), utterances=()):
    base = {"language": "en", "language_conf": 0.9, "n_speakers": 2, "duration_s": 60.0}
    return tagging.derive_tags({**base, **(clip or {})}, quality, list(speakers),
                               list(utterances), CFG)


def test_only_confident_matches_name_a_speaker():
    tags = _tags(speakers=[{"match_result": "confident", "display_name": "Ana"},
                           {"match_result": "suggested", "display_name": "Ben"},
                           {"match_result": "unknown", "display_name": None}])
    assert "speaker:Ana" in tags
    assert "speaker:Ben" not in tags


def test_language_needs_confidence():
    assert "lang:en" in _tags()
    assert not any(t.startswith("lang:") for t in _tags(clip={"language_conf": 0.3}))


def test_voice_count_families():
    assert "voices:solo" in _tags(clip={"n_speakers": 1})
    assert "voices:dialogue" in _tags(clip={"n_speakers": 2})
    assert "voices:group" in _tags(clip={"n_speakers": 4})


def test_no_tone_or_sentiment_tags_are_ever_derived():
    """Both features were removed; a stale column on an old database must not bring them back."""
    tags = _tags(clip={"sentiment": "negative", "mood": "stressed"},
                 utterances=[{**_u(0, 5), "mood": "stressed", "text_sentiment": "positive"}])
    assert not any(t.split(":")[0] in ("tone", "sentiment") or t == "mixed-signals" for t in tags)


def test_caveat_tags():
    noisy = [_u(0, 10, conf=0.4, overlap=0.5)]
    tags = _tags(quality={"grade": "poor"}, utterances=noisy)
    assert {"audio:poor", "asr:low-confidence", "crosstalk"} <= set(tags)
    assert not {"asr:low-confidence", "crosstalk"} & set(_tags(utterances=[_u(0, 10)]))


def test_candidate_phrases_skip_stopword_edges_and_short_words():
    phrases = tag.candidate_phrases(["we need to check the tyre temperatures on the car",
                                     "tyre temperatures look good"])
    assert "tyre temperatures" in phrases
    assert phrases.index("tyre temperatures") < phrases.index("tyre")  # recurring bigram ranks high
    assert not any(p.split()[0] in tag.STOPWORDS or p.split()[-1] in tag.STOPWORDS for p in phrases)
    assert "car" not in phrases  # single words need four letters


def test_trigrams_keep_connectors_but_not_verbs_in_the_middle():
    phrases = tag.candidate_phrases(["the point of view matters", "the recording is estimated"])
    assert "point of view" in phrases
    assert "recording is estimated" not in phrases


def test_long_transcripts_need_recurring_phrases():
    text = ["alpha bravo"] * 2 + ["filler words here"] * 60 + ["zebra crossing once"]
    phrases = tag.candidate_phrases(text)
    assert "alpha bravo" in phrases
    assert "zebra crossing" not in phrases


def test_pick_topics_drops_nested_phrases_and_respects_floor():
    doc = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    phrases = ["tyre wear", "tyre", "fuel saving", "weather"]
    vecs = np.array([[0.9, 0.43, 0.0], [0.88, 0.47, 0.0], [0.6, 0.0, 0.8], [0.1, 0.99, 0.0]],
                    dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    picked = tag.pick_topics(doc, vecs, phrases, k=3, min_sim=0.3)
    assert picked[0] == "tyre wear"
    assert "tyre" not in picked          # subset of an already-picked phrase
    assert "weather" not in picked       # under the similarity floor
    assert "fuel saving" in picked


def test_topics_only_for_english():
    cfg = SimpleNamespace(auto_tag_max_topics=3, auto_tag_min_topic_sim=0.3)
    assert tag.topic_tags(None, [_u(0, 10, text="word " * 40)], "de", cfg) == []
