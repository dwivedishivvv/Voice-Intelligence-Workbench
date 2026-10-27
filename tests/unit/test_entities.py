"""The entity extractor turns capitalised words in ASR text into graph nodes a model and a
reader both treat as "things the corpus talks about". It is tuned to abstain, and these
pin the cases where it must."""
from common import entities


def test_a_name_must_recur_to_count():
    assert entities.extract(["We met Reliance today."], min_mentions=2) == {}
    found = entities.extract(["We met Reliance today.", "Reliance said no."], min_mentions=2)
    assert found == {"Reliance": {"kind": "name", "n": 2}}


def test_sentence_initial_words_alone_are_not_names():
    """"So", "Today", "Then" open sentences; a capital letter there means nothing."""
    texts = ["Today we shipped.", "Today was long.", "Then it broke.", "Then we fixed it."]
    assert entities.extract(texts, min_mentions=2) == {}


def test_a_capitalised_verb_opening_a_sentence_does_not_glue_onto_the_name():
    texts = ["See Dunzo grew fast.", "And then Dunzo shut down.", "We liked Dunzo."]
    found = entities.extract(texts, min_mentions=2)
    assert "Dunzo" in found and "See Dunzo" not in found
    assert entities.mentions("See Dunzo grew fast.", {"dunzo": "Dunzo"}) == ["Dunzo"]


def test_multi_word_names_and_possessives():
    texts = ["We spoke to Kabir Biswas about it.", "Then Kabir Biswas's team left."]
    found = entities.extract(texts, min_mentions=2)
    assert found["Kabir Biswas"]["kind"] == "person"


def test_organisation_suffix_and_connectors():
    texts = ["Money came from Goodwater Capital and Bank of America.",
             "Later Goodwater Capital and Bank of America pulled out."]
    found = entities.extract(texts, min_mentions=2)
    assert found["Goodwater Capital"]["kind"] == "organisation"
    assert "Bank of America" in found and found["Bank of America"]["kind"] == "organisation"


def test_spellings_are_merged_case_insensitively():
    found = entities.extract(["we use Zepto daily", "but ZEPTO is costly", "and Zepto again"], 2)
    assert list(found) == ["Zepto"] and found["Zepto"]["n"] == 3


def test_fillers_and_pronouns_never_become_entities():
    texts = ["Yeah I think OK it works.", "Okay so I said Yeah."] * 2
    assert entities.extract(texts, min_mentions=2) == {}
