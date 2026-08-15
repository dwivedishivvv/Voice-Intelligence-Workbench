"""Entity linking is a dictionary matcher over a closed vocabulary (common/entities.py).
It is the fiddliest part of the graph projection and the quietest when wrong: a mention
that fails to match just produces one fewer edge, and a mention that matches something it
shouldn't produces a wrong edge that every later query walks as though it were a fact.
"""
import pytest

from common.entities import compile_aliases, find_mentions

TABLE = {
    "verstappen": [("driver", "1")],
    "ver": [("driver", "1")],
    "norris": [("driver", "4")],
    "red bull": [("team", "Red Bull Racing")],
    # A real overlap, not a contrived one: the team is Red Bull and the circuit is the
    # Red Bull Ring, and one alias is a strict prefix of the other.
    "red bull ring": [("circuit", "Red Bull Ring")],
    "monaco": [("circuit", "Monaco")],
}
COMPILED = compile_aliases(TABLE)


@pytest.mark.parametrize("text,expected", [
    ("Verstappen is closing", {("driver", "1")}),
    ("verstappen is closing", {("driver", "1")}),          # case-insensitive
    ("VER is closing", {("driver", "1")}),
    ("Box now, Norris is on softs", {("driver", "4")}),
    ("Gap to Verstappen and Norris?", {("driver", "1"), ("driver", "4")}),
    ("Nothing relevant in this call", set()),
])
def test_matches_known_aliases(text, expected):
    assert find_mentions(text, COMPILED) == expected


@pytest.mark.parametrize("text", [
    "silverstone",             # 'ver' inside the word
    "converse about it",       # 'ver' inside the word
    "the driver ahead",        # 'ver' at the END of a word — needs the leading \b
    "which version of the map", # 'ver' at the START of a word — needs the trailing \b
    "silver arrows",           # 'ver' at the end again
])
def test_does_not_match_inside_longer_words(text):
    """Word boundaries are the whole reason a substring search wasn't good enough: 'ver' is
    a driver code and also three letters sitting inside ordinary English. 'driver' is the
    case that matters most — it ends in 'ver' and is the most common word in a team-radio
    transcript, so a missing leading boundary would tag nearly every call as a mention of
    Verstappen. Both ends are covered deliberately: each boundary is load-bearing on its
    own, and a test that only exercises infix matches passes with either one removed."""
    assert find_mentions(text, COMPILED) == set()


def test_longest_alias_wins_at_the_same_position():
    """'red bull ring' and 'red bull' start at the same offset and mean different things.
    Python's alternation returns the first branch that matches, not the longest, so
    compile_aliases sorts longest-first — without that the circuit could never win and
    every mention of the venue would be silently filed as a mention of the team."""
    assert find_mentions("Big crowd at the Red Bull Ring", COMPILED) == {
        ("circuit", "Red Bull Ring")}
    # and the shorter alias still matches on its own
    assert find_mentions("Red Bull have the undercut", COMPILED) == {("team", "Red Bull Racing")}


def test_one_alias_can_resolve_to_several_entities():
    """A surname keeps its meaning across seasons while the car number does not, so the
    table is alias -> list. Both candidates come back; the session picks the right one at
    MATCH time in the projection."""
    table = {"verstappen": [("driver", "1"), ("driver", "33")]}
    assert find_mentions("Verstappen again", compile_aliases(table)) == {
        ("driver", "1"), ("driver", "33")}


def test_repeated_mention_yields_one_entity():
    """Return type is a set: two mentions of one driver are one MENTIONS edge, not two."""
    assert find_mentions("Verstappen, Verstappen, VER!", COMPILED) == {("driver", "1")}


@pytest.mark.parametrize("text,compiled", [
    (None, COMPILED), ("", COMPILED),
    ("Verstappen", None), ("Verstappen", compile_aliases({})),
])
def test_empty_inputs_are_not_errors(text, compiled):
    """An un-analyzed call has no text, and a database with no F1 data yet has no aliases.
    Both are ordinary states during a first run, not failures."""
    assert find_mentions(text, compiled) == set()


def test_regex_metacharacters_in_an_alias_are_literal():
    """Aliases are user-editable rows. One containing '.' or '(' must match itself, not act
    as a pattern — and must never raise while compiling the combined regex."""
    compiled = compile_aliases({"a.j. foyt": [("driver", "9")]})
    assert find_mentions("here comes a.j. foyt", compiled) == {("driver", "9")}
    assert find_mentions("here comes axjy foyt", compiled) == set()


# --- seeding policy ---------------------------------------------------------
# These assert the *vocabulary* rules, not the matcher. Both were established by measuring
# against a real corpus, so they are the kind of decision that gets quietly reverted by
# someone adding "just one more" alias source without re-running that measurement.

def test_seeding_does_not_derive_three_letter_driver_codes():
    """Codes like VER/PER measured zero true positives against 674 real utterances (ASR
    spells names out) while producing false ones — 'three seconds per sector' linked to
    Perez. Re-adding them re-introduces noise for no recall."""
    from common.entities import SEED_ALIASES_SQL
    joined = " ".join(SEED_ALIASES_SQL).lower()
    assert "name_acronym" not in joined


def test_seeding_requires_a_meaningful_surname_length():
    """Short surnames collide with ordinary words; the length floor is what keeps the
    auto-derived vocabulary conservative."""
    from common.entities import MIN_SURNAME_LEN, SEED_ALIASES_SQL
    assert MIN_SURNAME_LEN >= 4
    assert f">= {MIN_SURNAME_LEN}" in " ".join(SEED_ALIASES_SQL)
