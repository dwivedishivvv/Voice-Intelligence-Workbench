"""Dictionary entity linking over F1 transcripts.

A matcher over a closed alias vocabulary rather than an NER model, for two reasons: the
vocabulary is small and already in the database (f1_drivers, f1_sessions), and ASR mangles
proper nouns in ways a general NER has never been trained on — "Verstappen" comes back as
"Verstapen" often enough that the fix is an alias row, not a better model.

Kept out of graph_sync.py so the matching itself is testable without Postgres or Neo4j;
it is the part most likely to be wrong, and the part where wrongness is quietest (a
mention that silently fails to link just produces one fewer edge).
"""
import re

# alias -> the entities it may refer to. One alias can map to several: a driver's surname
# is the same across seasons while the number on the car is not, so resolution happens at
# match time using the session the speech belongs to.
AliasTable = dict[str, list[tuple[str, str]]]


def compile_aliases(table: AliasTable):
    """Build a single alternation regex over every alias.

    One pass per text instead of one pass per (text, alias) pair — with a few thousand
    utterances and a few dozen aliases that is the difference between one scan and tens
    of thousands. Longest-first so "red bull racing" wins over "red bull" at the same
    starting position; Python's alternation returns the first branch that matches, not
    the longest, so the ordering is doing real work rather than being cosmetic."""
    if not table:
        return None
    ordered = sorted(table, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(a) for a in ordered) + r")\b",
                         re.IGNORECASE)
    return pattern, table


def find_mentions(text: str | None, compiled) -> set[tuple[str, str]]:
    """Every (entity_type, entity_key) named in `text`. Empty set when there is nothing to
    match against — an unanalyzed call or an empty alias table is normal, not an error."""
    if not text or compiled is None:
        return set()
    pattern, table = compiled
    found: set[tuple[str, str]] = set()
    for m in pattern.finditer(text):
        found.update(table[m.group(1).lower()])
    return found


# Auto-derived aliases are deliberately conservative: surnames of 4+ characters and full
# team/circuit names. A false MENTIONS edge is worse than a missing one -- it silently
# pollutes every query that walks it -- so anything ambiguous is left to a hand-added
# source='manual' row where a session's transcripts justify it.
#
# Two exclusions, both evidence-backed against a real 674-utterance corpus:
#
#   First names. "Max" and "Lando" are ordinary words in a radio call ("max attack",
#   "land it").
#
#   Three-letter codes (VER, LEC, PER). These were seeded originally and measured zero
#   true positives: ASR spells names out and never emits a bare code -- a case-sensitive
#   scan for VER/LEC/HAM/NOR/PER/SAI/RUS/PIA/ALO/GAS across the corpus found none. They
#   did produce false ones, because the codes are ordinary English: "three seconds *per*
#   sector" linked to Perez. Zero signal, real noise.
MIN_SURNAME_LEN = 4

# Every statement is parameterless so the caller can just run the list. MIN_SURNAME_LEN is
# interpolated rather than bound: it is a module-level int under our control, never input.
SEED_ALIASES_SQL = [
    # surname: last whitespace-separated token of the full name
    f"""INSERT INTO f1_aliases (alias, entity_type, entity_key, source)
       SELECT DISTINCT lower(regexp_replace(btrim(full_name), '^.*\\s+', '')),
              'driver', driver_number::text, 'auto'
       FROM f1_drivers
       WHERE full_name IS NOT NULL
         AND length(regexp_replace(btrim(full_name), '^.*\\s+', '')) >= {MIN_SURNAME_LEN}
       ON CONFLICT DO NOTHING""",
    """INSERT INTO f1_aliases (alias, entity_type, entity_key, source)
       SELECT DISTINCT lower(team_name), 'team', team_name, 'auto'
       FROM f1_drivers WHERE team_name IS NOT NULL
       ON CONFLICT DO NOTHING""",
    """INSERT INTO f1_aliases (alias, entity_type, entity_key, source)
       SELECT DISTINCT lower(circuit), 'circuit', circuit, 'auto'
       FROM f1_sessions WHERE circuit IS NOT NULL
       ON CONFLICT DO NOTHING""",
]
