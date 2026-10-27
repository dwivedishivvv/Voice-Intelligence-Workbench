"""Named things in transcripts: the people, companies, places and products a corpus talks
about, read out of ASR text without a model.

Proper nouns arrive capitalised from Whisper, and that is the whole signal this uses. It is
a heuristic, and it is tuned to abstain: a capitalised word at the start of a sentence is
only a candidate if the same word also appears capitalised mid-sentence somewhere, and
graph_sync keeps an entity only once the corpus names it more than once
(graph_entity_min_mentions). A missing entity means "not clearly named", never "not there".

`kind` is a coarse reading, not a classification: `organisation` for a name containing a
company word ("Goodwater Capital", "Bank of America"), `person` for two or three plain title-cased words
("Kabir Biswas"), and `name` for everything else ("Dunzo", "Swiggy", "Bangalore") -- a
single capitalised word is as often a company as a city, and guessing which would be
inventing a fact the transcript does not state.

Pure functions, so tests/unit/test_entities.py checks them without a database.
"""
import re
from collections import Counter

# Words Whisper capitalises that are not names: sentence openers, pronouns, fillers, days
# and months (a date is not a subject), and the title words that begin a name but are not
# one on their own.
STOPWORDS = frozenset("""
a an and are as at be but by can could did do does for from had has have he her here hers
him his how i i'd i'll i'm i've if in into is it it's its just let let's me my no not now of
oh ok okay on or our ours she so some such than that that's the their them then there these
they this those to too up us very was we we're well were what when where which while who why
will with would yeah yes yep you you're your yours hi hello hey right also because like
actually basically really maybe sure thanks thank please sorry anyway however although
after before during since until today tomorrow yesterday first second third next last
one two three four five six seven eight nine ten hundred thousand million billion
monday tuesday wednesday thursday friday saturday sunday january february march april
may june july august september october november december mr mrs ms dr sir madam
""".split())

ORG_SUFFIXES = frozenset("""
inc ltd llc plc corp corporation company co group capital ventures partners bank labs
technologies technology systems solutions holdings foundation institute university
college school fund network networks media studios motors airlines
""".split())

# Words allowed *inside* a multi-word name ("Bank of America", "Tata & Sons"). Not "and":
# "Swiggy and Zomato" is two names, and joining across it would invent a third.
CONNECTORS = frozenset({"of", "de", "da", "von", "van", "&"})

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'’\-]*|&|[.!?]")
_POSSESSIVE = re.compile(r"['’]s$")


def _clean(word: str) -> str:
    return _POSSESSIVE.sub("", word).strip("'’-")


def _is_name_word(word: str) -> bool:
    w = _clean(word)
    if len(w) < 2 or not w[0].isupper() or w.lower() in STOPWORDS:
        return False
    # An all-caps word is an acronym ("IBM", "NASA") only while it is short; a long one is
    # Whisper shouting a whole phrase.
    return not w.isupper() or (len(w) <= 5 and w.isalpha())


def candidates(text: str) -> list[tuple[str, bool]]:
    """Capitalised spans in `text`, each with whether it opened a sentence.

    A span is a run of name words, optionally joined by connectors ("Bank of America"); a
    trailing connector is dropped. Returned in order, duplicates kept -- counting is the
    caller's job."""
    out: list[tuple[str, bool]] = []
    tokens = _TOKEN.findall(text or "")
    sentence_start = True
    span: list[str] = []
    span_at_start = False

    def flush():
        nonlocal span
        while span and span[-1].lower() in CONNECTORS:
            span.pop()
        if span:
            out.append((" ".join(_clean(w) for w in span), span_at_start))
        span = []

    for tok in tokens:
        if tok in ".!?":
            flush()
            sentence_start = True
            continue
        if _is_name_word(tok):
            if not span:
                span_at_start = sentence_start
            span.append(tok)
        elif span and tok.lower() in CONNECTORS:
            span.append(tok)
        else:
            flush()
        sentence_start = False
    flush()
    return out


def kind_of(name: str) -> str:
    words = name.split()
    # Any company word, not only a trailing one: "Bank of America", "University of Leeds".
    if len(words) > 1 and any(w.lower() in ORG_SUFFIXES for w in words) or "&" in words:
        return "organisation"
    if 2 <= len(words) <= 3 and all(w[0].isupper() and w[1:].islower() and w.isalpha()
                                    for w in words):
        return "person"
    return "name"


def extract(texts: list[str], min_mentions: int = 2) -> dict[str, dict]:
    """Corpus-level entities: {name: {kind, n}} for every name mentioned at least
    `min_mentions` times, at least once mid-sentence.

    Matching is case-insensitive and the most common spelling wins, so "DUNZO" and "Dunzo"
    are one entity."""
    found = [c for text in texts for c in candidates(text)]
    seen_mid = {name.lower() for name, at_start in found if not at_start}
    counts: Counter[str] = Counter()
    mid_sentence: set[str] = set()
    spellings: dict[str, Counter[str]] = {}
    for name, at_start in found:
        if at_start and " " in name and name.lower() not in seen_mid:
            # "See Dunzo ..." opens a sentence with a capitalised verb. Unless the whole
            # span also appears mid-sentence, the first word is the sentence's, not the name's.
            name = name.split(" ", 1)[1]
        key = name.lower()
        counts[key] += 1
        spellings.setdefault(key, Counter())[name] += 1
        if not at_start or key in seen_mid:
            mid_sentence.add(key)
    out = {}
    for key, n in counts.items():
        if n < min_mentions or key not in mid_sentence:
            continue
        name = spellings[key].most_common(1)[0][0]
        out[name] = {"kind": kind_of(name), "n": n}
    return out


def mentions(text: str, known: dict[str, str]) -> list[str]:
    """The known entities (lowercase name -> canonical name) that `text` names."""
    found = []
    for name, _ in candidates(text):
        # Longest known suffix of the span, so "See Dunzo" still names Dunzo.
        words = name.split()
        canonical = next((known[k] for k in (" ".join(words[i:]).lower()
                                             for i in range(len(words))) if k in known), None)
        if canonical and canonical not in found:
            found.append(canonical)
    return found
