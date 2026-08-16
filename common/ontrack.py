"""On-track events, read out of what was said rather than matched in it.

The graph knows who spoke, when, on which lap, and which drivers a line names. It did not
know what was happening *between the cars*, which is most of what radio is actually about:
someone is catching someone, someone got past, someone is in the dirty air, someone locked
up. This module is the vocabulary and the scoring for that; the extraction pass that reads
Postgres and the projection that writes Neo4j live elsewhere.

Classification is by SEMANTIC SIMILARITY, deliberately, not by keyword or alias matching.
Each event type carries a set of exemplar phrasings; an utterance is compared against them
in the same embedding space the corpus is already indexed in, and takes the type it is
closest to — if, and only if, it clears a similarity floor and beats the runner-up type by
a margin. Keyword matching would collapse the moment a driver says "I got him" instead of
"overtake", and the transcripts here are ASR output over a compressed radio channel: this
corpus contains "lando can you get closer to the car i had" for *car ahead*, which no word
list survives.

The margin rule is borrowed from speaker identification (`id_min_margin`), for the same
reason it exists there: two candidates a hair apart is not a close call to be broken by
picking the larger number, it is the system not knowing. Speech that clears neither test
gets no event at all, which is the common case and not a failure.

WHAT AN EVENT IS NOT: an event is an inference about a sentence, not an observation of the
race. "reads as an overtake attempt" is the strongest claim available from a radio line —
a driver saying "ready to get past Russell" may never have tried. Anything rendering these
for a human or a model says so, the same way tone readings do (see graph_context._tone_phrase).

Exemplars are written from real team radio: the phrasings below are drawn from this
corpus's own transcripts and from published radio transcripts and F1 radio glossaries.
They are intentionally clipped, unpunctuated and repetitive, because that is what the
channel sounds like — polished sentences would sit in a different part of the embedding
space from anything a driver actually says.
"""
from typing import NamedTuple

# Families group the types for display and for coarse queries. They are also the unit a
# reader can reason about: "something happened between these two cars" is a useful answer
# even when which of four overtaking types it was is genuinely ambiguous.
OVERTAKING = "overtaking"
PROXIMITY = "proximity"
INCIDENT = "incident"
FLAGS = "flags"


class EventType(NamedTuple):
    name: str
    family: str
    # How the type is worded when shown to a person or a model. Always a reading of the
    # line, never an assertion about the race.
    reads_as: str
    exemplars: tuple[str, ...]


EVENT_TYPES: tuple[EventType, ...] = (
    # ── overtaking ────────────────────────────────────────────────────────────
    EventType(
        "overtake_attempt", OVERTAKING,
        "an attempt to pass another car",
        (
            "ready to get past Russell",
            "let's go get Oscar",
            "I'm quicker than him, let me race",
            "I can pass him on this straight",
            "go for it, you have a chance into turn one",
            "I want to have a go at him this lap",
            "we're going to attack him now",
            "push now, this is your chance to get by",
            "I think I can get him into the last corner",
            "have a look down the inside",
        ),
    ),
    EventType(
        "overtake_completed", OVERTAKING,
        "a pass that has been completed",
        (
            "I got him",
            "yes, that's him done, nice move",
            "we are past him now",
            "great overtake, that was clean",
            "he's behind us now, well done",
            "position gained, you are P4",
            "you cleared him on the straight",
            "that's the move done, head down now",
            "I've overtaken him",
            "we got the position back",
        ),
    ),
    EventType(
        "overtaken_by", OVERTAKING,
        "being passed by another car",
        (
            "he got past me",
            "he's gone by, I couldn't hold him",
            "I lost the position to him",
            "he's ahead of us now",
            "he overtook me on the straight",
            "I've dropped behind him",
            "he came past into turn three",
            "we've lost a place there",
        ),
    ),
    EventType(
        "defending", OVERTAKING,
        "defending a position from a car behind",
        (
            "he's right behind you, defend the inside",
            "I'm holding him off",
            "keep him behind, don't let him through",
            "he's attacking, cover the inside line",
            "I can't keep him back much longer",
            "defend into turn one",
            "he's trying to come past, hold your line",
        ),
    ),

    # ── proximity and aero ────────────────────────────────────────────────────
    EventType(
        "closing", PROXIMITY,
        "closing on the car ahead, or a car behind closing in",
        (
            "can you get closer to the car ahead",
            "he's catching us, half a second a lap",
            "the gap is coming down, you're reeling him in",
            "we expect him to be within a second in a lap",
            "he's closing quickly behind you",
            "you're faster than him, keep chipping away",
            "gap ahead is one point two and falling",
        ),
    ),
    EventType(
        "gap_opening", PROXIMITY,
        "a gap growing between cars",
        (
            "the gap is growing, he's pulling away",
            "you're losing time to the car ahead",
            "he's four seconds up the road now",
            "we're dropping back from him",
            "the gap behind is stable, keep building the advantage",
            "you've got a comfortable margin now",
        ),
    ),
    EventType(
        "traffic", PROXIMITY,
        "being held up in traffic",
        (
            "traffic is going to ruin my race",
            "I'm stuck behind this car",
            "we're losing a lot of time in traffic",
            "there's a train of cars ahead",
            "he's holding me up, I can't get a lap in",
            "you'll have traffic in the last sector",
            "I'm in the queue, nothing I can do",
        ),
    ),
    EventType(
        "dirty_air", PROXIMITY,
        "dirty air, tow or slipstream between cars",
        (
            "I've got dirty air already from the car ahead",
            "I can't follow him closely, the front end goes away",
            "I'm losing downforce behind him",
            "you'll get a tow down the straight",
            "stay in his slipstream into the braking zone",
            "the air is really bad behind this car",
        ),
    ),
    EventType(
        "drs", PROXIMITY,
        "DRS range between two cars",
        (
            "he's got DRS on you",
            "you'll be in DRS range next lap",
            "DRS is enabled, he is within a second",
            "keep him out of the DRS window",
            "you have DRS on the main straight",
            "he's using DRS to close up",
        ),
    ),

    # ── incidents ─────────────────────────────────────────────────────────────
    EventType(
        "contact", INCIDENT,
        "contact between cars",
        (
            "he hit me, we made contact",
            "I've been hit from behind",
            "he ran into the side of me",
            "we touched wheels in that corner",
            "he drove straight into the side of me",
            "there was contact between us at the start",
        ),
    ),
    EventType(
        "lock_up", INCIDENT,
        "locking a wheel under braking",
        (
            "I locked up into turn one",
            "the rear locking is getting really bad in turn six",
            "big lock up at the hairpin, flat spotted it",
            "I keep locking the front left",
            "he locked up and went straight on",
        ),
    ),
    EventType(
        "off_track", INCIDENT,
        "going off the track or running wide",
        (
            "I went off at turn four",
            "he had an off, just be careful out there",
            "I ran wide and lost time",
            "went straight on at the chicane",
            "I put a wheel in the gravel",
            "off the track and back on again",
        ),
    ),
    EventType(
        "spin", INCIDENT,
        "spinning the car",
        (
            "I spun it, I'm facing the wrong way",
            "he's spun ahead of you, be careful",
            "I lost the rear and spun",
            "spun at the exit of turn two",
        ),
    ),
    EventType(
        "damage", INCIDENT,
        "damage to the car",
        (
            "check the car, it's pretty damaged",
            "I've got front wing damage",
            "there's damage on the floor, we're losing downforce",
            "the endplate is broken after that contact",
            "something is broken, the car feels wrong",
        ),
    ),

    # ── flags and lapping ─────────────────────────────────────────────────────
    EventType(
        "blue_flags", FLAGS,
        "blue flags for a car being lapped",
        (
            "he should have blue flags",
            "why are there no blue flags for this car",
            "blue flags are out, he will move",
            "the car ahead has blue flags",
        ),
    ),
    EventType(
        "lapping", FLAGS,
        "lapping a car, or being lapped",
        (
            "you're about to lap this car",
            "he's a lapped car, he'll let you through",
            "we are being lapped by the leader",
            "the leaders are coming, let them past",
        ),
    ),
    EventType(
        "let_by", FLAGS,
        "letting another car past, or being asked to",
        (
            "let him by, we'll get the position back",
            "I'm happy to give the position back",
            "swap positions on the next straight",
            "you need to give that place back",
            "move aside and let him through",
        ),
    ),
    EventType(
        "safety_car", FLAGS,
        "safety car or virtual safety car behaviour",
        (
            "safety car, safety car, close up to the car ahead",
            "we are under VSC, respect the delta",
            "safety car is coming in this lap, get ready",
            "virtual safety car deployed, lift and coast",
            "stay within ten car lengths under the safety car",
        ),
    ),
)

BY_NAME = {e.name: e for e in EVENT_TYPES}
FAMILIES = (OVERTAKING, PROXIMITY, INCIDENT, FLAGS)

# Everything else radio is about. Without these the nearest event type always wins by
# default, because "closest of eighteen" says nothing about whether any of them fit: on
# this corpus a floor alone read "Head down." as a completed overtake and "this car is so
# hot to drive, it's snappy like crazy" as dirty air. Competing against what radio is
# usually about turns "closest event" into "an event rather than the usual traffic of
# strategy, tyres, weather and encouragement".
#
# These are anchors, not a blocklist — an utterance that genuinely describes a pass still
# beats them. They exist so the common case has something true to be close to.
NONE_ANCHORS: tuple[str, ...] = (
    # strategy and pit work
    "box this lap, box box",
    "we're going to plan B, target lap 25",
    "do you think a one stop will work",
    "we are boxing the cars this way around to cover him",
    "pit window is open, we'll decide next lap",
    # tyres and car handling
    "how are the tyres feeling, any deg",
    "the rear tyres are gone, I have no grip",
    "the car is snappy on entry and understeering mid corner",
    "we need more front wing at the next stop",
    "the balance is much better now",
    # pace and lap times
    "that was a good lap, keep that pace",
    "still really good lap times, keep banging them in",
    "your last sector was two tenths quicker",
    "we need to be quicker than him on this stint",
    # weather and track
    "we expect rain in eight minutes around lap seven",
    "the track is drying, crossover in five to ten minutes",
    "it's pretty slippery out there",
    # mechanical and systems
    "suspension failure, I'm out",
    "oil leak, bring it back to the garage",
    "there's smoke in the cockpit, I don't know if I'm on fire",
    "check the engine mode, switch position two",
    # people talking to each other
    "well done everybody, fantastic job today",
    "sorry mate, you were fast today, unlucky",
    "thanks guys, I really appreciate it",
    "if you speak to me every lap I will disconnect",
    "yeah, copy that, understood",
    "I don't agree with that call, please",
)

# The pseudo-type the anchors score under. Never stored — reaching it is how the classifier
# says nothing.
NONE = "__none__"


class Reading(NamedTuple):
    """What a line was read as, and how strongly."""
    event_type: str
    family: str
    score: float   # cosine similarity to the closest exemplar of the winning type
    margin: float  # gap to the best score of any *other* type


def too_short(text: str | None, min_words: int) -> bool:
    """Whether a line is too thin to read anything into.

    Not a quality judgement about the speech — plenty of meaningful radio is two words.
    It is a statement about the method: a sentence embedding of "Head down." encodes those
    two words, and its nearest neighbour among eighteen event types is arbitrary."""
    return len((text or "").split()) < min_words


def classify(scores: dict[str, float], floor: float, min_margin: float) -> Reading | None:
    """The winning event type for one utterance, or None when the corpus should say nothing.

    `scores` maps event type name to its best exemplar similarity. Two ways to decline, and
    both are ordinary:

    - nothing clears `floor`: the line is not about anything on track. Most radio is not —
      it is strategy, tyres, encouragement and swearing.
    - the top two types are closer than `min_margin`: "he's right behind me" reads as both
      defending and closing, and picking the larger float would manufacture a distinction
      the sentence does not carry.

    Returning None rather than a low-confidence guess is the whole posture: an event edge
    in the graph is something a model will state as fact, so a wrong one is worse than a
    missing one.
    """
    events = {k: v for k, v in scores.items() if k != NONE}
    if not events:
        return None
    ranked = sorted(events.items(), key=lambda kv: kv[1], reverse=True)
    (best_name, best), *rest = ranked
    if best < floor:
        return None

    # Ordinary radio is the third test, and the one that does most of the work. A line
    # closer to "how are the tyres feeling" than to any event is about tyres, however well
    # it clears the floor.
    none_score = scores.get(NONE, 0.0)

    # The runner-up must be a *different* type; a second exemplar of the winner is
    # agreement, not competition. Types are the keys here, so `rest` is already that.
    runner_up = max(rest[0][1] if rest else 0.0, none_score)
    margin = best - runner_up
    if margin < min_margin:
        return None
    return Reading(best_name, BY_NAME[best_name].family, best, margin)


def phrase(reading: Reading) -> str:
    """How an event is worded wherever a person or a model reads it.

    Hedged in the same way tone is, and for the same reason: this is a classifier over one
    sentence of ASR output from a compressed radio channel, not a marshal's report."""
    return f"reads as {BY_NAME[reading.event_type].reads_as} ({reading.score:.2f})"
