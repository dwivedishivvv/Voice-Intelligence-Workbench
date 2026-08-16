"""The agentic layer: an LLM that answers questions about the corpus by querying it.

Tools, not a pre-stuffed context blob (AGENT_LAYER_PLAN.md). Handing the model the top-k
search results and asking the question fails as soon as an answer needs two hops, because
retrieval cannot know what to fetch until after the model has reasoned about the first
result. Four read-only tools plus the SDK's tool runner is less code than a query planner
and is the only shape that uses the graph for what a graph is for.

Every tool returns *rendered text* carrying stable ids, not JSON. That is deliberate: the
renderer in graph_context.py is where tone is phrased as a reading rather than a fact, so
routing tool output through it means the model never sees a confident-sounding claim the
underlying heuristic does not support.

Nothing here writes. The graph and the corpus are read-only to the agent; corrections stay
in the existing human-in-the-loop UI.
"""
import json
import re

from anthropic import AsyncAnthropic, beta_async_tool

from common import db
from common.config import get_effective_settings
from . import graph_context, search as search_svc

# Anchors are UUIDs; this is what turns a cited id in the answer into a link in the UI.
SPEECH_ID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)


# Aggregated in two separate CTEs, then joined 1:1. Doing both counts in a single query
# with two LEFT JOINs fans out -- every utterance row is repeated once per clip_speakers
# row for the same profile -- and reported 693 utterances for a speaker who has 33, in a
# corpus of 674. The numbers still looked plausible, which is exactly why this is written
# out rather than left as the obvious one-query version.
#
# The status filter matches every other query that reads pipeline output: a half-processed
# clip has no finished transcript, and counting it inflates a figure the agent will state
# as fact.
COMPARE_SPEAKERS_SQL = """
WITH utt AS (
  SELECT u.profile_id,
         count(*)                                   AS n_utterances,
         count(DISTINCT u.clip_id)                  AS n_clips,
         round(avg(u.sentiment_score)::numeric, 2)  AS avg_sentiment,
         mode() WITHIN GROUP (ORDER BY u.mood)      AS common_mood
  FROM utterances u JOIN clips c ON c.id = u.clip_id
  WHERE c.status = 'COMPLETE' AND u.profile_id = ANY($1::uuid[])
  GROUP BY u.profile_id
),
share AS (
  SELECT cs.profile_id, round(avg(cs.talk_share)::numeric, 2) AS avg_talk_share
  FROM clip_speakers cs JOIN clips c ON c.id = cs.clip_id
  WHERE c.status = 'COMPLETE' AND cs.profile_id = ANY($1::uuid[])
  GROUP BY cs.profile_id
)
SELECT p.id, p.display_name, p.status, p.n_enrollments,
       coalesce(utt.n_utterances, 0) AS n_utterances,
       coalesce(utt.n_clips, 0)      AS n_clips,
       utt.avg_sentiment, utt.common_mood, share.avg_talk_share
FROM speaker_profiles p
LEFT JOIN utt   ON utt.profile_id = p.id
LEFT JOIN share ON share.profile_id = p.id
WHERE p.id = ANY($1::uuid[])
"""


# Values a model sends when it means "no filter". Exact matches only -- a prefix or
# substring rule would swallow real names ("Norris" starts like "no").
PLACEHOLDER_ARGS = frozenset({"", "null", "none", "undefined", "nil", "n/a", "na", "-"})


# Words that name a tone reading rather than a topic. Models reach for the word before the
# filter -- "which calls sound stressed" arrives as a text query, which matches nothing,
# because the tone is a stored label and not something anybody says out loud. Kept to
# synonyms of the three labels the classifier actually emits: mapping "angry" or "urgent"
# onto `stressed` would be this layer inventing a category the pipeline does not have.
TONE_WORDS = {
    "stressed": "stressed", "stress": "stressed", "stressful": "stressed",
    "calm": "calm", "relaxed": "calm",
    "tired": "tired", "fatigued": "tired", "exhausted": "tired",
}


def tone_word(text: str) -> str | None:
    """The tone reading a query is really asking for, if any. Word-boundary matched so
    "distressed" does not read as "stressed"."""
    for word, mood in TONE_WORDS.items():
        if re.search(rf"\b{word}\b", text, re.I):
            return mood
    return None


def is_placeholder(value: str | None) -> bool:
    """True when an optional string argument is really absent.

    Models routinely send an omitted optional argument as the *string* "null" rather than
    omitting it or sending JSON null. Taken literally that filters results down to a
    speaker who cannot exist, and the tool answers "there is no speech by null" -- which
    the user reads as a statement about the corpus rather than a malformed tool call.
    Observed in a real session."""
    return value is None or value.strip().lower() in PLACEHOLDER_ARGS


class LLMUnavailable(RuntimeError):
    """The agent is disabled or unconfigured. Callers turn this into a 503 — a deployment
    state, not a bug in the request."""


# --------------------------------------------------------------------------- tools
# Tool descriptions are the highest-leverage text in this file. Each says *when* to reach
# for it, not just what it does: on current models the most common failure is a tool that
# never gets called because its description only described itself.

@beta_async_tool
async def search_speech(query: str = "", spoken_by: str | None = None,
                        mood: str | None = None, limit: int = 8) -> str:
    """Find speech in the corpus by topic, by tone, or by both. Start here for almost any
    question.

    Searches both pipeline-processed utterances and F1 team-radio calls. Returns one line
    per hit: the speech id, who said it, the session and lap where known, the tone reading
    where there is one, and a clipped quote. Call expand_speech on the ids that matter
    before drawing a conclusion.

    Search the TOPIC, not the person. A name in `query` matches speech that *mentions*
    that name, which is usually somebody else talking about them — to find what a person
    said, put the subject in `query` and the person in `spoken_by`.

    Tone is a separate axis from topic, because it is a column rather than words: no
    amount of searching for "stressed" finds speech whose voice reads stressed. Use
    `mood` for that, alone to list speech by tone, or with a `query` to combine the two.
    Every filter composes: query + spoken_by + mood is one search, not three.

    Args:
        query: The topic to look for, in natural language — "power failure", "tyre wear".
            Do not put a driver or speaker name here unless you want speech that mentions
            them. Omit it when filtering by tone alone.
        spoken_by: Restrict to one speaker: a driver code ("ALB"), a driver number ("23"),
            or an enrolled speaker's name. Omit to search everyone.
        mood: Restrict to one tone reading: "calm", "stressed" or "tired". These are
            heuristic readings of the audio, not measurements of how anyone felt, and with
            no `query` the results are ordered by recency — the classifier emits a label,
            not a magnitude, so there is no "most stressed".
        limit: How many hits to return. Default 8, capped at 25.
    """
    if is_placeholder(spoken_by):
        spoken_by = None
    if is_placeholder(mood):
        mood = None
    if mood:
        mood = mood.strip().lower()
        if mood not in search_svc.MOODS:
            return (f'"{mood}" is not a tone reading. The classifier emits only '
                    f'{", ".join(search_svc.MOODS)} — anything else matches nothing, which '
                    f"is a bad argument rather than a fact about the corpus.")
    limit = max(1, min(int(limit or 8), 25))
    topic = None if is_placeholder(query) else query.strip()
    if not topic and not mood:
        return ("Give a topic in `query`, or a tone reading in `mood`, or both. "
                "An empty search has nothing to rank and nothing to filter.")

    # Over-fetch before filtering: the speaker filter is applied after ranking, so asking
    # for exactly `limit` would return almost nothing once a filter is set.
    over = limit * 6 if spoken_by else limit
    note = ""
    if topic:
        anchors = await search_svc.anchor_speech(topic, limit=over, mood=mood)
        # Recovery, not guesswork: only when the text search found nothing, only when no
        # tone filter was given, and the reinterpretation is stated in the result so the
        # model reports what was actually searched rather than what it asked for.
        if not anchors and not mood and (inferred := tone_word(topic)):
            anchors = await search_svc.speech_by_mood(inferred, limit=over)
            mood = inferred
            note = (f"Nothing in the transcripts mentions {topic!r}. A voice reading is a "
                    f"stored label, not a word anyone says, so this is speech whose voice "
                    f"reads {inferred}, most recent first:\n")
    else:
        anchors = await search_svc.speech_by_mood(mood, limit=over)

    if not anchors:
        scope = ", ".join(filter(None, [
            f"topic {topic!r}" if topic else None,
            f"tone {mood}" if mood else None]))
        return f"No speech found for {scope}."
    try:
        # One extra graph round trip to attach speaker, session and lap to each hit, reusing
        # expand() rather than adding a second Cypher query for the same edges. The graph
        # row is layered *over* the anchor, so ranking order is preserved and a speech the
        # graph has not seen yet (added since the last sync) keeps its text.
        by_id = {r["speech_id"]: r for r in
                 await graph_context.expand([a["speech_id"] for a in anchors])}
        rows = [{**a, **by_id.get(a["speech_id"], {})} for a in anchors]
    except Exception:
        # Graph down or un-synced: degrade to what search alone knows rather than failing
        # the tool. A hit without its speaker is still a hit.
        rows = anchors

    if spoken_by:
        want = spoken_by.strip().lower()

        def is_match(r: dict) -> bool:
            # Matched in both directions on purpose. The graph stores a driver as the
            # three-letter code, so a model asking for "Albon" finds nothing against "ALB"
            # with a one-way containment check -- which reads as "no such speech" and gets
            # reported as a gap in the data rather than a mismatch in the lookup.
            for field in ("driver", "driver_name", "speaker", "team"):
                have = str(r.get(field) or "").strip().lower()
                if have and (want in have or have in want):
                    return True
            return False

        rows = [r for r in rows if is_match(r)]
        if not rows:
            scope = " and ".join(filter(None, [
                f"matching {topic!r}" if topic else None,
                f"whose voice reads {mood}" if mood else None])) or "in the corpus"
            return (f'No speech by "{spoken_by}" {scope}. The speaker may not be '
                    f"enrolled, or that driver has no analyzed radio in the corpus.")
    return note + (graph_context.render_brief(rows[:limit]) or "No matching speech found.")


@beta_async_tool
async def expand_speech(speech_id: str) -> str:
    """Show everything the graph knows around one piece of speech, or several at once.

    Who said it, in which session, the lap it happened on and that lap's time against the
    previous one, the drivers or teams it names, and the lines immediately before and
    after. Call this before answering a question about a specific moment — search results
    alone omit the surroundings that usually decide what a quote actually means.

    Pass several ids separated by commas to expand them in one call. A question about a
    pattern across the corpus needs the context of every hit, and one id per call turns
    that into a round trip each.

    Args:
        speech_id: One id from search_speech, of the form "3f2b1a4c-…", or several
            separated by commas. Ids appear in square brackets at the start of each
            search result. At most 10 are expanded per call.
    """
    ids = SPEECH_ID_RE.findall(speech_id or "")
    if not ids:
        return (f"No speech id found in {speech_id!r}. Ids are UUIDs from search_speech, "
                f"in square brackets at the start of each result.")
    # resolve(), not expand(): a stale graph would otherwise answer "no such speech" for an
    # id search_speech had just returned, which the model reports as a gap in the corpus.
    rows = await graph_context.resolve(ids[:10])
    if not rows:
        return (f"No speech with {'id' if len(ids) == 1 else 'those ids'} {', '.join(ids[:10])}. "
                f"Ids come from search_speech.")
    # Say which ids came back empty rather than silently returning fewer than asked for —
    # a model comparing counts would otherwise read the gap as absence of evidence.
    missing = [i for i in ids[:10] if i not in {r["speech_id"] for r in rows}]
    out = graph_context.render_context(rows)
    if missing:
        out += f"\n\nNot in the graph: {', '.join(missing)}."
    return out


@beta_async_tool
async def driver_timeline(session_key: int, driver_number: int,
                          from_lap: int | None = None, to_lap: int | None = None) -> str:
    """Follow one driver through one session: lap times with the radio spoken on each lap.

    Use this for questions about how a driver's race unfolded, or whether what was said
    lines up with a change in pace — the comparison search cannot make, because it ranks
    speech without knowing what the car was doing.

    With no lap range, returns only the laps that have speech on them, plus pace summary
    statistics. Give a range to see every lap in it, including silent ones.

    Args:
        session_key: OpenF1 session key. The available sessions are listed in your context.
        driver_number: The car number.
        from_lap: First lap of an explicit range.
        to_lap: Last lap of an explicit range.
    """
    rows = await graph_context.driver_laps(session_key, driver_number, from_lap, to_lap)
    return graph_context.render_timeline(rows, from_lap, to_lap) or (
        f"No lap data for driver {driver_number} in session {session_key}.")


@beta_async_tool
async def compare_speakers(profile_id_a: str, profile_id_b: str | None = None) -> str:
    """Contrast enrolled speakers: how much they talk, and how they come across.

    Use this for questions that put people side by side. It reports counts and aggregate
    readings only — for what was actually said, search instead.

    Two speakers is the common case, but either argument accepts a comma-separated list,
    so "how do all the engineers compare" is one call rather than a matrix of pairs.
    Names work as well as ids: a name is resolved against the speaker directory, because a
    question about "Marta" should not fail on the model not having copied a UUID.

    Args:
        profile_id_a: A speaker profile id or display name, or several separated by commas.
        profile_id_b: The other speaker, when comparing exactly two. Omit when the first
            argument already lists everyone to compare.
    """
    tokens = [t.strip() for arg in (profile_id_a, profile_id_b)
              if not is_placeholder(arg) for t in str(arg).split(",") if t.strip()]
    if not tokens:
        return "Name at least one speaker, by profile id or display name."

    ids = [t for t in tokens if SPEECH_ID_RE.fullmatch(t)]
    names = [t for t in tokens if not SPEECH_ID_RE.fullmatch(t)]
    if names:
        # Case-insensitive exact match: a LIKE would let "Dan" pull in "Dan Reisz" *and*
        # "Daniel Voss" and report a comparison between the wrong pair.
        found = await db.fetch(
            "SELECT id, display_name FROM speaker_profiles WHERE lower(display_name) = ANY($1)",
            [n.lower() for n in names])
        ids += [str(r["id"]) for r in found]
        unknown = [n for n in names
                   if n.lower() not in {r["display_name"].lower() for r in found}]
        if unknown:
            return (f"No enrolled speaker named {', '.join(unknown)}. The directory in your "
                    f"context lists everyone who is enrolled.")

    rows = await db.fetch(COMPARE_SPEAKERS_SQL, ids)
    if not rows:
        return "No profile matched. Ids and names come from the speaker directory."
    out = []
    for r in rows:
        out.append(
            f"[{r['id']}] {r['display_name']} ({r['status']}, {r['n_enrollments']} enrollments)\n"
            f"  {r['n_utterances']} utterances across {r['n_clips']} clips"
            + (f", mean talk share {r['avg_talk_share']}" if r["avg_talk_share"] is not None else "")
            + "\n  aggregate readings: "
            + (f"sentiment {r['avg_sentiment']:+}" if r["avg_sentiment"] is not None else "no sentiment")
            + (f", voice most often {r['common_mood']}" if r["common_mood"] else ""))
    return "\n".join(out)


TOOLS = [search_speech, expand_speech, driver_timeline, compare_speakers]

# The tool layer is the provider seam, exactly as designed: these four functions are plain
# async code that knows nothing about who is calling them, so swapping the model means
# swapping the loop below and nothing else.
#
# @beta_async_tool stays the single declaration even on the OpenAI-compatible path -- it
# derives the schema from the type hints and docstring, which is what keeps each tool's
# description living next to its implementation instead of in a hand-maintained dict.
# _openai_tools() adapts that one source of truth into the other wire format.
BY_NAME = {t.to_dict()["name"]: t for t in TOOLS}

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    # Verified to emit well-formed tool_calls against this corpus; meta/llama-3.1-70b-instruct
    # also works if this one is unavailable.
    # Non-reasoning on purpose. The nemotron-super reasoning models on this endpoint
    # return content=null plus a separate `reasoning` field and spend the whole token
    # budget thinking before emitting anything, which turns a four-tool loop into minutes
    # per turn. This one answers in ~2s and emits well-formed tool_calls. Point LLM_MODEL
    # at a reasoning model deliberately, with a much larger LLM_MAX_TOKENS, if wanted.
    "nvidia": "meta/llama-3.1-70b-instruct",
}


def _openai_tools() -> list[dict]:
    """Anthropic tool schema -> OpenAI function schema. Same names, same JSON Schema for
    parameters; only the envelope differs."""
    out = []
    for t in TOOLS:
        d = t.to_dict()
        out.append({"type": "function", "function": {
            "name": d["name"], "description": d["description"],
            "parameters": d["input_schema"]}})
    return out


def _coerce_args(schema: dict, args: dict) -> dict:
    """Coerce model-supplied arguments to the types the schema declares.

    Tool arguments arrive as JSON the model wrote, and a declared type is a request, not a
    guarantee -- smaller models routinely emit "1" where the schema says integer. That
    string reaches asyncpg as a bind parameter for a Postgres LIMIT and raises DataError,
    failing the whole turn on what is really a formatting slip. Coercing here keeps the
    tool implementations written against real types instead of defensively re-parsing.

    Anything that will not convert is passed through untouched, so it surfaces as a normal
    argument error the model can read and retry rather than a silent wrong value."""
    def declared_type(spec: dict):
        """The type to coerce to. An optional parameter is rendered as
        anyOf: [{type: integer}, {type: null}] rather than a plain type, and those are
        precisely the parameters a model is most likely to stringify -- reading only
        `type` would skip every one of them."""
        if "type" in spec:
            return spec["type"]
        for branch in spec.get("anyOf", []):
            if branch.get("type") and branch["type"] != "null":
                return branch["type"]
        return None

    props = (schema or {}).get("properties", {})
    out = {}
    for key, value in args.items():
        want = declared_type(props.get(key, {}))
        try:
            if value is None or want is None:
                out[key] = value
            elif want == "integer" and not isinstance(value, bool):
                out[key] = int(value)
            elif want == "number" and not isinstance(value, bool):
                out[key] = float(value)
            elif want == "boolean" and isinstance(value, str):
                out[key] = value.strip().lower() in ("true", "1", "yes")
            elif want == "string" and not isinstance(value, str):
                out[key] = str(value)
            else:
                out[key] = value
        except (TypeError, ValueError):
            out[key] = value
    return out


async def _dispatch(name: str, args: dict) -> str:
    """Run one tool by name. A bad name or bad arguments come back as text for the model to
    read and correct, rather than as an exception that kills the turn -- a model that
    mis-calls a tool should get a chance to try again."""
    tool = BY_NAME.get(name)
    if tool is None:
        return f"No tool named {name}. Available: {', '.join(sorted(BY_NAME))}."
    try:
        return await tool.func(**_coerce_args(tool.to_dict().get("input_schema"), args))
    except TypeError as e:
        return f"Bad arguments for {name}: {e}"
    except Exception as e:
        # One tool failing is not the turn failing: hand the model the error so it can try
        # a different call, rather than 500-ing a request that may still be answerable.
        return f"{name} failed: {type(e).__name__}: {str(e)[:200]}"


# --------------------------------------------------------------------------- prompt

SYSTEM = """You answer questions about a voice-intelligence corpus: recorded speech that \
has been transcribed, diarized, attributed to speakers where possible, and — for Formula 1 \
team radio — lined up against session and lap data.

Answer by using your tools. Search first, expand what looks relevant, then answer from what \
you actually read. Do not answer from general knowledge about motorsport or about the \
people involved; the only thing you know is what the corpus says.

Three rules govern how you report what you find.

Cite. Every factual claim carries the speech id it came from, in square brackets, exactly \
as the tools return it. A claim you cannot attach an id to is one you should not make.

Tone is a reading, not a fact. The mood label on a piece of speech comes from a threshold \
heuristic over acoustic features — speech rate, pitch, energy — not from a trained emotion \
model, and the text sentiment is a separate model that can disagree with it. Say "his voice \
reads stressed" or "the words read negative", never "he was stressed". Where the two \
readings disagree, report the disagreement; it is a signal, not noise to average away.

Tone is searchable, but only through the mood filter. It is a stored label, not words, so \
searching for the word "stressed" finds speech that says the word — pass mood="stressed" \
instead, on its own or alongside a topic. With no topic those results are ordered by \
recency, not by intensity: there is no "most stressed", only speech the classifier labelled \
that way.

Say when the corpus cannot answer. Speaker identification abstains rather than guess, so \
plenty of speech is attributed to nobody. Lap alignment needs a recording timestamp that \
manually uploaded audio does not have. If the data does not support an answer, say which \
part is missing instead of filling the gap.

Keep responses brief and specific. Lead with the answer, then the evidence. Do not restate \
the question, do not narrate which tools you are about to call, and do not pad with caveats \
beyond the ones above.

Do not re-quote the evidence. Every id you cite is rendered beside your answer with its \
full quote, speaker and a link to the audio, so repeating those lines in prose adds nothing \
and makes the reader wait for it. Characterise what was said in your own words and attach \
the ids: "three calls read stressed, all about tyre wear [id] [id] [id]" — not a bulleted \
list of the quotes themselves. Two or three sentences answers most questions.

Brevity never costs citations. A bare count or a summary with no ids attached is not an \
answer — it is the one thing the reader cannot check. Every id behind what you are \
describing goes in the text, however short the text is."""


async def _roster() -> str:
    """Entity roster, rendered into the cached system prompt rather than exposed as a tool.

    It is a few hundred stable tokens against a tool round-trip on nearly every question,
    and it removes an entire failure class: the model inventing a session key or a driver
    number instead of looking one up."""
    parts: list[str] = []

    sessions = await db.fetch(
        """SELECT session_key, name, year, circuit, country FROM f1_sessions
           WHERE circuit IS NOT NULL ORDER BY date_start DESC LIMIT 40""")
    if sessions:
        parts.append("F1 sessions (session_key: name):\n" + "\n".join(
            f"  {s['session_key']}: {s['name']} {s['year']} — {s['circuit']}, {s['country']}"
            for s in sessions))

    drivers = await db.fetch(
        """SELECT DISTINCT driver_number, full_name, team_name FROM f1_drivers
           WHERE full_name IS NOT NULL ORDER BY driver_number""")
    if drivers:
        parts.append("Drivers (number: name, team):\n" + "\n".join(
            f"  {d['driver_number']}: {d['full_name']} — {d['team_name']}" for d in drivers))

    speakers = await db.fetch(
        "SELECT id, display_name FROM speaker_profiles ORDER BY display_name")
    if speakers:
        parts.append("Enrolled speakers (profile id: name):\n" + "\n".join(
            f"  {s['id']}: {s['display_name']}" for s in speakers))

    races = await db.fetch("SELECT name, circuit, race_date FROM races ORDER BY race_date DESC")
    if races:
        parts.append("Races (user-created groupings of clips):\n" + "\n".join(
            f"  {r['name']} — {r['circuit'] or 'no circuit'}" for r in races))

    return "\n\n".join(parts) if parts else "The corpus currently contains no F1 or speaker data."


async def build_system(cfg=None) -> list[dict]:
    """System prompt + roster as one cached block.

    Stable across every turn of a conversation and across conversations, comfortably over
    the 512-token minimum, so it is a cache read rather than a re-send on all but the first
    request. Nothing volatile — no timestamp, no request id — goes in here; that would
    invalidate the prefix on every call and silently cost full price."""
    return [{"type": "text",
             "text": SYSTEM + "\n\n--- what is in the corpus right now ---\n\n" + await _roster(),
             "cache_control": {"type": "ephemeral"}}]


# --------------------------------------------------------------------------- runner

async def answer(question: str, history: list[dict] | None = None, cfg=None,
                 on_event=None) -> dict:
    """Run one agent turn to completion, on whichever provider is configured.

    `on_event(kind, **payload)` is awaited for each step so a caller can stream progress;
    the full result is returned regardless, so a client that never subscribes still gets
    the answer. That is the reason this is not fire-and-forget: pub/sub has no replay, and
    an answer that exists only as a dropped message is an answer that is gone.
    """
    cfg = cfg or await get_effective_settings()
    if not cfg.llm_enabled:
        raise LLMUnavailable("agent is disabled (set LLM_ENABLED=true)")

    provider = (cfg.llm_provider or "anthropic").lower()
    if provider not in DEFAULT_MODELS:
        raise LLMUnavailable(f"unknown LLM_PROVIDER {provider!r}; "
                             f"expected one of {', '.join(sorted(DEFAULT_MODELS))}")
    model = cfg.llm_model or DEFAULT_MODELS[provider]

    async def emit(kind, **kw):
        if on_event:
            await on_event(kind, **kw)

    # Credentials are checked before anything reads the corpus. build_system() queries the
    # roster out of the database, and doing that ahead of the gate means an unconfigured or
    # disabled agent still pulls corpus data into memory on its way to failing -- the exact
    # ordering the "check the gate before you build a request" rule exists to prevent.
    key = cfg.anthropic_api_key if provider == "anthropic" else cfg.nvidia_api_key
    if not key:
        raise LLMUnavailable(
            f"{'ANTHROPIC_API_KEY' if provider == 'anthropic' else 'NVIDIA_API_KEY'} is not set")

    system = (await build_system(cfg))[0]["text"]
    messages = list(history or []) + [{"role": "user", "content": question}]

    if provider == "anthropic":
        result = await _run_anthropic(cfg, model, system, messages, emit)
    else:
        result = await _run_openai_compatible(cfg, model, system, messages, emit)

    result["model"] = model
    result["provider"] = provider
    if result["refused"]:
        await emit("error", message="declined")
    else:
        await emit("done", tools_used=result["tools_used"], citations=result["citations"])
    return result


def _finish(text: str, tools_used: list[str], usage: dict, *, refused=False, detail=None,
            sources: list[str] | None = None) -> dict:
    """`citations` are ids the model actually wrote into its answer. `sources` are the ids
    its tools put in front of it.

    They are kept apart because models honour the citation instruction inconsistently --
    observed answering correctly with no id at all -- and an empty evidence panel under a
    correct answer is worse than showing what was consulted. Conflating them would let an
    uncited claim borrow the authority of a cited one, so the distinction survives all the
    way to the UI label."""
    return {"answer": text.strip(), "refused": refused, "detail": detail,
            "tools_used": tools_used,
            "citations": list(dict.fromkeys(SPEECH_ID_RE.findall(text))),
            "sources": list(dict.fromkeys(sources or [])),
            "usage": usage}


async def _run_anthropic(cfg, model, system, messages, emit) -> dict:
    client = AsyncAnthropic(api_key=cfg.anthropic_api_key)
    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": cfg.llm_effort},
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        tools=TOOLS,
        messages=messages,
        max_iterations=cfg.llm_max_iterations,
        # Safety classifiers can decline a benign request; without a fallback the turn just
        # stops. "default" routes by refusal category rather than pinning a model, so there
        # is no fallback-model migration to own later.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    final, tools_used = None, []
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
    async for message in runner:
        final = message
        for block in message.content:
            if block.type == "text" and block.text.strip():
                await emit("message", text=block.text)
            elif block.type == "tool_use":
                tools_used.append(block.name)
                await emit("tool_use", tool=block.name, input=block.input)
        u = getattr(message, "usage", None)
        if u:
            for k in usage:
                usage[k] += getattr(u, k, 0) or 0

    # A refusal is a successful HTTP response with empty or partial content -- reading
    # content[0] unconditionally is how this crashes instead of reporting.
    if final is not None and getattr(final, "stop_reason", None) == "refusal":
        detail = getattr(getattr(final, "stop_details", None), "explanation", None)
        return _finish("", tools_used, usage, refused=True, detail=detail)

    text = "".join(b.text for b in (final.content if final else []) if b.type == "text")
    return _finish(text, tools_used, usage)


async def _run_openai_compatible(cfg, model, system, messages, emit) -> dict:
    """Hand-written tool loop for OpenAI-shaped endpoints (NVIDIA NIM, vLLM, Ollama).

    No tool-runner helper exists on this path, so the loop is explicit: call, execute any
    tool_calls, append the results, repeat until the model answers without calling a tool.
    Bounded by llm_max_iterations for the same reason the Anthropic runner is -- a model
    that keeps searching and never concludes should stop, not spend.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=cfg.nvidia_api_key, base_url=cfg.nvidia_base_url)
    msgs = [{"role": "system", "content": system}] + messages
    tools, tools_used, sources = _openai_tools(), [], []
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
    text = ""

    for _ in range(max(1, cfg.llm_max_iterations)):
        r = await client.chat.completions.create(
            model=model, messages=msgs, tools=tools,
            max_tokens=cfg.llm_max_tokens, temperature=0.2)
        if getattr(r, "usage", None):
            usage["input_tokens"] += r.usage.prompt_tokens or 0
            usage["output_tokens"] += r.usage.completion_tokens or 0

        m = r.choices[0].message
        calls = m.tool_calls or []
        # Some OpenAI-compatible endpoints reject an assistant turn carrying more than one
        # tool_call -- NVIDIA's llama-3.1 answers "This model only supports single
        # tool-calls at once" with a 400, which kills the turn on the *next* request rather
        # than the one that fanned out. Serialising costs an extra hop and works on every
        # provider; the model asks for the rest on the following iteration.
        if len(calls) > 1:
            calls = calls[:1]
        if m.content and m.content.strip():
            text = m.content
            await emit("message", text=m.content)

        # Echo the assistant turn back explicitly rather than via model_dump(): the
        # response object carries fields (refusal, audio, annotations) that some
        # OpenAI-compatible servers reject on the way back in.
        msgs.append({"role": "assistant", "content": m.content or "",
                     **({"tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name,
                                                      "arguments": c.function.arguments}}
                                        for c in calls]} if calls else {})})
        if not calls:
            # A reasoning model that exhausts its budget mid-thought returns content=null
            # with no tool calls. Returning that as the final answer would hand the caller
            # an empty string that looks like a considered response.
            if not text.strip():
                return _finish("", tools_used, usage, sources=sources, detail=(
                    "model returned no content and called no tools -- if it is a reasoning "
                    "model, raise LLM_MAX_TOKENS; its budget went on reasoning"))
            return _finish(text, tools_used, usage, sources=sources)

        for c in calls:
            name = c.function.name
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args, result = {}, f"Could not parse arguments for {name}."
            else:
                result = await _dispatch(name, args)
            tools_used.append(name)
            # Every id the tool put in front of the model, in the order it saw them.
            sources.extend(SPEECH_ID_RE.findall(result))
            await emit("tool_use", tool=name, input=args)
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": result})

    # Out of iterations with the model still calling tools. Return what it last said rather
    # than nothing, and say plainly that the answer is truncated.
    return _finish(text, tools_used, usage, sources=sources,
                   detail="stopped at llm_max_iterations")
