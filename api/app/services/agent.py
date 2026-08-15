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
async def search_speech(query: str, spoken_by: str | None = None, limit: int = 8) -> str:
    """Find speech in the corpus by topic. Start here for almost any question.

    Searches both pipeline-processed utterances and F1 team-radio calls. Returns one line
    per hit: the speech id, who said it, the session and lap where known, and a clipped
    quote. Call expand_speech on the ids that matter before drawing a conclusion.

    Search the TOPIC, not the person. A name in `query` matches speech that *mentions*
    that name, which is usually somebody else talking about them — to find what a person
    said, put the subject in `query` and the person in `spoken_by`.

    Args:
        query: The topic to look for, in natural language — "power failure", "tyre wear".
            Do not put a driver or speaker name here unless you want speech that mentions
            them.
        spoken_by: Restrict to one speaker: a driver code ("ALB"), a driver number ("23"),
            or an enrolled speaker's name. Omit to search everyone.
        limit: How many hits to return. Default 8.
    """
    # Over-fetch before filtering: the speaker filter is applied after ranking, so asking
    # for exactly `limit` would return almost nothing once a filter is set.
    anchors = await search_svc.anchor_speech(query, limit=limit * 6 if spoken_by else limit)
    if not anchors:
        return "No matching speech found."
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

    if is_placeholder(spoken_by):
        spoken_by = None

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
            return (f'No speech by "{spoken_by}" matching {query!r}. The speaker may not be '
                    f"enrolled, or that driver has no analyzed radio in the corpus.")
    return graph_context.render_brief(rows[:limit]) or "No matching speech found."


@beta_async_tool
async def expand_speech(speech_id: str) -> str:
    """Show everything the graph knows around one piece of speech.

    Who said it, in which session, the lap it happened on and that lap's time against the
    previous one, the drivers or teams it names, and the lines immediately before and
    after. Call this before answering a question about a specific moment — search results
    alone omit the surroundings that usually decide what a quote actually means.

    Args:
        speech_id: An id from search_speech, of the form
            "3f2b1a4c-...". Ids appear in square brackets at the start of each result.
    """
    rows = await graph_context.expand([speech_id])
    if not rows:
        return (f"No speech with id {speech_id}. Ids come from search_speech; the graph "
                f"may also be out of date if it has not been re-synced.")
    return graph_context.render_context(rows)


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
async def compare_speakers(profile_id_a: str, profile_id_b: str) -> str:
    """Contrast two enrolled speakers: how much they talk, and how they come across.

    Use this for questions that put two people side by side. It reports counts and
    aggregate readings only — for what was actually said, search instead.

    Args:
        profile_id_a: A speaker profile id from the directory in your context.
        profile_id_b: The other speaker profile id.
    """
    rows = await db.fetch(COMPARE_SPEAKERS_SQL, [profile_id_a, profile_id_b])
    if not rows:
        return "Neither profile id was found. Ids come from the speaker directory."
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

Say when the corpus cannot answer. Speaker identification abstains rather than guess, so \
plenty of speech is attributed to nobody. Lap alignment needs a recording timestamp that \
manually uploaded audio does not have. If the data does not support an answer, say which \
part is missing instead of filling the gap.

Keep responses brief and specific. Lead with the answer, then the evidence. Do not restate \
the question, do not narrate which tools you are about to call, and do not pad with caveats \
beyond the ones above."""


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
