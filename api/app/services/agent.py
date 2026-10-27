"""The agentic layer: an LLM that answers questions about the corpus by querying it.

Tools, not a pre-stuffed context blob (AGENT_LAYER_PLAN.md). Handing the model the top-k
search results and asking the question fails as soon as an answer needs two hops, because
retrieval cannot know what to fetch until after the model has reasoned about the first
result. Five read-only tools plus the SDK's tool runner is less code than a query planner
and is the only shape that uses the graph for what a graph is for.

Every tool returns *rendered text* carrying stable ids, not JSON, so the model always has
an id in front of it to cite.

Nothing here writes. The graph and the corpus are read-only to the agent; corrections stay
in the existing human-in-the-loop UI.
"""
import json
import re

from anthropic import APIError as AnthropicAPIError, AsyncAnthropic, beta_async_tool

from common import db, graph
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
         round(sum(u.end_s - u.start_s)::numeric, 1) AS spoken_s
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
       utt.spoken_s, share.avg_talk_share
FROM speaker_profiles p
LEFT JOIN utt   ON utt.profile_id = p.id
LEFT JOIN share ON share.profile_id = p.id
WHERE p.id = ANY($1::uuid[])
"""


# Values a model sends when it means "no filter". Exact matches only -- a prefix or
# substring rule would swallow real names ("Norris" starts like "no").
PLACEHOLDER_ARGS = frozenset({"", "null", "none", "undefined", "nil", "n/a", "na", "-"})


# Question words and fillers a model leaves in when it pastes the user's question into
# `query` ("whats dunzo", "what did they say about churn"). They matter in two places:
# websearch_to_tsquery ANDs every word, so "whats" alone sinks the keyword leg; and the
# relevance check below must not count a hit as on-topic because it contains "what".
QUERY_STOPWORDS = frozenset("""
a an and are about any did does do for from had has have how is it its of on or said say
says the their there these they this those to was were what whats what's when where which
who whom whose why with tell me us anything something talk talked talking mention mentioned
discuss discussed corpus clip clips recording recordings audio speech
""".split())


def query_terms(text: str) -> list[str]:
    """The content words of a query: what a hit has to contain to be *about* it."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return list(dict.fromkeys(w for w in words if len(w) > 2 and w not in QUERY_STOPWORDS))


def mentions_any(text: str, terms: list[str]) -> bool:
    """Whether a line contains any query term. Substring rather than token match on
    purpose: the transcript says "Dunzo's" and the query says "dunzo"."""
    low = (text or "").lower()
    return any(t in low for t in terms)


def is_placeholder(value: str | None) -> bool:
    """True when an optional string argument is really absent.

    Models routinely send an omitted optional argument as the *string* "null" rather than
    omitting it or sending JSON null. Taken literally that filters results down to a
    speaker who cannot exist, and the tool answers "there is no speech by null" -- which
    the user reads as a statement about the corpus rather than a malformed tool call.
    Observed in a real session."""
    return value is None or value.strip().lower() in PLACEHOLDER_ARGS


# Upper bound on one expand_speech result, in characters (~1.5k tokens).
EXPAND_MAX_CHARS = 6000


class LLMUnavailable(RuntimeError):
    """The agent is disabled or unconfigured. Callers turn this into a 503 — a deployment
    state, not a bug in the request."""


class LLMRequestFailed(RuntimeError):
    """The provider refused or failed the request after the loop's own recovery: a rate or
    size limit it could not get under, an outage, a bad model id. Callers turn this into a
    502 with the provider's reason -- it used to surface as a bare 500, which tells the
    person asking nothing about whether to retry, wait, or change the model."""


# --------------------------------------------------------------------------- tools
# Tool descriptions are the highest-leverage text in this file. Each says *when* to reach
# for it, not just what it does: on current models the most common failure is a tool that
# never gets called because its description only described itself.

@beta_async_tool
async def search_speech(query: str, spoken_by: str | None = None, limit: int = 8) -> str:
    """Find speech in the corpus by topic. Start here for almost any question.

    Returns one line per hit: the speech id, who said it, the recording and offset, and a
    clipped quote. Call expand_speech on the ids that matter before drawing a conclusion.

    Search the TOPIC, not the person. A name in `query` matches speech that *mentions*
    that name, which is usually somebody else talking about them — to find what a person
    said, put the subject in `query` and the person in `spoken_by`.

    Put the key words of the topic in `query`, not the whole question: "dunzo funding",
    not "what happened with dunzo's funding". If nothing relevant comes back, search
    again with fewer or different words (a synonym, a shorter form, another spelling of a
    name — transcription mangles names) before concluding the corpus does not cover it.

    Args:
        query: The topic to look for, as its key words — "power failure", "churn",
            "dunzo". Do not put a speaker's name here unless you want speech that
            mentions them.
        spoken_by: Restrict to one enrolled speaker, by name. Omit to search everyone.
        limit: How many hits to return. Default 8, capped at 25.
    """
    if is_placeholder(spoken_by):
        spoken_by = None
    limit = max(1, min(int(limit or 8), 25))
    topic = None if is_placeholder(query) else query.strip()
    if not topic:
        return "Give a topic in `query`. An empty search has nothing to rank."

    # Over-fetch before filtering: the speaker filter is applied after ranking, so asking
    # for exactly `limit` would return almost nothing once a filter is set.
    over = limit * 6 if spoken_by else limit
    note = ""
    terms = query_terms(topic)
    anchors = await search_svc.anchor_speech(topic, limit=over)
    # websearch_to_tsquery ANDs every word, so one filler word ("whats dunzo") empties the
    # keyword leg. Retrying as an OR over the content words is the same search a person
    # would run next; it runs only when the literal query found nothing, and says so.
    if not anchors and len(terms) > 1:
        broad = " or ".join(terms)
        anchors = await search_svc.anchor_speech(broad, limit=over)
        if anchors:
            note = (f"Nothing matched {topic!r} as a whole; these match any of "
                    f"{', '.join(terms)}:\n")

    if not anchors:
        return (f"No speech found for {topic!r}. Before concluding the corpus does not cover "
                f"this, try other words for it.")
    try:
        # One extra round trip to attach speaker and clip to each hit. resolve(), not
        # expand(): expand() reads only the graph, so on a stale or switched-off graph
        # every hit lost its speaker and clip. Resolved rows are layered *over* the anchor,
        # so ranking order is preserved and an unresolvable id keeps its text.
        by_id = {r["speech_id"]: r for r in
                 await graph_context.resolve([a["speech_id"] for a in anchors])}
        rows = [{**a, **by_id.get(a["speech_id"], {})} for a in anchors]
    except Exception:
        # Both stores unreachable: degrade to what search alone knows rather than failing
        # the tool. A hit without its speaker is still a hit.
        rows = anchors

    if spoken_by:
        want = spoken_by.strip().lower()

        def is_match(r: dict) -> bool:
            # Both directions: "Marta" should find "Marta Ruiz", and "Marta Ruiz" "Marta".
            have = str(r.get("speaker") or "").strip().lower()
            return bool(have) and (want in have or have in want)

        rows = [r for r in rows if is_match(r)]
        if not rows:
            return (f'No speech by "{spoken_by}" matching {topic!r}. The speaker may not be '
                    f"enrolled under that name; the enrolled speakers are listed in your "
                    f"context.")
    rows = rows[:limit]
    # The vector leg of the hybrid ranking has no similarity floor: it returns the nearest
    # lines for *any* query, including one about something the corpus never mentions. A
    # list of plausible-looking quotes is then indistinguishable from a real answer, and a
    # model either builds one out of unrelated lines or -- seeing they are off-topic --
    # concludes absence without trying another wording. Saying which case this is, from
    # the text itself, lets it do neither.
    if terms and not any(mentions_any(r.get("text"), terms) for r in rows):
        note += (f"No line below contains {' or '.join(repr(t) for t in terms)}: these are "
                 f"only the closest in meaning and may be unrelated. Try another wording "
                 f"before concluding the corpus does not cover it, and do not cite them as "
                 f"evidence.\n")
    # 240 rather than the renderer's 160: uploaded clips diarize into long turns, and the
    # window is centred on the query's words, so a little more of it is the part that
    # answers. Eight hits at 240 is still well under a kilobyte of quotes per call.
    return note + (graph_context.render_brief(rows, max_text=240, terms=terms)
                   or "No matching speech found.")


@beta_async_tool
async def expand_speech(speech_id: str) -> str:
    """Show everything the graph knows around one piece of speech, or several at once.

    Who said it, in which recording, the names it mentions, what the recording is about,
    and the lines immediately before and after. Call this before answering a question
    about a specific moment — search results alone omit the surroundings that usually
    decide what a quote actually means.

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
    # Capped on a block boundary. Ten long podcast turns, each with the turns either side,
    # ran to a request one provider tier rejects outright (Groq on-demand: 8000 tokens per
    # minute, counted per request); a cut block is named by the "Not shown" line below.
    out = (graph_context.render_context(rows, max_chars=EXPAND_MAX_CHARS)
           # one block over the cap on its own is still shown: an empty result would
           # read as "no such speech"
           or graph_context.render_context(rows[:1]))
    shown = set(SPEECH_ID_RE.findall(out))
    cut = [r["speech_id"] for r in rows if r["speech_id"] not in shown]
    if cut:
        out += (f"\n\nNot shown, to keep this result small: {', '.join(cut)}. "
                f"Expand them in a separate call if they matter.")
    if missing:
        out += f"\n\nNot in the graph: {', '.join(missing)}."
    return out


@beta_async_tool
async def compare_speakers(profile_id_a: str, profile_id_b: str | None = None) -> str:
    """Contrast enrolled speakers: how much they talk and in how many recordings.

    Use this for questions that put people side by side. It reports counts only — for
    what was actually said, search instead.

    Two speakers is the common case, but either argument accepts a comma-separated list,
    so "how do all the engineers compare" is one call rather than a matrix of pairs.
    Names work as well as ids: a name is resolved against the speaker directory, because a
    question about "Marta" should not fail on the model not having copied a UUID.

    Args:
        profile_id_a: A speaker profile id or display name, several separated by commas,
            or "all" for every enrolled speaker.
        profile_id_b: The other speaker, when comparing exactly two. Omit when the first
            argument already lists everyone to compare.
    """
    tokens = [t.strip() for arg in (profile_id_a, profile_id_b)
              if not is_placeholder(arg) for t in str(arg).split(",") if t.strip()]
    if not tokens:
        return "Name at least one speaker, by profile id or display name."

    if {t.lower() for t in tokens} & {"all", "everyone", "everybody", "*"}:
        # "how do the speakers compare" arrives as profile_id_a="all"; failing it as an
        # unknown name sent the model off to search instead.
        tokens = [str(r["id"]) for r in await db.fetch("SELECT id FROM speaker_profiles")]
        if not tokens:
            return "No speakers are enrolled, so there is no one to compare."

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
            + (f", {r['spoken_s']}s of speech" if r["spoken_s"] is not None else "")
            + (f", mean talk share {r['avg_talk_share']}" if r["avg_talk_share"] is not None else ""))
    return "\n".join(out)


# Exact id or filename first, then substring; newest first among the rest. strpos rather
# than ILIKE so the "_" every uploaded filename is full of is not a wildcard.
RECORDING_MATCH_SQL = """
SELECT id, filename, duration_s FROM clips
 WHERE status = 'COMPLETE'
   AND (id::text = $1 OR strpos(lower(filename), lower($1)) > 0)
 ORDER BY (id::text = $1 OR lower(filename) = lower($1)) DESC, created_at DESC
 LIMIT 8
"""

# Speaker resolution matches graph_context.RESOLVE_UTTERANCES_SQL: clip_speakers is the
# authoritative identity row and utterances.profile_id can lag it.
RECORDING_LINES_SQL = """
SELECT u.id::text AS speech_id, u.start_s, u.text, u.local_label, p.display_name AS speaker
  FROM utterances u
  LEFT JOIN clip_speakers cs ON cs.clip_id = u.clip_id AND cs.local_label = u.local_label
  LEFT JOIN speaker_profiles p ON p.id = coalesce(u.profile_id, cs.profile_id)
 WHERE u.clip_id = $1 AND u.end_s > $2
 ORDER BY u.idx
"""


@beta_async_tool
async def read_recording(recording: str, from_minute: float = 0) -> str:
    """Read one recording's transcript in order, from the start or from a given minute.

    Use this when the question is about a whole recording rather than a topic inside the
    corpus: "summarise the Dunzo case study", "what happened in Tuesday's standup", "what
    did they cover after the funding part". search_speech returns the few lines that match
    words; it cannot tell you what a recording is about overall, or what came between two
    hits. The recordings are listed by filename in your context.

    Long recordings come back in pages of a few thousand characters. The result says where
    it stopped; call again with that minute to read on, and stop once you have enough to
    answer -- a summary does not need every line.

    Args:
        recording: The recording's filename, or a distinctive part of it ("Dunzo",
            "standup"), or its clip id.
        from_minute: Where to start reading, in minutes from the start of the recording.
            Default 0.
    """
    if is_placeholder(recording):
        return "Name a recording, by filename or part of it. They are listed in your context."
    want = recording.strip()
    clips = await db.fetch(RECORDING_MATCH_SQL, want)
    if not clips:
        return (f"No processed recording matches {want!r}. Use a word from one of the "
                f"filenames listed in your context.")
    exact = want.lower() in (str(clips[0]["id"]), clips[0]["filename"].lower())
    if len(clips) > 1 and not exact:
        return (f"{len(clips)} recordings match {want!r}; name one more precisely:\n"
                + "\n".join(f"  {c['filename']}" for c in clips))
    clip = clips[0]
    start = max(0.0, float(from_minute or 0)) * 60
    rows = await db.fetch(RECORDING_LINES_SQL, clip["id"], start)
    length = f"{float(clip['duration_s']) / 60:.1f} min" if clip["duration_s"] else "length unknown"
    head = f'Recording "{clip["filename"]}" ({length}), from {graph_context._clock(start)}:'
    if not rows:
        return head + " no speech from this point on."

    out, used = [head], len(head)
    for n, r in enumerate(rows):
        # An unidentified speaker keeps the diarizer's label: within one recording it is
        # still the difference between one person talking and two people disagreeing.
        who = r["speaker"] or (f"unidentified ({r['local_label']})" if r["local_label"]
                               else "unidentified speaker")
        line = f"[{r['speech_id']}] {graph_context._clock(r['start_s'])} {who}: {r['text']}"
        if n and used + len(line) > EXPAND_MAX_CHARS:
            out.append(f"\n{len(rows) - n} more lines. Continues at "
                       f"{graph_context._clock(r['start_s'])}: call read_recording again "
                       f"with from_minute={r['start_s'] / 60:.2f} to read on.")
            break
        out.append(line)
        used += len(line)
    return "\n".join(out)


# A name looked up across every kind of thing the graph holds. Exact (case-insensitive)
# before substring, so "Dunzo" finds the entity before a filename that contains it.
RELATED_LOOKUP_CYPHER = """
MATCH (n) WHERE NOT n:Utterance
  AND toLower(coalesce(n.name, n.filename, '')) CONTAINS toLower($q)
WITH n, toLower(coalesce(n.name, n.filename, '')) = toLower($q) AS exact
ORDER BY exact DESC, COUNT { (n)--() } DESC
LIMIT 1
OPTIONAL MATCH (n)-[r]-(m) WHERE NOT m:Utterance
WITH n, r, m ORDER BY coalesce(r.n_clips, r.n, r.speech_s, 1) DESC
RETURN [l IN labels(n) WHERE l <> 'Speech'][0] AS label,
       coalesce(n.name, n.filename) AS name, n.kind AS kind,
       collect(CASE WHEN m IS NULL THEN null ELSE {
         label: [l IN labels(m) WHERE l <> 'Speech'][0],
         name: coalesce(m.name, m.filename), rel: type(r),
         weight: coalesce(r.n_clips, r.n, 1)
       } END)[..40] AS links,
       [(u:Utterance)-[:MENTIONS|SPOKEN_BY]->(n) | u {.speech_id, .text}][..4] AS lines
"""

# How each relationship reads from the looked-up node's side, for the rendered result.
REL_PHRASES = {
    "APPEARS_IN": "appears with", "HEARD_IN": "voice heard with", "TALKED_WITH": "shares recordings with",
    "DISCUSSED": "linked by topic to", "ABOUT": "topic link with", "TAGGED": "tagged",
    "IN_LANGUAGE": "language", "MENTIONS": "mention link with", "MENTIONED": "mention link with",
    "CO_MENTIONED": "named alongside",
}


@beta_async_tool
async def related(name: str) -> str:
    """Look someone or something up in the knowledge graph and list what it connects to.

    Use this for relationship questions the transcript text cannot answer by search alone:
    "who does Marta talk to most", "which recordings mention Reliance", "what topics come
    up with Dunzo", "who talks about funding". It finds the best-matching node — an
    enrolled speaker, a recording, a topic, a tag, or a named person, company or place
    pulled from the transcripts — and lists its neighbours, strongest first, with counts.

    The links are aggregates, not quotes: to say *what* was said, follow up with
    search_speech and cite speech ids from there.

    Args:
        name: The name to look up — a speaker's name, part of a recording's filename, a
            topic, or a named entity like "Reliance".
    """
    if is_placeholder(name):
        return "Give a name to look up."
    try:
        rows = await graph.run(RELATED_LOOKUP_CYPHER, q=name.strip())
    except Exception as e:
        return (f"The knowledge graph is not available ({type(e).__name__}); use "
                f"search_speech instead.")
    if not rows:
        return (f"Nothing in the knowledge graph is called {name!r}. Named entities only "
                f"exist once the transcripts mention them more than once; try search_speech.")
    r = rows[0]
    kind = f", {r['kind']}" if r.get("kind") and r["kind"] != "name" else ""
    out = [f"{r['label']} \"{r['name']}\"{kind}:"]
    for link in r["links"]:
        weight = link["weight"]
        count = f" ({int(weight)}x)" if isinstance(weight, (int, float)) and weight > 1 else ""
        out.append(f"  {REL_PHRASES.get(link['rel'], link['rel'].lower())} "
                   f"{link['label']} \"{link['name']}\"{count}")
    if len(out) == 1:
        out.append("  no connections yet")
    # The links are counts with nothing to cite; these lines are what make a claim about
    # them checkable without a second call.
    if r.get("lines"):
        out.append("Lines that name it or are spoken by it (cite these ids):")
        out += [f"  [{u['speech_id']}] \"{graph_context._snippet(u['text'] or '', 160, [name])}\""
                for u in r["lines"]]
    else:
        out.append("No speech ids attached: call search_speech for the quotes before citing.")
    return "\n".join(out)


TOOLS = [search_speech, expand_speech, read_recording, related, compare_speakers]

# The tool layer is the provider seam, exactly as designed: these functions are plain
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
    # Same OpenAI-shaped loop as nvidia, different host and an order of magnitude more
    # tokens per second — the reason the provider is worth having rather than a base-URL
    # override on the nvidia one.
    "groq": "llama-3.3-70b-versatile",
    # Verified to emit well-formed tool_calls against this corpus; meta/llama-3.1-70b-instruct
    # also works if this one is unavailable.
    # Non-reasoning on purpose. The nemotron-super reasoning models on this endpoint
    # return content=null plus a separate `reasoning` field and spend the whole token
    # budget thinking before emitting anything, which turns a tool loop into minutes
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


def _tool_name(raw) -> str:
    """The tool a model meant, from the name it actually emitted.

    On the OpenAI-shaped path the name is free text the model wrote. gpt-oss in particular
    leaks its harmony-format scaffolding into it -- "functions.search_speech", or
    "search_speech<|channel|>commentary" -- and an exact dict lookup turns that into "No
    tool named ...", which the model reads as the tool not existing."""
    name = re.split(r"<\|", str(raw or ""))[0].strip()
    return name.rsplit(".", 1)[-1].strip()


def _parse_args(raw) -> tuple[dict | None, str | None]:
    """Tool-call arguments -> (dict, None), or (None, an error the model can act on).

    `function.arguments` is a JSON *string* the model wrote, and every malformation below
    has been produced by some OpenAI-compatible endpoint: empty or "null" for a call with
    no arguments, an object encoded twice (a JSON string containing JSON), and a valid
    object followed by trailing text. The first three are unambiguous and are accepted; an
    error comes back only for something that genuinely is not an object."""
    if isinstance(raw, dict):
        return raw, None
    text = str(raw or "").strip()
    if not text or text.lower() in ("null", "none", "{}"):
        return {}, None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        try:
            value = json.JSONDecoder().raw_decode(text[start:])[0] if start >= 0 else None
        except json.JSONDecodeError:
            value = None
        if value is None:
            return None, (f"Could not parse the arguments {text[:200]!r} as JSON. Send a "
                          f"JSON object, e.g. {{\"query\": \"churn\"}}.")
    if isinstance(value, str):  # double-encoded
        return _parse_args(value)
    if value is None:
        return {}, None
    if not isinstance(value, dict):
        return None, f"Arguments must be a JSON object, got {type(value).__name__}."
    return value, None


# Scaffolding small models wrap a written-out call in: Hermes/Qwen <tool_call> tags, Llama
# 3.1's <|python_tag|>, and markdown fences. Stripped before the text is read as a call.
TEXT_CALL_NOISE = re.compile(
    r"<\|?/?tool_call\|?>|<\|python_tag\|>|<\|eom_id\|>|<\|eot_id\|>|```[a-z_]*", re.I)

# Gemma's native call syntax, which nemotron-3.5-bolt also falls into for some questions:
# `call:search_speech{query:<|"|>stressed<|"|>}` -- bare keys, strings fenced by <|"|>.
GEMMA_CALL = re.compile(r"(?:call:)?([\w.]+)\{(.*?)\}", re.S)


def _text_tool_calls(content: str | None) -> list[tuple[str, dict]]:
    """Tool calls a model wrote into its message text instead of `tool_calls`.

    Some OpenAI-compatible endpoints do not parse tool calls out of every model's output.
    nvidia/nemotron-3.5-bolt-e4b on NVIDIA NIM answers every request -- tool_choice
    "required" included -- with the literal text `search_speech(query="swiggy")` and
    tool_calls=null. The loop read that as a final answer and showed the user the call
    instead of running it. Read back as a call, the same model answers correctly from the
    results, with citations.

    Strict on purpose: the whole message has to be calls to tools that exist, in one of
    the shapes models actually emit -- Python-style `name(k="v")` (one per line, or in a
    list), a JSON {"name", "arguments"|"parameters"} object or list of them, or Llama's
    <function=name>{...}</function>. Anything else is an answer, including an answer that
    happens to mention a tool by name."""
    import ast

    text = TEXT_CALL_NOISE.sub("\n", content or "").strip()
    if not text:
        return []

    def known(name) -> str | None:
        name = _tool_name(name)
        return name if name in BY_NAME else None

    # <function=search_speech>{"query": "x"}</function>
    tagged = re.findall(r"<function=([\w.]+)>(.*?)</function>", text, re.S)
    if tagged and not re.sub(r"<function=[\w.]+>.*?</function>", "", text, flags=re.S).strip():
        out = []
        for name, body in tagged:
            args, error = _parse_args(body)
            if not known(name) or error:
                return []
            out.append((known(name), args))
        return out

    # call:search_speech{query:<|"|>stressed<|"|>}
    if "<|\"|>" in text or text.startswith("call:") or re.match(r"[\w.]+\{", text):
        gemma = GEMMA_CALL.findall(text)
        if gemma and not GEMMA_CALL.sub("", text).strip():
            out = []
            for name, body in gemma:
                # Odd segments are string values; keys are quoted only outside them, so a
                # query like "tyres: worn, again" is not split into keys.
                parts = body.split('<|"|>')
                body = "".join(json.dumps(p) if n % 2 else
                               re.sub(r'(^|[,{])\s*([A-Za-z_]\w*)\s*:', r'\1"\2":', p)
                               for n, p in enumerate(parts))
                args, error = _parse_args("{" + body + "}")
                if not known(name) or error:
                    return []
                out.append((known(name), args))
            return out

    # {"name": "search_speech", "arguments": {...}}, or a list of those
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if value is not None:
        out = []
        for item in value if isinstance(value, list) else [value]:
            if not isinstance(item, dict):
                return []
            item = item.get("function", item) if isinstance(item.get("function"), dict) else item
            args, error = _parse_args(item.get("arguments", item.get("parameters")))
            if not known(item.get("name")) or error:
                return []
            out.append((known(item.get("name")), args))
        return out

    # search_speech(query="x") -- one per line, or [a(...), b(...)]
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    nodes = []
    for stmt in tree.body:
        if not isinstance(stmt, ast.Expr):
            return []
        v = stmt.value
        nodes += list(v.elts) if isinstance(v, (ast.List, ast.Tuple)) else [v]
    out = []
    for node in nodes:
        if not isinstance(node, ast.Call):
            return []
        name = known(ast.unparse(node.func))
        if not name:
            return []
        params = list(BY_NAME[name].to_dict()["input_schema"].get("properties", {}))
        try:
            args = {params[n]: ast.literal_eval(a) for n, a in enumerate(node.args)}
            args.update({k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg})
        except (ValueError, IndexError, SyntaxError):
            return []
        out.append((name, args))
    return out


async def _dispatch(name: str, args: dict) -> str:
    """Run one tool by name. A bad name or bad arguments come back as text for the model to
    read and correct, rather than as an exception that kills the turn -- a model that
    mis-calls a tool should get a chance to try again."""
    name = _tool_name(name)
    tool = BY_NAME.get(name)
    if tool is None:
        return f"No tool named {name!r}. Available: {', '.join(sorted(BY_NAME))}."
    schema = tool.to_dict().get("input_schema") or {}
    args = args if isinstance(args, dict) else {}
    # An argument the tool does not take is dropped and named, rather than raising
    # "unexpected keyword argument" and losing the whole call: models add plausible extras
    # ("mode": "hybrid") that change nothing about what they meant.
    known = set(schema.get("properties", {}))
    unknown = sorted(k for k in args if k not in known)
    args = {k: v for k, v in args.items() if k in known}
    prefix = f"(Ignored unknown argument(s): {', '.join(unknown)}.)\n" if unknown else ""
    try:
        return prefix + await tool.func(**_coerce_args(schema, args))
    except TypeError as e:
        return f"Bad arguments for {name}: {e}"
    except Exception as e:
        # One tool failing is not the turn failing: hand the model the error so it can try
        # a different call, rather than 500-ing a request that may still be answerable.
        return f"{name} failed: {type(e).__name__}: {str(e)[:200]}"


# --------------------------------------------------------------------------- prompt

SYSTEM = """You answer questions about a voice-intelligence corpus: recorded speech of any \
kind — meetings, interviews, podcasts, calls — that has been transcribed, diarized, and \
attributed to speakers where possible, and projected into a knowledge graph of the people, \
recordings, topics and named things in it.

Answer by using your tools. Every question about what is in the recordings — a topic, a \
name, a company, an event, who said what — starts with search_speech, \
even when you think you know the answer or think the corpus cannot contain it. You cannot \
see the transcripts until you search: the inventory at the end of this prompt lists the \
recordings, not what is said in them. Search first, expand what looks relevant, then answer \
from what you actually read. Do not answer from general knowledge about the subject or about \
the people involved; the only thing you know is what the corpus says. A question about one \
recording as a whole — summarise it, what it covers, what came after a point in it — is \
read_recording rather than search: search finds lines that match words, not what a \
recording is about. A question about connections — who talks with whom, which \
recordings mention a company, what a person tends to discuss — is what related answers; \
follow it with search_speech for the quotes to cite.

Absence is a finding, and it has to be earned. Never say the corpus does not mention \
something unless you have searched for it. If a search comes back empty or off-topic, search \
again before concluding: the key noun alone, a synonym, another spelling of a name \
(transcription mangles proper nouns). Only after two differently worded searches find \
nothing relevant, say the corpus does not cover it and say what you searched for. Do not \
build an answer out of hits that are not about the question, and do not cite them: an id \
attached to "nothing was found" presents unrelated speech as evidence.

Two rules govern how you report what you find.

Cite. Every factual claim carries the speech id it came from, in square brackets, exactly \
as the tools return it: a real id, never a placeholder like "[id]". A claim you cannot \
attach an id to is one you should not make. \
Names, numbers and dates are copied from what the tools showed you, never completed from \
memory: if a quote is cut off before the detail you need, call expand_speech on its id to \
read it in full; if the detail is still not there, leave it out.

Say when the corpus cannot answer. Speaker identification abstains rather than guess, so \
plenty of speech is attributed to nobody — "unidentified speaker" is the normal case, not a \
reason to doubt a hit, and the clip it came from is named beside it. If the data does not \
support an answer, say which part is missing instead of filling the gap.

Keep responses brief and specific. Lead with the answer, then the evidence. Do not restate \
the question, do not narrate which tools you are about to call, and do not pad with caveats \
beyond the ones above.

Do not re-quote the evidence. Every id you cite is rendered beside your answer with its \
full quote, speaker and a link to the audio, so repeating those lines in prose adds nothing \
and makes the reader wait for it. Characterise what was said in your own words and attach \
the ids: "three people raised the delivery costs, all in the March call [id] [id] [id]" — not a bulleted \
list of the quotes themselves. Two or three sentences answers most questions.

Brevity never costs citations. A bare count or a summary with no ids attached is not an \
answer — it is the one thing the reader cannot check. Every id behind what you are \
describing goes in the text, however short the text is."""


# Newest first, capped: the inventory is orientation, not an index -- search is the index.
# Past the cap the line count says how many more exist, so the model never reads a
# truncated list as the whole corpus.
ROSTER_CLIPS_SQL = """
SELECT c.filename, c.duration_s, c.tags, c.auto_tags,
       (SELECT count(*) FROM utterances u WHERE u.clip_id = c.id) AS n_utterances,
       count(*) OVER () AS n_total
  FROM clips c
 WHERE c.status = 'COMPLETE'
 ORDER BY c.created_at DESC
 LIMIT $1
"""
ROSTER_MAX_CLIPS = 60


def _clip_line(c) -> str:
    """One inventory line: filename, length, size of transcript, and its topic tags.

    Topic tags only. The other auto tags (lang:, voices:, audio:) describe the recording
    rather than its subject, and would spend tokens on every request to say little."""
    bits = []
    if c["duration_s"]:
        bits.append(f"{float(c['duration_s']) / 60:.1f} min")
    bits.append(f"{c['n_utterances']} utterances")
    topics = [t.split(":", 1)[1] for t in (c["auto_tags"] or []) if t.startswith("topic:")]
    topics += [t for t in (c["tags"] or []) if ":" not in t]
    if topics:
        bits.append("topics: " + ", ".join(dict.fromkeys(topics)))
    return f"  {c['filename']} — " + "; ".join(bits)


async def _roster() -> str:
    """Entity roster, rendered into the cached system prompt rather than exposed as a tool.

    It is a few hundred stable tokens against a tool round-trip on nearly every question,
    and it removes an entire failure class: the model inventing a recording or a speaker
    instead of looking one up.

    The clip inventory comes first, and it is the part that fixed a real failure. This
    roster once listed only enrolled speakers; on a corpus with none it rendered as "the
    corpus contains no data". A model told the corpus is empty believes it: asked "whats
    dunzo" with an eight-minute Dunzo case study sitting in the corpus, it called no tool
    at all and answered that the corpus had nothing on Dunzo. The inventory changes when a
    clip finishes processing, which breaks the prompt-cache prefix at that point; that is
    the correct trade, since a cached prompt describing the wrong corpus is worse than a
    cache miss."""
    parts: list[str] = []

    clips = await db.fetch(ROSTER_CLIPS_SQL, ROSTER_MAX_CLIPS)
    if clips:
        total = clips[0]["n_total"]
        head = (f"Recordings ({total} processed clips, newest first"
                + (f"; the {len(clips)} most recent shown" if total > len(clips) else "")
                + "). Titles and tags only: search to find out what is said in them:")
        parts.append(head + "\n" + "\n".join(_clip_line(c) for c in clips))

    speakers = await db.fetch(
        "SELECT id, display_name FROM speaker_profiles ORDER BY display_name")
    if speakers:
        parts.append("Enrolled speakers (profile id: name):\n" + "\n".join(
            f"  {s['id']}: {s['display_name']}" for s in speakers))

    # Worded as "nothing processed yet", not "the corpus is empty": a model told there is
    # no data stops searching, and a clip may finish while the question is being asked.
    return "\n\n".join(parts) if parts else (
        "No clip has finished processing yet; search before saying the corpus is empty.")


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

# Prior turns kept. The client holds the transcript and sends all of it; past this many
# messages the oldest are the least relevant and the most expensive to keep re-sending.
MAX_HISTORY_MESSAGES = 20


def normalise_history(history) -> list[dict]:
    """Whatever the client sent as prior turns -> provider-ready {role, content} messages.

    Two shapes arrive. The API echoes `history` back as role/content messages, but the Ask
    page builds its own from the turns it renders, as {question, answer} pairs. Those were
    spliced into `messages` verbatim, so every follow-up in a thread sent the provider a
    message with no `role` -- a 400 from the OpenAI-shaped endpoints and from Anthropic
    alike, surfaced to the user as "Could not answer". Both shapes are accepted here
    rather than the UI being made to match, because a stateless API that rejects its own
    client's history is the bug, wherever the mismatch started.

    Only text survives. A prior answer's tool calls and results are not replayed: the ids
    it cited are in its text, and expand_speech can re-fetch any of them for the cost of a
    call, against re-sending every tool result of every earlier turn on every request."""
    out: list[dict] = []

    def text_of(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):  # Anthropic-style content blocks
            return "".join(b.get("text", "") for b in content
                           if isinstance(b, dict) and b.get("type") == "text")
        return ""

    for turn in history or []:
        if not isinstance(turn, dict):
            continue
        if turn.get("role") in ("user", "assistant"):
            pairs = [(turn["role"], text_of(turn.get("content")))]
        else:
            pairs = [("user", text_of(turn.get("question"))),
                     ("assistant", text_of(turn.get("answer")))]
        out += [{"role": r, "content": c} for r, c in pairs if c.strip()]

    out = out[-MAX_HISTORY_MESSAGES:]
    # A conversation opens with the user. Trimming can leave an assistant turn first, and
    # Anthropic rejects that outright.
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


# The backstop for a model that answers without looking. The prompt already says every
# question about the corpus starts with a search; this is what happens when a model ignores
# it, which is exactly the failure that was reported: asked "whats dunzo", gpt-oss-20b
# called no tool and told the user the corpus had nothing on Dunzo, with an eight-minute
# Dunzo case study in it. An answer produced without consulting the corpus is not an answer
# about the corpus, so it is never returned as one -- the model is sent back once to search.
# Once, not until it complies: a second refusal to search is returned as it stands.
SEARCH_FIRST_NUDGE = (
    "You answered without searching the corpus, so that answer is not based on it. You "
    "cannot know what the recordings contain without searching them. Call search_speech "
    "with the key words of my question now, then answer from what it returns, with "
    "citations. If nothing relevant turns up after a differently worded second search, "
    "say plainly that the corpus does not cover it and what you searched for.")

# For a model that read its tool results and then returned nothing.
ANSWER_NUDGE = (
    "You have the tool results above but gave no answer. Answer my question now from "
    "them, citing the speech ids in square brackets. If they do not answer it, say so.")


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
    key = getattr(cfg, f"{provider}_api_key", "")
    if not key:
        raise LLMUnavailable(
            f"no API key for {provider} — set one in Settings › Ask AI, "
            f"or {provider.upper()}_API_KEY in .env")

    system = (await build_system(cfg))[0]["text"]
    messages = normalise_history(history) + [{"role": "user", "content": question}]

    if provider == "anthropic":
        try:
            result = await _run_anthropic(cfg, model, system, messages, emit)
        except AnthropicAPIError as e:
            raise LLMRequestFailed(_provider_message(provider, e)) from e
    else:
        result = await _run_openai_compatible(cfg, model, system, messages, emit, provider)

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


def _result_text(content) -> str:
    """A tool_result's content, which the runner may hand back as a string or as blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


async def _run_anthropic(cfg, model, system, messages, emit) -> dict:
    client = AsyncAnthropic(api_key=cfg.anthropic_api_key)
    final, tools_used, sources, text = None, [], [], ""
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
    convo = list(messages)

    # At most two passes: the second exists only for the search-first backstop. tool_choice
    # cannot do this job here -- {"type": "any"} is incompatible with extended thinking, and
    # on a tool runner it would apply to every iteration, so the model could never stop
    # calling tools to answer.
    for attempt in range(2):
        runner = client.beta.messages.tool_runner(
            model=model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            output_config={"effort": cfg.llm_effort},
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=TOOLS,
            messages=convo,
            max_iterations=cfg.llm_max_iterations,
            # Safety classifiers can decline a benign request; without a fallback the turn
            # just stops. "default" routes by refusal category rather than pinning a model,
            # so there is no fallback-model migration to own later.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

        async for message in runner:
            final = message
            called = False
            for block in message.content:
                if block.type == "text" and block.text.strip():
                    await emit("message", text=block.text)
                elif block.type == "tool_use":
                    called = True
                    tools_used.append(block.name)
                    await emit("tool_use", tool=block.name, input=block.input)
            if called:
                # The runner caches this response and appends it itself, so asking for it
                # here does not run the tools twice. It is the only way to see what the
                # tools returned on this path, and without it `sources` was always empty
                # here: an uncited Anthropic answer had no evidence panel at all, while
                # the same answer from the OpenAI-shaped loop did.
                response = await runner.generate_tool_call_response()
                for r in (response or {}).get("content", []):
                    sources.extend(SPEECH_ID_RE.findall(_result_text(r.get("content"))))
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
        if tools_used or attempt:
            break
        convo = convo + [{"role": "assistant", "content": text.strip() or "(no answer)"},
                         {"role": "user", "content": SEARCH_FIRST_NUDGE}]

    return _finish(text, tools_used, usage, sources=sources)


# Endpoints that reject an assistant turn carrying more than one tool_call. NVIDIA's
# llama-3.1 answers "This model only supports single tool-calls at once" with a 400, which
# kills the turn on the *next* request rather than the one that fanned out. Everywhere else
# a fan-out is run in full: dropping all but the first call silently cost the model an
# iteration per extra call, out of a budget of twelve.
SINGLE_TOOL_CALL_PROVIDERS = frozenset({"nvidia"})


def _is_failed_tool_generation(err: Exception) -> bool:
    """Groq validates a generated tool call server-side and, when the model botches it --
    a name not in `tools`, arguments that are not JSON -- fails the whole request with
    400 `tool_use_failed` instead of returning the call. That is a sampling accident, not
    a bad request: the identical request usually succeeds on a second draw."""
    s = str(err).lower()
    return any(k in s for k in ("tool_use_failed", "failed to call a function",
                                "tool call validation", "parsing tool call"))


def _is_too_large(err: Exception) -> bool:
    """The request exceeds a size limit: the model's context window, or -- on Groq's
    on-demand tier -- a tokens-per-minute cap that one request on its own can exceed
    (8000 TPM for gpt-oss-20b; observed rejecting a single 8733-token request with 413,
    which escaped the loop as an unhandled 500)."""
    s = str(err).lower()
    return getattr(err, "status_code", None) == 413 or any(k in s for k in (
        "request too large", "reduce your message size", "context_length",
        "context length", "maximum context", "too many tokens"))


def _provider_message(provider: str, err: Exception) -> str:
    """A provider failure as one readable line. The key never appears in these errors;
    the body is clipped because some providers echo the whole request back."""
    status = getattr(err, "status_code", None)
    return (f"{provider} rejected the request" + (f" ({status})" if status else "")
            + f": {str(err)[:300]}")


# What a trimmed tool result is replaced with. Lines lead with their speech id, so keeping
# the head of each line keeps every id citable; expand_speech re-reads any of them in full.
COMPACT_MARK = "[trimmed to fit the provider's request limit; expand_speech re-reads any id]"


def _compact(msgs: list[dict]) -> bool:
    """Trim earlier tool results in place, newest batch last. True if anything changed.

    Earlier results first because the model has already read and acted on them; the
    latest batch is what it is about to reason over, so it is trimmed only when nothing
    older is left to trim."""
    tool_idx = [n for n, m in enumerate(msgs) if m.get("role") == "tool"
                and not str(m.get("content", "")).startswith(COMPACT_MARK)]
    if not tool_idx:
        return False
    last_assistant = max((n for n, m in enumerate(msgs) if m.get("role") == "assistant"),
                         default=-1)
    older = [n for n in tool_idx if n < last_assistant]
    for n in older or tool_idx:
        lines = str(msgs[n]["content"]).splitlines()
        kept = [ln[:140] for ln in lines if ln.strip()][:15]
        msgs[n] = {**msgs[n], "content": COMPACT_MARK + "\n" + "\n".join(kept)}
    return True


async def _run_openai_compatible(cfg, model, system, messages, emit, provider="nvidia") -> dict:
    """Hand-written tool loop for OpenAI-shaped endpoints (Groq, NVIDIA NIM, vLLM, Ollama).

    No tool-runner helper exists on this path, so the loop is explicit: call, execute any
    tool_calls, append the results, repeat until the model answers without calling a tool.
    Bounded by llm_max_iterations for the same reason the Anthropic runner is -- a model
    that keeps searching and never concludes should stop, not spend.

    The first request of a turn is sent with tool_choice="required", and so is every
    request until some tool has run. That is the fix for the reported failure, where
    gpt-oss answered "whats dunzo" from nothing; it costs one search on a question that
    did not need one, against an answer about the corpus that never looked at it. Once a
    tool has run, tool_choice goes back to "auto" so the model can stop and answer. An
    endpoint that rejects "required" is retried with "auto", and the search-first nudge is
    the backstop there.
    """
    import openai
    from openai import AsyncOpenAI

    # max_retries above the SDK's 2: a multi-hop turn on a tokens-per-minute tier hits 429
    # mid-loop, and the SDK's retry honours the provider's retry-after. Failing the whole
    # turn on the fourth search because the minute had not rolled over wastes the first three.
    client = AsyncOpenAI(api_key=getattr(cfg, f"{provider}_api_key"),
                         base_url=getattr(cfg, f"{provider}_base_url"), max_retries=4)
    msgs = [{"role": "system", "content": system}] + messages
    tools, tools_used, sources = _openai_tools(), [], []
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
    text = ""
    force_supported, nudged, answer_nudged, retries = True, False, False, 0

    max_tokens = cfg.llm_max_tokens

    def relieve() -> bool:
        """One step of making the next request smaller; False once there is nothing left.

        Ordered by what each costs the answer. The output reservation goes first: Groq
        counts max_tokens into the request's size, and a 4096 reservation for a
        few-hundred-token answer is most of an 8000 TPM budget. Older evidence is trimmed
        next (ids kept), and the reservation shrinks again only after that."""
        nonlocal max_tokens
        if max_tokens > 2048:
            max_tokens = 2048
            return True
        if _compact(msgs):
            return True
        if max_tokens > 1024:
            max_tokens = 1024
            return True
        return False

    iterations = max(1, cfg.llm_max_iterations)
    i = 0
    while i < iterations:
        i += 1
        force = force_supported and not tools_used
        try:
            r = await client.chat.completions.create(
                model=model, messages=msgs, tools=tools,
                tool_choice="required" if force else "auto",
                max_tokens=max_tokens, temperature=0.2)
        except openai.APIStatusError as e:
            if force and e.status_code == 400 and "tool_choice" in str(e).lower():
                force_supported = False
                i -= 1  # the endpoint's limitation, not a model step
                continue
            if _is_failed_tool_generation(e) and retries < 2:
                retries += 1
                continue
            if _is_too_large(e) and relieve():
                i -= 1
                continue
            raise LLMRequestFailed(_provider_message(provider, e)) from e
        except openai.APIError as e:  # connection failures, timeouts
            raise LLMRequestFailed(_provider_message(provider, e)) from e
        if getattr(r, "usage", None):
            usage["input_tokens"] += r.usage.prompt_tokens or 0
            usage["output_tokens"] += r.usage.completion_tokens or 0

        m = r.choices[0].message
        content = m.content or ""
        calls = [(c.id, c.function.name, c.function.arguments) for c in (m.tool_calls or [])]
        if not calls and (written := _text_tool_calls(content)):
            # The call arrived as text (see _text_tool_calls). It is echoed back as a real
            # tool call, and the text is dropped: it is neither narration nor an answer.
            calls = [(None, name, json.dumps(args)) for name, args in written]
            content = ""
        if len(calls) > 1 and provider in SINGLE_TOOL_CALL_PROVIDERS:
            # Serialising costs an extra hop and works; the model asks for the rest on the
            # following iteration, having seen only the call that was echoed.
            calls = calls[:1]
        if content.strip():
            text = content
            await emit("message", text=content)

        # Echo the assistant turn back explicitly rather than via model_dump(): the
        # response object carries fields (refusal, audio, annotations, reasoning) that some
        # OpenAI-compatible servers reject on the way back in. Ids are filled in where a
        # server omitted them, because the tool message must name the call it answers.
        call_ids = [cid or f"call_{i}_{n}" for n, (cid, _, _) in enumerate(calls)]
        msgs.append({"role": "assistant", "content": content,
                     **({"tool_calls": [{"id": cid, "type": "function",
                                         "function": {"name": _tool_name(name),
                                                      "arguments": raw or "{}"}}
                                        for cid, (_, name, raw) in zip(call_ids, calls)]}
                        if calls else {})})
        if not calls:
            if not tools_used and not nudged:
                nudged = True
                text = ""  # the unsearched answer is never the one returned
                msgs.append({"role": "user", "content": SEARCH_FIRST_NUDGE})
                continue
            # Only this final message is the answer. Text that rode along with an earlier
            # tool call is narration ("let me search for that"), and returning it when the
            # final message is empty passed narration off as the answer.
            if not content.strip() and tools_used and not answer_nudged:
                # Small models sometimes stop after reading the results without saying
                # anything. They have the evidence in context; one prompt to use it is
                # cheaper than an empty turn the user has to re-ask.
                answer_nudged = True
                msgs.append({"role": "user", "content": ANSWER_NUDGE})
                continue
            # A reasoning model that exhausts its budget mid-thought returns content=null
            # with no tool calls. Returning that as the final answer would hand the caller
            # an empty string that looks like a considered response.
            if not content.strip():
                return _finish("", tools_used, usage, sources=sources, detail=(
                    "model returned no content and called no tools -- if it is a reasoning "
                    "model, raise LLM_MAX_TOKENS; its budget went on reasoning"))
            return _finish(content, tools_used, usage, sources=sources)

        for cid, (_, name, raw) in zip(call_ids, calls):
            name = _tool_name(name)
            args, error = _parse_args(raw)
            result = (f"{error} Arguments for {name} were not run." if error
                      else await _dispatch(name, args))
            tools_used.append(name)
            # Every id the tool put in front of the model, in the order it saw them.
            sources.extend(SPEECH_ID_RE.findall(result))
            await emit("tool_use", tool=name, input=args or {})
            msgs.append({"role": "tool", "tool_call_id": cid, "content": result})

    # Out of iterations with the model still calling tools. Return what it last said rather
    # than nothing, and say plainly that the answer is truncated.
    return _finish(text, tools_used, usage, sources=sources,
                   detail="stopped at llm_max_iterations")
