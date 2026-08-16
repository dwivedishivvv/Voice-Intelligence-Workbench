"""The agent layer is the one feature that sends corpus text off the box, and the one that
states conclusions in prose a reader will take at face value. Both make its contracts worth
asserting rather than reviewing: a weakened tool description or a dropped prompt rule
produces output that still looks entirely plausible.

None of this calls the model. What it checks is everything decided *before* the request —
the gate, the tool surface, the prompt rules, and the SQL — which is where the failures
that matter live.
"""
import asyncio
import inspect
import re

import pytest

from api.app.services import agent
from common.config import Settings, TUNABLE_FIELDS


# --- the privacy gate -------------------------------------------------------

def test_agent_is_off_by_default():
    """The product's headline claim is that no audio and no text leaves the box. Enabling
    the agent is what makes that claim conditional, so it has to be an explicit act.

    Asserts the *declared* default rather than Settings().llm_enabled: Settings reads .env,
    so on a machine where someone has switched the agent on, instantiating it tests the
    local configuration instead of the shipped default and passes for the wrong reason."""
    assert Settings.model_fields["llm_enabled"].default is False


def test_default_provider_does_not_assume_a_third_party():
    """Same reasoning for the provider: whichever way it defaults, that is the vendor a
    fresh deployment talks to."""
    assert Settings.model_fields["llm_provider"].default == "anthropic"
    assert Settings.model_fields["llm_model"].default == ""


def test_agent_cannot_be_switched_on_from_the_settings_page():
    """Everything in TUNABLE_FIELDS is editable live from a web page by anyone holding the
    API key. Sending transcripts to a third party is a deployment decision, not a
    threshold — it belongs in .env, where turning it on is deliberate and reviewable."""
    assert "llm_enabled" not in TUNABLE_FIELDS
    assert "anthropic_api_key" not in TUNABLE_FIELDS


def test_disabled_agent_raises_rather_than_calling_out():
    """The gate must be checked before any client is constructed — a disabled agent that
    still builds a request is a disabled agent that can still leak.

    asyncio.run rather than pytest-asyncio: the repo's suite is sync-only, and this is not
    worth a new test dependency."""
    cfg = Settings(llm_enabled=False, anthropic_api_key="sk-test")
    with pytest.raises(agent.LLMUnavailable):
        asyncio.run(agent.answer("anything", cfg=cfg))


@pytest.mark.parametrize("provider,field,expected", [
    ("anthropic", "anthropic_api_key", "ANTHROPIC_API_KEY"),
    ("nvidia", "nvidia_api_key", "NVIDIA_API_KEY"),
])
def test_enabled_agent_without_a_key_fails_clearly(provider, field, expected):
    """The provider is pinned explicitly: without it these inherit whatever .env sets, and
    the test silently checks the wrong provider's credentials."""
    cfg = Settings(llm_enabled=True, llm_provider=provider, **{field: ""})
    with pytest.raises(agent.LLMUnavailable, match=expected):
        asyncio.run(agent.answer("anything", cfg=cfg))


def test_unknown_provider_is_rejected_before_any_call_goes_out():
    cfg = Settings(llm_enabled=True, llm_provider="wat", nvidia_api_key="x")
    with pytest.raises(agent.LLMUnavailable, match="unknown LLM_PROVIDER"):
        asyncio.run(agent.answer("anything", cfg=cfg))


def test_each_provider_has_its_own_default_model():
    """llm_model defaults to empty and is resolved per provider, so switching provider
    cannot leave the request pointing at the other vendor's model id."""
    assert set(agent.DEFAULT_MODELS) == {"anthropic", "nvidia"}
    assert agent.DEFAULT_MODELS["anthropic"].startswith("claude")
    assert "/" in agent.DEFAULT_MODELS["nvidia"]


def test_openai_schema_is_derived_from_the_same_tool_declarations():
    """One source of truth for tool definitions across both wire formats — a tool added for
    one provider must not be invisible to the other."""
    openai_names = {f["function"]["name"] for f in agent._openai_tools()}
    assert openai_names == set(TOOL_SCHEMAS)
    for f in agent._openai_tools():
        fn = f["function"]
        assert fn["description"] == TOOL_SCHEMAS[fn["name"]]["description"]
        assert fn["parameters"] == TOOL_SCHEMAS[fn["name"]]["input_schema"]


# --- the tool surface -------------------------------------------------------

TOOL_SCHEMAS = {t.to_dict()["name"]: t.to_dict() for t in agent.TOOLS}


def test_the_four_planned_tools_are_exposed():
    assert set(TOOL_SCHEMAS) == {"search_speech", "expand_speech",
                                 "driver_timeline", "compare_speakers"}


@pytest.mark.parametrize("name", sorted(TOOL_SCHEMAS))
def test_every_tool_documents_when_to_call_it_not_just_what_it_is(name):
    """Tool descriptions are the highest-leverage text in the layer: the most common
    failure on current models is a tool that never gets called because its description only
    described itself. Each one has to carry a usage trigger."""
    desc = TOOL_SCHEMAS[name]["description"].lower()
    assert len(desc) > 200, f"{name}: description too thin to steer selection"
    assert any(k in desc for k in ("use this", "call this", "start here")), \
        f"{name}: no trigger condition, only a definition"


@pytest.mark.parametrize("name", sorted(TOOL_SCHEMAS))
def test_every_tool_parameter_is_described(name):
    """An undescribed parameter is one the model fills in by guessing."""
    props = TOOL_SCHEMAS[name]["input_schema"]["properties"]
    assert props, f"{name}: no parameters at all"
    for param, spec in props.items():
        assert spec.get("description"), f"{name}.{param} has no description"


def test_no_tool_can_write():
    """The agent is a lens over the corpus, not an editor of it. Corrections stay in the
    human-in-the-loop UI, where they are audited and reversible.

    Uppercase-only, and not case-insensitive: SQL and Cypher keywords are written in caps
    throughout this codebase while prose is not, so a case-insensitive match flags the
    English words "set", "create" and "update" wherever they appear in a comment. It did
    exactly that, which is a false positive that trains people to ignore the test."""
    WRITE_KEYWORDS = {"INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "MERGE", "SET"}

    # The tool bodies, plus every query constant they reach for: compare_speakers holds its
    # SQL in a module constant, so checking only the function source would miss it.
    from api.app.services import graph_context
    sources = {t.to_dict()["name"]: inspect.getsource(t.func) for t in agent.TOOLS}
    sources["COMPARE_SPEAKERS_SQL"] = agent.COMPARE_SPEAKERS_SQL
    sources["EXPAND_CYPHER"] = graph_context.EXPAND_CYPHER
    sources["DRIVER_LAPS_CYPHER"] = graph_context.DRIVER_LAPS_CYPHER

    for name, src in sources.items():
        found = set(re.split("[^A-Za-z_]+", src)) & WRITE_KEYWORDS
        assert not found, f"{name} contains write keyword(s): {sorted(found)}"


# --- the prompt rules -------------------------------------------------------

def test_prompt_requires_citation():
    """Without ids, no claim the agent makes can be traced back to source audio, and the
    UI has nothing to link."""
    assert "cite" in agent.SYSTEM.lower()
    assert "speech id" in agent.SYSTEM.lower()


def test_prompt_forbids_stating_tone_as_fact():
    """tone.py is a threshold heuristic over acoustic features. The renderer hedges its
    output, but the model can still paraphrase 'reads stressed' into 'was stressed' unless
    it is told not to."""
    s = agent.SYSTEM.lower()
    assert "heuristic" in s
    assert "never" in s and "was stressed" in s


def test_prompt_requires_admitting_missing_data():
    """Identification abstains and lap alignment needs a timestamp much of the corpus does
    not have. An agent that fills those gaps rather than naming them is worse than no
    agent, because the product's whole posture is abstain-over-guess."""
    s = agent.SYSTEM.lower()
    assert "cannot answer" in s or "does not support" in s


def test_prompt_scopes_the_agent_to_the_corpus():
    """The model knows a great deal about Formula 1 that is not in this database. Answers
    sourced from that knowledge are uncitable and unverifiable."""
    assert "general knowledge" in agent.SYSTEM.lower()


# --- SQL ---------------------------------------------------------------------

def test_compare_speakers_excludes_unfinished_clips():
    """Same rule the graph projection is held to: a half-processed clip has no finished
    transcript, and counting it inflates a number the agent will state as fact."""
    assert agent.COMPARE_SPEAKERS_SQL.count("status = 'COMPLETE'") == 2


def test_compare_speakers_aggregates_in_separate_ctes():
    """utterances and clip_speakers are both one-to-many from a profile. Aggregating them
    in one query with two LEFT JOINs fans out: it reported 693 utterances for a speaker who
    has 33, in a corpus of 674, and the figure looked plausible enough to ship."""
    sql = agent.COMPARE_SPEAKERS_SQL
    assert "WITH utt AS" in sql and "share AS" in sql
    body = sql[:sql.index("SELECT p.id")]
    tail = sql[sql.index("SELECT p.id"):]
    # The final SELECT joins only the pre-aggregated CTEs, never the base tables again.
    # Matched on FROM/JOIN rather than the bare name: the output column is called
    # n_utterances, so a plain substring check passes for the wrong reason.
    base = re.compile(r"\b(FROM|JOIN)\s+(utterances|clip_speakers)\b", re.I)
    assert not base.search(tail), "final SELECT re-joins a base table — fan-out is back"
    assert len(base.findall(body)) == 2, "each base table should be aggregated once, in a CTE"


# --- citations ---------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Nothing cited here.", []),
    ("He said this [3dc30df5-8d0c-4788-8296-2a3f539f2c8b].",
     ["3dc30df5-8d0c-4788-8296-2a3f539f2c8b"]),
    ("Two [3dc30df5-8d0c-4788-8296-2a3f539f2c8b] and [f0841aeb-8014-4989-9dca-d4e4112ea7b7].",
     ["3dc30df5-8d0c-4788-8296-2a3f539f2c8b", "f0841aeb-8014-4989-9dca-d4e4112ea7b7"]),
])
def test_citation_extraction(text, expected):
    assert agent.SPEECH_ID_RE.findall(text) == expected


def test_repeated_citation_is_reported_once():
    """The UI renders one link per citation; the same id twice is one source, not two."""
    sid = "3dc30df5-8d0c-4788-8296-2a3f539f2c8b"
    found = list(dict.fromkeys(agent.SPEECH_ID_RE.findall(f"[{sid}] and again [{sid}]")))
    assert found == [sid]


@pytest.mark.parametrize("sentinel", ["null", "None", "undefined", "NIL", "n/a", "-", "  none  ", ""])
def test_placeholder_speaker_filters_are_ignored(sentinel):
    """A model that omits an optional argument often sends the string "null" instead of
    omitting it. Taken literally that filters to a speaker who cannot exist, and the tool
    replies "there is no speech by null" -- which a user reads as a fact about the corpus
    rather than a malformed call. Seen in a real session."""
    assert agent.is_placeholder(sentinel) is True


@pytest.mark.parametrize("real", ["ALB", "Albon", "Alexander Albon", "23", "Mercedes", "no"])
def test_real_speaker_filters_are_kept(real):
    """The guard must not swallow a genuine filter. "no" is the trap: it starts like
    "none" but is a plausible surname fragment, and a prefix match would drop it."""
    assert agent.is_placeholder(real) is False


def test_null_speaker_filter_does_not_suppress_results(monkeypatch):
    """The guard has to be wired into search_speech, not merely exist.

    Testing is_placeholder alone passes even if search_speech never calls it — which was
    the case when this was first written. Stubs the two dependencies so the tool runs
    without Postgres or Neo4j, and asserts a bogus "null" filter does not swallow a real
    hit."""
    async def fake_anchor(q, limit=8, mode="hybrid", speaker_id=None, mood=None):
        return [{"speech_id": "abc-1", "kind": "utterance", "text": "hello there", "score": 1.0}]

    async def fake_expand(ids):
        raise RuntimeError("graph unavailable")  # forces the degraded path too

    monkeypatch.setattr(agent.search_svc, "anchor_speech", fake_anchor)
    monkeypatch.setattr(agent.graph_context, "expand", fake_expand)

    out = asyncio.run(agent.search_speech.func(query="anything", spoken_by="null"))
    assert "No speech by" not in out, "a placeholder filter was treated as a real speaker"
    assert "abc-1" in out


# --- searching by tone --------------------------------------------------------

def _stub_search(monkeypatch, *, anchors=None, by_mood=None):
    """Run the tools without Postgres or Neo4j, recording what they asked for."""
    seen = {}

    async def fake_anchor(q, limit=8, mode="hybrid", speaker_id=None, mood=None):
        seen["anchor"] = {"q": q, "limit": limit, "mood": mood}
        return anchors or []

    async def fake_by_mood(mood, limit=10, speaker_id=None):
        seen["by_mood"] = {"mood": mood, "limit": limit}
        return by_mood or []

    async def fake_expand(ids):
        raise RuntimeError("graph unavailable")

    monkeypatch.setattr(agent.search_svc, "anchor_speech", fake_anchor)
    monkeypatch.setattr(agent.search_svc, "speech_by_mood", fake_by_mood)
    monkeypatch.setattr(agent.graph_context, "expand", fake_expand)
    return seen


def test_tone_only_search_does_not_go_through_text_ranking(monkeypatch):
    """The bug this fixes: "which calls sound stressed" was answered by searching the word
    "stressed" in transcripts, which matches nothing, and the model reported the empty
    result as a fact about the corpus. A mood with no topic must take the column path."""
    hit = [{"speech_id": "abc-1", "kind": "radio_call", "text": "box box", "mood": "stressed"}]
    seen = _stub_search(monkeypatch, by_mood=hit)

    out = asyncio.run(agent.search_speech.func(mood="stressed"))
    assert seen["by_mood"]["mood"] == "stressed"
    assert "anchor" not in seen, "a tone-only search still ran a text query"
    assert "abc-1" in out


def test_tone_filter_composes_with_a_topic(monkeypatch):
    """Topic and tone are separate axes; asking for both is one search, not two."""
    seen = _stub_search(monkeypatch, anchors=[
        {"speech_id": "abc-1", "kind": "utterance", "text": "tyres are gone", "mood": "tired"}])

    asyncio.run(agent.search_speech.func(query="tyre wear", mood="tired"))
    assert seen["anchor"] == {"q": "tyre wear", "limit": 8, "mood": "tired"}


def test_unknown_mood_is_reported_as_a_bad_argument(monkeypatch):
    """A mood the classifier cannot emit matches nothing. Left to run, "no speech found"
    reads as a statement about the corpus rather than about the call."""
    seen = _stub_search(monkeypatch)
    out = asyncio.run(agent.search_speech.func(mood="furious"))
    assert "not a tone reading" in out
    assert all(k not in seen for k in ("anchor", "by_mood")), "a bad filter still hit the db"
    for valid in agent.search_svc.MOODS:
        assert valid in out, "the error should list what the classifier can emit"


def test_empty_search_asks_for_a_filter_rather_than_scanning(monkeypatch):
    seen = _stub_search(monkeypatch)
    out = asyncio.run(agent.search_speech.func())
    assert "query" in out and "mood" in out
    assert not seen


def test_placeholder_mood_is_ignored_like_any_other_optional_argument(monkeypatch):
    """Same failure mode as spoken_by="null": taken literally it filters to a tone that
    cannot exist and the tool reports an empty corpus."""
    seen = _stub_search(monkeypatch, anchors=[
        {"speech_id": "abc-1", "kind": "utterance", "text": "hello", "score": 1.0}])
    out = asyncio.run(agent.search_speech.func(query="anything", mood="null"))
    assert seen["anchor"]["mood"] is None
    assert "abc-1" in out


def test_search_limit_is_capped(monkeypatch):
    """A model asking for 1000 hits should get a bounded query, not a table scan rendered
    into the context window."""
    seen = _stub_search(monkeypatch, anchors=[])
    asyncio.run(agent.search_speech.func(query="anything", limit=1000))
    assert seen["anchor"]["limit"] <= 25


def test_search_results_carry_the_tone_reading(monkeypatch):
    """A tone-filtered search whose hits render as plain quotes invites the model to drop
    the qualifier. The reading rides along on the line, hedged."""
    _stub_search(monkeypatch, by_mood=[
        {"speech_id": "abc-1", "kind": "radio_call", "text": "box box", "mood": "stressed"}])
    out = asyncio.run(agent.search_speech.func(mood="stressed"))
    assert "voice reads stressed" in out
    assert "was stressed" not in out


def test_prompt_tells_the_model_tone_is_filtered_not_searched():
    """The tool description alone is not enough: the model has to know that the word and
    the label are different things before it decides how to look."""
    s = agent.SYSTEM.lower()
    assert "mood" in s and "recency" in s


# --- tools compose ------------------------------------------------------------

def test_expand_accepts_several_ids_in_one_call(monkeypatch):
    """A question about a pattern needs the context of every hit. One id per call turns
    that into a round trip each, and models give up on the fifth."""
    ids = ["3dc30df5-8d0c-4788-8296-2a3f539f2c8b", "f0841aeb-8014-4989-9dca-d4e4112ea7b7"]
    asked = {}

    async def fake_expand(passed):
        asked["ids"] = passed
        return [{"speech_id": i, "text": "x"} for i in passed]

    monkeypatch.setattr(agent.graph_context, "expand", fake_expand)
    monkeypatch.setattr(agent.graph_context, "render_context", lambda rows: "ctx")
    asyncio.run(agent.expand_speech.func(", ".join(ids)))
    assert asked["ids"] == ids


def test_expand_names_the_ids_the_graph_did_not_have(monkeypatch):
    """Returning fewer rows than asked for, silently, reads as absence of evidence."""
    ids = ["3dc30df5-8d0c-4788-8296-2a3f539f2c8b", "f0841aeb-8014-4989-9dca-d4e4112ea7b7"]

    async def fake_expand(passed):
        return [{"speech_id": passed[0], "text": "x"}]

    monkeypatch.setattr(agent.graph_context, "expand", fake_expand)
    monkeypatch.setattr(agent.graph_context, "render_context", lambda rows: "ctx")
    out = asyncio.run(agent.expand_speech.func(", ".join(ids)))
    assert ids[1] in out and "Not in the graph" in out


def test_one_tool_call_per_turn_on_openai_compatible_endpoints(monkeypatch):
    """NVIDIA's llama-3.1 endpoint answers a request whose history carries two tool_calls
    in one assistant turn with `400: This model only supports single tool-calls at once`,
    which kills the turn on the request *after* the one that fanned out. Making the tools
    composable made the model fan out, so the loop serialises them itself.
    """
    from types import SimpleNamespace as NS

    def call(name, args, cid):
        return NS(id=cid, function=NS(name=name, arguments=args))

    sent = []
    responses = [
        NS(choices=[NS(message=NS(content=None, tool_calls=[
            call("search_speech", '{"mood": "calm"}', "c1"),
            call("search_speech", '{"mood": "stressed"}', "c2")]))],
           usage=NS(prompt_tokens=10, completion_tokens=2)),
        NS(choices=[NS(message=NS(content="done", tool_calls=[]))],
           usage=NS(prompt_tokens=10, completion_tokens=2)),
    ]

    class FakeCompletions:
        async def create(self, **kw):
            sent.append(kw["messages"])
            return responses[len(sent) - 1]

    class FakeClient:
        def __init__(self, **kw):
            self.chat = NS(completions=FakeCompletions())

    monkeypatch.setattr("openai.AsyncOpenAI", FakeClient)
    dispatched = []

    async def fake_dispatch(name, args):
        dispatched.append((name, args))
        return "no rows"

    monkeypatch.setattr(agent, "_dispatch", fake_dispatch)

    cfg = Settings(llm_enabled=True, llm_provider="nvidia", nvidia_api_key="x",
                   llm_max_iterations=4)
    result = asyncio.run(agent._run_openai_compatible(
        cfg, "m", "sys", [{"role": "user", "content": "q"}], lambda *a, **k: _noop()))

    assert len(dispatched) == 1, "both calls in one turn is what the provider rejects"
    echoed = [m for m in sent[1] if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(echoed[0]["tool_calls"]) == 1
    assert result["answer"] == "done"


async def _noop(*a, **k):
    return None


def test_compare_speakers_resolves_names_as_well_as_ids(monkeypatch):
    """The roster gives ids, but a question about "Marta" should not fail on the model not
    having copied a UUID across."""
    calls = {}

    async def fake_fetch(sql, arg):
        calls.setdefault("sql", []).append((sql, arg))
        if "display_name" in sql and "ANY" in sql and "speaker_profiles WHERE" in sql:
            return [{"id": "3dc30df5-8d0c-4788-8296-2a3f539f2c8b", "display_name": "Marta Kolar"}]
        return []

    monkeypatch.setattr(agent.db, "fetch", fake_fetch)
    out = asyncio.run(agent.compare_speakers.func("Marta Kolar", "Dan Reisz"))
    # "Dan Reisz" is not in the stubbed directory, so the tool says which name it could not
    # place rather than comparing whoever it did find.
    assert "Dan Reisz" in out and "No enrolled speaker named" in out
