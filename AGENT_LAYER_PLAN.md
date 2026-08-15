# Agentic LLM Layer — Design Plan

An LLM that answers questions about the corpus by *querying* it, not by being handed a
pre-stuffed context blob. Sits on top of the graph engine (`GRAPH_RAG_PLAN.md`) and the
existing hybrid search.

Depends on Phase 3 of the graph plan for its best tool, but degrades to search-only if the
graph isn't there yet.

---

## Implementation status

Built and wired: four read-only tools, the cached system prompt with its entity roster,
audit, the privacy gate, and progress events on the existing websocket channel.

Deviations from the design below, each for a reason worth keeping:

| Deviation | Why |
|---|---|
| **`/ask` blocks and returns the answer**, rather than returning ids and streaming the result. | Redis pub/sub has no replay. An answer delivered only over the socket is lost to any client that reconnects a second late. The body is the durable path; the socket carries progress. The client supplies `conversation_id` so it can subscribe *before* posting. |
| **Progress events are per-message, not per-token.** | Tool calls and assistant turns, not deltas. No accumulator, no reconciliation, and the plan's Phase B/C value (seeing what it is doing) lands either way. Token-level streaming is the upgrade, not the starting point. |
| **`search_speech` returns a brief list; `expand_speech` returns the neighbourhood.** | The plan had search returning rich rows. Keeping search sparse makes expansion the path of least resistance instead of an optional extra — a model that can answer from the search output will, from truncated quotes with none of the surroundings. |

**Verified against a live model** — on NVIDIA NIM (`meta/llama-3.1-70b-instruct`), not
Anthropic. `LLM_PROVIDER` selects between them: the four tools, the prompt, the citation
extraction and the audit are shared; only the loop differs (the SDK tool runner for
Anthropic, an explicit call/execute/append loop for OpenAI-compatible endpoints). Tool
schemas are generated once from the `@beta_async_tool` declarations and adapted to the
OpenAI shape, so neither provider can drift from the other's tool surface.

Running it on a 70B model surfaced four things a stronger model would have hidden:

| Finding | Fix |
|---|---|
| The model emitted `limit` as the **string** `"1"`; asyncpg rejects a string for a Postgres LIMIT and the turn died. | Coerce tool arguments to their declared types before dispatch. Optional params render as `anyOf:[{integer},{null}]`, so the resolver reads through `anyOf` — otherwise every optional parameter, the ones most often stringified, skipped coercion. |
| Reasoning models (`nemotron-super`) return `content: null` plus a separate `reasoning` field and spend the whole budget thinking. | Default to a non-reasoning model; treat empty content with no tool calls as an explicit failure rather than returning "" as the answer. |
| Search results omitted **who spoke**, so "what did X say" was unanswerable from them — models picked the first hit and cited an utterance that merely *mentioned* the driver. | `render_brief` now carries speaker, session and lap. |
| Asking for `spoken_by="Albon"` matched nothing, because the graph stores the code `ALB` — reported to the user as "the corpus has no such speech". | Match names and codes in both directions. |

With those, the question *"What did Albon say about power, and which lap was it on?"* is
answered correctly and cited: lap 33, `[204fa940…]`. `mistral-nemotron` still misses it,
which is the point of the ceiling below — this is one question, not an eval suite. Everything decided before the request is verified: the
tool schemas the SDK generates, the prompt assembly, the gate, and all four tool functions
executed against the live Postgres and Neo4j.

Live tool runs found one real bug — `compare_speakers` reported **693 utterances for a
speaker who has 33**, in a corpus of 674. Two independent one-to-many joins (`utterances`
and `clip_speakers`) in one `GROUP BY` fanned out, multiplying the count by 21. The figure
looked entirely plausible, which is exactly the kind of number an agent would have stated
as fact. Now aggregated in separate CTEs, with a test that fails if the base tables are
re-joined in the final `SELECT`.

---

## The one architectural decision

**Tools, not a context blob. The model navigates; the retrieval layer stays dumb.**

The tempting shape is: run hybrid search, paste the top 20 utterances into a prompt, ask
the question. That fails the moment the question needs two hops ("was he stressed on the
laps where he lost time to the car ahead") because retrieval can't know what to fetch until
after the model has reasoned about the first result.

Giving the model four read-only tools and letting it loop is *less* code than building a
query planner, and it's the only shape that uses the graph for what a graph is for.

Second decision, equally load-bearing: **every tool result carries stable IDs, and the
system prompt requires citing them.** That's the anti-hallucination mechanism, and it's
free — `utterance.id`, `clip.id`, and lap numbers already exist. A claim without an ID is
a claim the UI can't link, which makes it visibly suspect.

---

## Where it runs: the API, not the worker

The worker exists to own GPU models and serialize access to them (`ModelPool`,
`live_sem`). An agent turn is network-bound — minutes of waiting on Anthropic, zero GPU.
Putting it in the worker would occupy an arq concurrency slot doing nothing, competing
with actual transcription jobs.

The API is already async FastAPI with a Postgres pool and a Redis connection. That's the
whole runtime requirement. New router `api/app/routers/agent.py`, and the model client
lives in `api/app/agent/` alongside it.

### Streaming needs no new transport

`api/app/routers/ws.py` subscribes to `job:{job_id}` and forwards whatever is published,
verbatim — the comment in `worker/worker/events.py:emit_live` already notes this was
designed to need "no live-specific plumbing." The agent gets the same treatment:

```
POST /v1/agent/ask  ->  {"conversation_id": "...", "ws_url": "/v1/ws/jobs/{conversation_id}"}
```

The agent publishes `{"type": "agent_delta", ...}`, `{"type": "agent_tool_use", ...}`,
`{"type": "agent_done", ...}` to `job:{conversation_id}`. The existing WS route and the
existing frontend hook both work unchanged.

One small consolidation: `worker/worker/events.py` imports nothing worker-specific. Move it
to `common/events.py` and have both the API and worker import it. That's a file move, not
a new module — the API currently has no publish helper, and adding a second one would be
the wrong kind of new code.

---

## Tool surface

Four tools. All read-only. **No tool writes to Postgres, Neo4j, or disk.**

| Tool | Input | Returns |
|---|---|---|
| `search_utterances` | `query`, optional `speaker_id`/`race_id`/`sentiment`/`session_key`, `limit` | Ranked utterances with IDs, text, speaker, clip, timestamp |
| `expand` | `utterance_id`, `depth=1` | Rendered neighbourhood — speaker, driver, session, lap ±2, adjacent utterances, mentions |
| `driver_timeline` | `driver_number`, `session_key` | Laps in order with times, interleaved with that driver's radio calls and their tone |
| `compare_speakers` | `a`, `b`, optional `scope` | Talk share, sentiment/mood rollups, co-occurrence, side by side |

`search_utterances` wraps the existing hybrid RRF in `api/app/routers/search.py` — the
function, not an HTTP call to itself. `expand` wraps the Phase 3 graph context renderer.
Both already exist or are already planned; the tool layer is a thin typed shell over them.

### The roster goes in the prompt, not in a tool

The obvious fifth tool is `list_drivers` / `list_races`, so the model can resolve "Max" to
driver number 1 without inventing a session key. Don't build it. The roster is ~20 drivers,
~10 teams, and however many races exist — a few hundred tokens, and it changes rarely.

Render it into the **cached** system prompt instead. It costs one cache read per turn
instead of a full tool round-trip on nearly every question, and it removes an entire class
of "model guessed an ID" failure before it can happen.

Revisit only if the roster grows past a few hundred entries.

### Why these four and not twelve

Every tool definition is tokens on every request and one more thing for the model to pick
wrong. These four cover: find something, understand its surroundings, follow one subject
through time, contrast two subjects. Add a fifth when a real question can't be answered by
composing these — not before.

---

## The loop

Use the SDK's tool runner. Don't hand-write the agentic loop.

```python
import anthropic
from anthropic import beta_async_tool

client = anthropic.AsyncAnthropic()

@beta_async_tool
async def search_utterances(query: str, speaker_id: str | None = None,
                            sentiment: str | None = None, limit: int = 10) -> str:
    """Search radio calls and transcripts by meaning and keyword.

    Call this to find utterances relevant to a question before answering it. Returns
    ranked results with utterance IDs — cite those IDs in your answer.

    Args:
        query: What to search for, in natural language.
        speaker_id: Restrict to one enrolled speaker profile (UUID).
        sentiment: One of negative, neutral, positive.
        limit: Max results, default 10.
    """
    ...

runner = client.beta.messages.tool_runner(
    model="claude-opus-5",
    max_tokens=16000,
    thinking={"type": "adaptive"},
    output_config={"effort": "high"},
    system=[{"type": "text", "text": SYSTEM_PROMPT + roster,
             "cache_control": {"type": "ephemeral"}}],
    tools=[search_utterances, expand, driver_timeline, compare_speakers],
    messages=messages,
    max_iterations=12,
)

async for message in runner:
    await publish_deltas(conversation_id, message)
```

Notes that are load-bearing rather than decorative:

- **Tool descriptions are the highest-leverage text in the whole layer.** The docstring is
  the tool's contract *and* its trigger condition — say when to call it, not just what it
  does. Under-describing a tool is the most common cause of a model that won't use it.
- **`max_iterations=12`** caps a runaway loop. Cheaper and simpler than a token budget for
  a first cut; swap in `output_config.task_budget` (beta `task-budgets-2026-03-13`,
  minimum 20k) if real questions start hitting the iteration cap legitimately.
- **Prompt caching pays immediately.** System prompt + roster + four tool schemas is a
  stable prefix on every request in a conversation, comfortably over the 512-token minimum
  on `claude-opus-5`. Put the breakpoint on the last system block. Verify with
  `usage.cache_read_input_tokens` — if it's zero across turns, something volatile leaked
  into the prefix (a timestamp, a per-request ID).
- **Check `stop_reason` before reading `content`.** `claude-opus-5` runs safety classifiers
  and can return `stop_reason: "refusal"` with an empty content array — code that indexes
  `content[0]` breaks on it. Opt into `fallbacks: "default"` (beta header
  `server-side-fallback-2026-07-01`) so a false-positive decline is re-served rather than
  surfaced to the user as a crash.
- **No LangChain, no agent framework.** The runner is the loop. A framework here would add
  a dependency, an abstraction layer, and a second place for prompts to live, to replace
  roughly fifteen lines.

---

## The system prompt: three things it must say

Short, and three constraints carry all the weight:

1. **Cite utterance IDs for every factual claim.** The UI renders them as links into clip
   detail at the right timestamp. No ID, no claim.
2. **Tone and mood are a heuristic, not ground truth.** `worker/worker/audio/tone.py` is
   threshold-based over acoustic features — the README says so plainly and the agent must
   too. "His voice read as stressed" is honest; "he was stressed" is not. Same for
   `sentiment`: the pipeline stores text sentiment and acoustic mood *separately* precisely
   so disagreement stays visible, and the agent should surface that disagreement rather
   than averaging it away.
3. **Say when the data doesn't answer the question.** Speaker identification abstains by
   design (`unknown`, `abstained` in `id_result`); the agent inherits that posture. An
   unattributed voice is "an unidentified speaker," not a guess.

The existing product is conservative everywhere — reliability gating, margin checks,
abstention over coin-flips. An agent that confidently over-claims on top of that stack
would undercut the thing that makes the rest of it trustworthy.

---

## Privacy: the seam that decides the product

**This is the first feature that sends transcript text off the box.** The README's opening
claim is "no audio and no text ever leaves the box." That claim and this feature cannot
both be unconditionally true.

Handle it explicitly rather than letting it erode:

- **`LLM_ENABLED=false` by default**, in `.env.example` with a comment saying exactly what
  turning it on means. Not a `TUNABLE_FIELDS` entry — this is a deployment decision, not a
  threshold to tweak from a Settings page.
- **The router 404s when disabled.** Same shape as the `GRAPH_ENABLED` gate. No half-enabled
  state, no silent fallback.
- **Amend the README's claim in the same PR that adds the feature.** "The core pipeline
  makes no outbound calls; the optional agent layer, off by default, sends transcript
  excerpts to the Anthropic API." Accurate beats aspirational.
- **The tool layer is the swap seam for a local model.** Tools are plain async functions;
  the client is one import. Running a local model through the same four tools is a
  one-file change. Don't build a provider abstraction now for a second provider that may
  never arrive — the seam already exists because the tools don't know who's calling them.

### Audit, without leaking transcripts

`common/audit.py` gives a hash-chained append-only log, and `LOG_TRANSCRIPTS=false` is the
existing default. Those pull in opposite directions for an agent that trades in text.

Resolve it the way the pipeline already does — log the *shape*, not the content:

```python
await audit("agent.ask", "conversation", conversation_id, after={
    "tools_called": ["search_utterances", "expand"],
    "n_iterations": 4,
    "input_tokens": usage.input_tokens,
    "output_tokens": usage.output_tokens,
    "cited_utterances": ["uuid", "uuid"],
    "question": question if cfg.log_transcripts else None,
})
```

Which tools ran, how many hops, what it cited, what it cost — enough to reconstruct and
audit the reasoning path without writing user text into an append-only log that by design
can never be deleted. `cited_utterances` is the important field: it makes every answer
traceable back to source audio.

---

## Conversation state: start with none

The Anthropic API is stateless and the client already holds the message array. Start there
— `POST /v1/agent/ask` takes `messages`, returns the assistant turn, stores nothing. Zero
tables, zero migrations, and history survives exactly as long as the browser tab does.

Add `agent_conversations (id UUID, messages JSONB, created_at, updated_at)` when someone
actually asks for history to survive a refresh. One table, one meaningful column. Not
before — and note that storing it turns a privacy question ("we send text to an API") into
a bigger one ("we retain it"), which deserves its own decision rather than arriving by
default.

If conversations do get long enough to matter, the API's own compaction
(`context_management`, beta `compact-2026-01-12`) handles it server-side. Don't build a
summarizer.

---

## Build order

| Phase | Ships alone as | Rough size |
|---|---|---|
| A — Walking skeleton | Ask a question, get a cited answer, non-streaming, 2 tools (`search_utterances`, `expand`) | 1 router, 1 tools file, ~150 lines |
| B — Citations in the UI | Answers link into clip detail at the timestamp — the credibility feature | frontend only |
| C — Streaming | Tokens and tool calls appear live over the existing WS route | `common/events.py` move + publish calls |
| D — Full tool set | `driver_timeline`, `compare_speakers`; the two-hop questions start working | 2 tool functions |
| E — Hardening | Refusal handling, fallbacks, audit rows, iteration caps, cost metrics | scattered, small |

Phase A is deliberately two tools, not four. Get one question answered end to end with a
real citation before deciding what the other tools should look like — the second pair will
be better designed for having watched the first pair get used.

Prompt caching goes in Phase A, not Phase E. It's two lines and it changes the cost
profile of every subsequent phase.

**Metrics** slot into the existing Prometheus setup (`api/app/observability.py`):
`agent_turn_seconds`, `agent_tool_calls_total{tool}`, `agent_tokens_total{kind}`. The
tool-call counter is the one that earns its keep — it tells you which tools the model
actually reaches for, which is how you find out that a description is under-written.

---

## Known ceilings, stated up front

- **Answer quality is bounded by tone accuracy.** The agent can only report what the
  heuristic classifier produced. A confident-sounding narrative built on threshold-based
  mood detection is the most likely way this layer misleads someone. The system prompt
  hedge is a mitigation, not a fix; the fix is a trained SER model, and the seam for that
  is still `extract_features` + the classifier beneath it.
- **`expand` is only as good as `DURING_LAP` coverage**, which depends on
  `clips.recorded_at` being populated. Manually uploaded clips have no lap edges, so
  timing questions about them return nothing. Absent, not guessed.
- **No evals.** There is no ground-truth question set, so "did the change help" is a vibe
  check. A dozen hand-written question/expected-citation pairs in `tests/` would make tool
  and prompt changes measurable — worth doing before tuning anything.
- **Cost is unmetered per user.** Single static API key, no per-caller accounting. Fine for
  a single-box deployment; `api/app/auth.py:get_current_user` is the documented seam if
  that changes, same as for everything else.
- **Two hops is the practical depth.** `max_iterations=12` bounds the loop, but questions
  needing genuinely deep traversal will hit it and answer partially rather than failing
  loudly. Watch `agent_tool_calls_total` for turns that flatline at the cap.
