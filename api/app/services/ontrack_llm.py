"""Adjudicate on-track event candidates with the configured model.

The embedding pass in ontrack_extract.py is a candidate generator: cheap, on-box, and
matching topic rather than proposition, which is why it reads "Head down." as a completed
overtake. This pass hands those candidates to a model that can read the sentence, and keeps
only what it confirms. Measured on this corpus, that is the difference between ~65% and the
number in the report — the embedding score cannot rise much further because similarity is
not comprehension.

THIS SENDS TRANSCRIPT TEXT OFF THE BOX. It is the second thing in the system that does, and
unlike Ask — where a person asks a question and knowingly gets an answer composed remotely —
this one runs over stored corpus text in the background. So it is off by default behind its
own setting, it refuses when the agent itself is disabled, and it only ever sends lines that
already cleared the local similarity bar: strategy, tyres and morale never leave.
"""
import asyncio

from common import ontrack
from common.config import get_effective_settings
from .agent import DEFAULT_MODELS, LLMUnavailable


async def _complete(cfg, prompt: str) -> str:
    """One completion from whichever provider is configured.

    Deliberately not the agent's tool-runner: this is a single labelling call with no tools
    and no conversation, and routing it through the loop would buy nothing but its failure
    modes."""
    provider = (cfg.llm_provider or "anthropic").lower()
    if provider not in DEFAULT_MODELS:
        raise LLMUnavailable(f"unknown LLM_PROVIDER {provider!r}")
    key = getattr(cfg, f"{provider}_api_key", "")
    if not key:
        raise LLMUnavailable(f"no API key for {provider} — set one in Settings › Ask AI")
    model = cfg.llm_model or DEFAULT_MODELS[provider]

    if provider == "anthropic":
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=key)
        msg = await client.messages.create(
            model=model, max_tokens=cfg.llm_max_tokens,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if b.type == "text")

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=key, base_url=getattr(cfg, f"{provider}_base_url"))
    # temperature 0: this is a labelling task with a closed vocabulary, and sampling only
    # buys variety in a place where variety is error.
    res = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=cfg.llm_max_tokens, temperature=0)
    return res.choices[0].message.content or ""


async def adjudicate(candidates: list[dict], cfg=None,
                     on_progress=None) -> tuple[list[dict], list[str]]:
    """Label a list of candidate lines, in batches.

    `candidates` are dicts with at least `speech_id` and `text`. Returns the lines the
    model *confirmed* — carrying the event type, its confidence, the other car it named and
    why — and the errors from any batch that could not be read at all.

    Partial labelling is a usable outcome, so a batch that fails after its retries is
    skipped rather than fatal. It is never silent: the caller gets the reasons and can say
    how much of the corpus went unread.
    """
    cfg = cfg or await get_effective_settings()
    if not cfg.llm_enabled:
        raise LLMUnavailable("agent is disabled (Settings › Ask AI), so nothing may be sent off-box")
    if not cfg.ontrack_llm_enabled:
        raise LLMUnavailable(
            "on-track LLM refinement is off (ontrack_llm_enabled) — it sends corpus text "
            "off the box, so it is opt-in separately from Ask")

    size = max(1, cfg.ontrack_llm_batch)
    confirmed: list[dict] = []
    failures: list[str] = []
    for start in range(0, len(candidates), size):
        batch = candidates[start:start + size]
        reply, error = None, None
        # Hosted endpoints rate-limit a run like this hard — a long pass fires dozens of
        # requests at a free tier, and the first 429 is not a reason to give up on the
        # batch. Backoff rather than a fixed sleep, because the limiter's window is what
        # decides how long the wait needs to be.
        for attempt in range(3):
            try:
                reply = await _complete(cfg, ontrack.build_prompt([c["text"] for c in batch]))
                break
            except LLMUnavailable:
                raise
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)[:120]}"
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt * 5)
        if reply is None:
            # Reported, not swallowed. A pass that quietly labels half the corpus and says
            # it succeeded is worse than one that says which part it could not read.
            failures.append(error or "unknown")
            if on_progress:
                await on_progress(start + len(batch), len(candidates), failed=True)
            continue

        for verdict in ontrack.parse_verdicts(reply, len(batch), cfg.ontrack_llm_min_confidence):
            row = batch[verdict.index - 1]
            confirmed.append({
                **row,
                "event_type": verdict.event_type,
                "family": ontrack.BY_NAME[verdict.event_type].family,
                "score": verdict.confidence,
                "other_car": verdict.other_car,
                "why": verdict.why,
            })
        if on_progress:
            await on_progress(start + len(batch), len(candidates), failed=False)
        # Politeness gap between batches. Free tiers rate-limit hard, and a 429 storm costs
        # more wall time than the pause it would have taken to avoid it.
        await asyncio.sleep(0.2)
    return confirmed, failures
