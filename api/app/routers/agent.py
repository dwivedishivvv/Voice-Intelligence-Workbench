import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from common.audit import audit
from common.config import get_effective_settings
from common.events import emit_agent
from common.graph import GraphUnavailable
from ..auth import get_current_user
from ..services import agent as agent_svc, graph_context

router = APIRouter(prefix="/v1/agent", tags=["agent"])


class AskBody(BaseModel):
    question: str
    # Prior turns, verbatim from a previous response's `history`. The API is stateless by
    # design (AGENT_LAYER_PLAN.md): the client holds the transcript, so there is no
    # conversations table to migrate and no decision to make yet about retaining user text.
    history: list[dict] | None = None
    # Supplied by the client so it can open the progress websocket *before* posting.
    # Generated here if absent, in which case progress events are simply unobserved.
    conversation_id: str | None = None


@router.post("/ask")
async def ask(body: AskBody, user=Depends(get_current_user)):
    """Answer a question about the corpus, using the graph and search as tools.

    Blocking on purpose. The websocket carries progress, but redis pub/sub has no replay,
    so an answer delivered *only* over the socket is an answer lost to any client that
    reconnects a second late. The response body is the durable path; the socket is the
    nice-to-have."""
    cfg = await get_effective_settings()
    conversation_id = body.conversation_id or uuid.uuid4().hex

    async def on_event(kind: str, **payload):
        await emit_agent(conversation_id, kind, **payload)

    try:
        result = await agent_svc.answer(body.question, body.history, cfg, on_event)
    except agent_svc.LLMUnavailable as e:
        # 503, not 500: disabled or unconfigured is a deployment state, not a bad request.
        raise HTTPException(503, str(e)) from e
    except GraphUnavailable as e:
        raise HTTPException(503, f"agent needs the graph: {e}") from e

    # Resolve each cited id to something the UI can render and link: the quote, who said
    # it, the lap, and the clip plus offset to seek to. Done here rather than in the browser
    # so a citation is never a bare uuid the user has to go and look up themselves -- an
    # answer you cannot check is the failure mode the whole citation contract exists to
    # prevent. Best-effort: a missing graph costs the links, not the answer.
    # Prefer what the model cited; fall back to what its tools showed it, capped so an
    # uncited answer does not dump every hit it ever saw into the panel.
    shown = result["citations"] or result.get("sources", [])[:6]
    citations = []
    if shown:
        try:
            # resolve(), not expand(): the graph is rebuilt by an explicit sync and is
            # routinely behind the corpus, so anything processed since the last one
            # expands to nothing. Falling back to Postgres for the quote, speaker and clip
            # offset is what keeps a cited id clickable on a stale projection.
            citations = await graph_context.resolve(shown)
        except Exception:
            citations = []

    # Shape, not content. `LOG_TRANSCRIPTS` is false by default and audit_log is
    # append-only, so writing user text here would durably retain something the rest of the
    # product deliberately does not. The citations are the useful part regardless: they
    # make every answer traceable back to source audio.
    await audit("agent.ask", "conversation", conversation_id, actor=user, after={
        "tools_used": result["tools_used"],
        "citations": result["citations"],
        "refused": result["refused"],
        "usage": result["usage"],
        "model": cfg.llm_model,
        "question": body.question if cfg.log_transcripts else None,
    })

    return {
        "conversation_id": conversation_id,
        "citation_details": citations,
        "ws_url": f"/v1/ws/jobs/{conversation_id}",
        **result,
        # Echoed back so the next turn can be sent without the client reconstructing it.
        "history": (body.history or []) + [
            {"role": "user", "content": body.question},
            {"role": "assistant", "content": result["answer"]},
        ] if result["answer"] else (body.history or []),
    }
