from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from common.graph import GraphUnavailable
from ..auth import get_current_user
from ..services import graph_context, search as search_svc

router = APIRouter(prefix="/v1/graph", tags=["graph"])


class ContextBody(BaseModel):
    q: str
    limit: int = 8
    mode: str = "hybrid"          # fts | semantic | hybrid, as on /v1/search
    speaker_id: str | None = None
    # Explicit anchors, skipping the search step. For a caller that already knows which
    # speech it cares about -- the clip page asking "what surrounds this utterance" --
    # searching for text it is already holding would be a round trip to find itself.
    speech_ids: list[str] | None = None
    max_chars: int | None = None


@router.post("/context")
async def context(body: ContextBody, user=Depends(get_current_user)):
    """Anchor on the corpus, expand the neighbourhood in the graph, render it as text.

    Returns both the rendered block and the structured rows: the UI draws from the same
    call that a model would read, so the two can never show different neighbourhoods."""
    if body.speech_ids:
        anchors = [{"speech_id": sid, "kind": None, "score": None} for sid in body.speech_ids]
    else:
        anchors = await search_svc.anchor_speech(
            body.q, limit=body.limit, mode=body.mode, speaker_id=body.speaker_id)

    if not anchors:
        return {"anchors": [], "items": [], "context": ""}

    try:
        items = await graph_context.expand([a["speech_id"] for a in anchors])
    except GraphUnavailable as e:
        # 503, not 500: the graph being off is a deployment state. The message names the
        # setting, because the likeliest cause is that nobody has run a sync yet.
        raise HTTPException(503, str(e)) from e

    return {"anchors": anchors, "items": items,
            "context": graph_context.render_context(items, body.max_chars)}
