from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ..auth import get_current_user
from ..services import search as search_svc

router = APIRouter(prefix="/v1/search", tags=["search"])


class SearchBody(BaseModel):
    q: str
    mode: str = "hybrid"  # fts | semantic | hybrid
    speaker_id: str | None = None
    limit: int = 25


@router.post("")
async def search(body: SearchBody, user=Depends(get_current_user)):
    # Ranking lives in services/search.py so the graph-context endpoint anchors on exactly
    # this, rather than growing a second implementation that drifts from it.
    items = await search_svc.hybrid_search(body.q, body.mode, body.speaker_id, body.limit)
    return {"items": items}
