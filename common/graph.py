"""Thin neo4j wrapper, mirroring common/db.py. No OGM — the Cypher is the interesting
part and mapping it through an object layer would be more indirection than code.

Two deliberate differences from db.py:

- The driver connects *lazily*, on first query, and reads its own config rather than
  being handed a pool at startup. Neo4j holds a derived read model (see
  GRAPH_RAG_PLAN.md); the api must start, serve uploads, run search and drive the
  pipeline whether or not it is reachable. Eager connection in the lifespan would make
  an optional datastore a hard startup dependency.
- `neo4j` is imported inside the connect path, not at module scope, so common/ stays
  importable from the worker — which shares this package but has no reason to carry a
  graph client in its dependency set.
"""
import asyncio

from .config import get_settings

_driver = None
_lock = asyncio.Lock()


class GraphUnavailable(RuntimeError):
    """Neo4j is disabled or unreachable. Callers turn this into a 503 rather than a 500 —
    it is a missing optional dependency, not a bug in the request."""


async def driver():
    global _driver
    cfg = get_settings()
    if not cfg.graph_enabled:
        raise GraphUnavailable("graph is disabled (set GRAPH_ENABLED=true)")
    if _driver is None:
        async with _lock:
            if _driver is None:
                from neo4j import AsyncGraphDatabase
                _driver = AsyncGraphDatabase.driver(
                    cfg.graph_uri, auth=(cfg.graph_user, cfg.graph_password))
    return _driver


async def close():
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


# Transport- and auth-level failures, as opposed to a bad query. These mean the database
# is unreachable or misconfigured -- a deployment state the caller should see as 503 --
# whereas a Cypher error is a bug in this code and must keep surfacing as a 500.
def _unavailable_errors():
    from neo4j import exceptions as e
    return (e.ServiceUnavailable, e.SessionExpired, e.AuthError, e.ConfigurationError, OSError)


async def _reset():
    """Drop the cached driver so the next call reconnects.

    The driver is cached for the process lifetime, so a Neo4j restart leaves it holding a
    dead socket and every later call fails on a closed connection until the api itself is
    restarted. Observed exactly that after the database container was recreated. Clearing
    it here makes the next request re-dial instead."""
    global _driver
    d, _driver = _driver, None
    if d is not None:
        try:
            await d.close()
        except Exception:
            pass


async def run(cypher: str, **params) -> list[dict]:
    d = await driver()
    try:
        async with d.session() as s:
            res = await s.run(cypher, **params)
            return [r.data() async for r in res]
    except _unavailable_errors() as e:
        await _reset()
        raise GraphUnavailable(f"neo4j is unreachable: {type(e).__name__}") from e


async def write(cypher: str, rows: list[dict], batch: int = 1000) -> int:
    """Run an `UNWIND $rows AS row ...` write in batches.

    Every projection is one statement over many rows rather than one statement per row:
    at a few thousand utterances the difference is a couple of round trips against a
    couple of thousand."""
    if not rows:
        return 0
    d = await driver()
    try:
        async with d.session() as s:
            for i in range(0, len(rows), batch):
                await s.run(cypher, rows=rows[i:i + batch])
    except _unavailable_errors() as e:
        await _reset()
        raise GraphUnavailable(f"neo4j is unreachable: {type(e).__name__}") from e
    return len(rows)
