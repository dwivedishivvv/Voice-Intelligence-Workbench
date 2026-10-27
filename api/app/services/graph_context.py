"""Graph-RAG retrieval: anchor in Postgres, expand in Neo4j, render for a reader.

The split is deliberate (GRAPH_RAG_PLAN.md, Phase 3). Ranking stays in Postgres, where
the hybrid index already lives; the graph is used only for what it is actually better at,
pulling the neighbourhood of an already-chosen node in one round trip. Embeddings are
never copied into Neo4j — that would be the same vectors in two stores with no added
capability.

`render_context` is the part that decides whether any of this is useful. A JSON dump of
the expansion is expensive to read and easy to misread; short natural-language lines are
not. It is a pure function so it can be tested without either database.

`corpus_map` is the other reader: the whole graph minus individual lines of speech, shaped
for the Graph page to draw.
"""
from common import db, graph

# One round trip. Every relationship is OPTIONAL: an unenrolled voice has no speaker, a
# line that names nothing has no entities, and a partial neighbourhood is the normal case
# rather than an error.
#
# collect() drops nulls, and the CASE keeps that true for the map projections -- without
# it a speech that names nothing would collect a map full of nulls instead of an empty list.
EXPAND_CYPHER = """
UNWIND $ids AS sid
MATCH (sp:Speech {speech_id: sid})
OPTIONAL MATCH (sp)-[:SPOKEN_BY]->(spk:Speaker)
OPTIONAL MATCH (sp)-[:IN_CLIP]->(clip:Clip)
OPTIONAL MATCH (prev:Speech)-[:NEXT]->(sp)
OPTIONAL MATCH (sp)-[:NEXT]->(nxt:Speech)
OPTIONAL MATCH (sp)-[:MENTIONS]->(ent:Entity)
OPTIONAL MATCH (clip)-[:ABOUT]->(topic:Topic)
RETURN sid AS speech_id,
       sp.text AS text, sp.start_s AS start_s, sp.end_s AS end_s,
       spk.name AS speaker,
       clip.clip_id AS clip_id, clip.filename AS filename,
       prev.text AS prev_text, nxt.text AS next_text,
       head([(prev)-[:SPOKEN_BY]->(ps:Speaker) | ps.name]) AS prev_speaker,
       head([(nxt)-[:SPOKEN_BY]->(ns:Speaker) | ns.name]) AS next_speaker,
       collect(DISTINCT CASE WHEN ent IS NULL THEN null ELSE {
         kind: ent.kind, name: ent.name
       } END) AS mentions,
       collect(DISTINCT topic.name) AS topics
"""


async def expand(speech_ids: list[str]) -> list[dict]:
    """Neighbourhood of each anchor, in graph order."""
    if not speech_ids:
        return []
    rows = await graph.run(EXPAND_CYPHER, ids=speech_ids)
    by_id = {r["speech_id"]: r for r in rows}
    # Preserve the ranking the anchor step produced; Cypher makes no ordering promise.
    return [by_id[sid] for sid in speech_ids if sid in by_id]


# The same fields EXPAND_CYPHER returns, from the tables the graph is projected *from*.
# Speaker resolution: clip_speakers is the authoritative identity row and
# utterances.profile_id can lag it, so either may carry the name.
RESOLVE_UTTERANCES_SQL = """
SELECT u.id::text AS speech_id, u.text, u.start_s, u.end_s,
       p.display_name AS speaker, u.clip_id::text AS clip_id, c.filename
  FROM utterances u
  JOIN clips c ON c.id = u.clip_id
  LEFT JOIN clip_speakers cs ON cs.clip_id = u.clip_id AND cs.local_label = u.local_label
  LEFT JOIN speaker_profiles p ON p.id = coalesce(u.profile_id, cs.profile_id)
 WHERE u.id = ANY($1::uuid[])
"""


async def resolve(speech_ids: list[str]) -> list[dict]:
    """Everything known about each id, graph first, Postgres for the rest.

    The graph is a *derived* read model, rebuilt by an explicit sync — so it is routinely
    behind the corpus, and any id processed since the last sync expands to nothing. That
    is tolerable for the entity edges, which only the graph has. It is not tolerable for
    the quote, the speaker and the clip offset: those are what turn a cited id into a link
    the reader can check, and resolving them only through the graph meant a stale
    projection silently produced an answer whose citations went nowhere.

    Graph rows win where they exist, because they carry the edges Postgres cannot.
    """
    if not speech_ids:
        return []
    found: dict[str, dict] = {}
    try:
        found = {r["speech_id"]: r for r in await expand(speech_ids)}
    except Exception:
        # Graph off or unreachable: a deployment state, not a failure of the request.
        pass

    missing = [sid for sid in speech_ids if sid not in found]
    if missing:
        try:
            rows = await db.fetch(RESOLVE_UTTERANCES_SQL, missing)
        except Exception:
            rows = []
        for r in rows:
            found[r["speech_id"]] = {**dict(r), "mentions": [], "topics": []}

    return [found[sid] for sid in speech_ids if sid in found]


def _clock(seconds) -> str:
    """Offset into a clip as m:ss -- the form a reader scrubs an audio player by."""
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


def _clip_phrase(row: dict) -> str:
    """Which recording a piece of speech is in.

    Most of a corpus is uploaded clips -- a meeting, a podcast, an interview -- and the
    filename is often the only statement of what a clip *is about* ("The Decline of
    Dunzo... Startup Case Study"). A model that cannot see it has no way to connect a quote
    to its subject or to tell two recordings apart when it summarises."""
    if not row.get("filename"):
        return ""
    out = f'clip "{row["filename"]}"'
    if row.get("start_s") is not None:
        out += f" at {_clock(row['start_s'])}"
    return out


def render_context(rows: list[dict], max_chars: int | None = None) -> str:
    """Flatten the expansion into short lines a person (or a model) can read.

    Every block leads with its speech id: that is what makes a downstream claim citable
    back to source audio, and an answer that cannot cite is an answer that cannot be
    checked."""
    blocks: list[str] = []
    for r in rows:
        head_bits = [_clip_phrase(r), r.get("speaker") or "unidentified speaker"]
        lines = [f"[{r['speech_id']}] " + " | ".join(b for b in head_bits if b)]
        if r.get("text"):
            lines.append(f'  "{r["text"].strip()}"')
        mentions = [m for m in (r.get("mentions") or []) if m.get("name")]
        if mentions:
            lines.append("  names: " + ", ".join(
                f"{m['name']} ({m['kind']})" if m.get("kind") and m["kind"] != "name"
                else m["name"] for m in mentions))
        if r.get("topics"):
            lines.append("  recording topics: " + ", ".join(r["topics"]))
        if r.get("prev_text"):
            who = f" ({r['prev_speaker']})" if r.get("prev_speaker") else ""
            lines.append(f'  before{who}: "{r["prev_text"].strip()}"')
        if r.get("next_text"):
            who = f" ({r['next_speaker']})" if r.get("next_speaker") else ""
            lines.append(f'  after{who}: "{r["next_text"].strip()}"')
        blocks.append("\n".join(lines))

    out = "\n\n".join(blocks)
    if max_chars is not None and len(out) > max_chars:
        # Truncate on a block boundary: half a block is a block whose speech id may be
        # missing, and an uncitable fragment is worse than one fewer result.
        kept: list[str] = []
        used = 0
        for b in blocks:
            if used + len(b) + 2 > max_chars:
                break
            kept.append(b)
            used += len(b) + 2
        out = "\n\n".join(kept)
    return out


def _snippet(text: str, max_text: int, terms: list[str] | None = None) -> str:
    """At most `max_text` characters of `text`, windowed on the first query term.

    Uploaded clips diarize into long turns -- a podcast monologue runs to a thousand
    characters per utterance -- and the opening 160 are often not the part that matched.
    Observed: asked who founded Dunzo, the search for "Kabir" returned the line that names
    him, clipped before the surname, and the model supplied one of its own. Showing the
    window around the match is what puts the matched words in front of the reader."""
    text = " ".join((text or "").split())
    if len(text) <= max_text:
        return text
    low = text.lower()
    hits = [p for p in (low.find(t.lower()) for t in terms or []) if p >= 0]
    start = max(0, min(hits) - max_text // 3) if hits else 0
    # Start on a word boundary so the window does not open mid-word.
    if start:
        start = text.find(" ", start) + 1 or start
    end = start + max_text - (1 if start else 0) - 1
    out = text[start:end].rstrip()
    return ("…" if start else "") + out + ("…" if end < len(text) else "")


def render_brief(rows: list[dict], max_text: int = 160,
                 terms: list[str] | None = None) -> str:
    """One line per search hit: id, who said it, where, and a clipped quote.

    Deliberately thinner than render_context - search is for choosing what to look at,
    and a caller that can answer straight from the search output will do exactly that,
    from a truncated quote with none of its surroundings.

    The speaker is the one piece of context that belongs here rather than behind an
    expand call. Without it a question of the form "what did X say" is unanswerable from
    search output: every hit looks equally plausible.

    `terms` are the query's words; when given, a long quote is windowed on the first of
    them rather than cut from the start (see _snippet)."""
    lines = []
    for r in rows:
        text = _snippet(r.get("text") or "", max_text, terms)
        who = r.get("speaker") or "unidentified speaker"
        where = f", {_clip_phrase(r)}" if _clip_phrase(r) else ""
        lines.append(f'[{r["speech_id"]}] {who}{where}: "{text}"')
    return chr(10).join(lines)


# --------------------------------------------------------------------------- the map

# Utterances are left out: a corpus of a few hours is thousands of lines, which is a
# hairball, not a map. Everything they connect is still drawn, through the aggregate edges
# (APPEARS_IN, MENTIONED, CLIP MENTIONS) the projection derives from them.
MAP_NODES_CYPHER = """
MATCH (n) WHERE NOT n:Utterance
WITH n, [l IN labels(n) WHERE l <> 'Speech'][0] AS label
RETURN elementId(n) AS id, label,
       coalesce(n.name, n.filename, n.code) AS name,
       coalesce(n.profile_id, n.clip_id, n.cluster_id) AS ref,
       n {.status, .n_enrollments, .total_speech_s, .duration_s, .language, .n_speakers,
          .needs_review, .quality, .kind, .n_mentions, .n_members, .notes, .tags} AS props,
       COUNT { (n)--(m) WHERE NOT m:Utterance } AS degree
ORDER BY degree DESC
LIMIT $limit
"""

MAP_LINKS_CYPHER = """
MATCH (a)-[r]->(b)
WHERE elementId(a) IN $ids AND elementId(b) IN $ids
RETURN elementId(a) AS source, elementId(b) AS target, type(r) AS type,
       coalesce(r.n_clips, r.n, r.speech_s, 1) AS weight
"""


def _plain(v):
    """Neo4j temporal and list values -> JSON-safe."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, list):
        return [_plain(x) for x in v]
    return str(v)


async def corpus_map(limit: int = 600) -> dict:
    """Nodes and links for the Graph page, best-connected first when capped."""
    nodes = await graph.run(MAP_NODES_CYPHER, limit=min(max(limit, 10), 3000))
    ids = [n["id"] for n in nodes]
    links = await graph.run(MAP_LINKS_CYPHER, ids=ids) if ids else []
    return {
        "nodes": [{**n, "props": {k: _plain(v) for k, v in (n["props"] or {}).items()
                                  if v is not None}} for n in nodes],
        "links": [{**l, "weight": float(l["weight"] or 1)} for l in links],
    }


NEIGHBOURHOOD_CYPHER = """
MATCH (n) WHERE elementId(n) = $id
OPTIONAL MATCH (n)-[r]-(m) WHERE NOT m:Utterance
WITH n, r, m ORDER BY coalesce(r.n_clips, r.n, r.speech_s, 1) DESC
WITH n, collect(CASE WHEN m IS NULL THEN null ELSE {
       id: elementId(m), label: [l IN labels(m) WHERE l <> 'Speech'][0],
       name: coalesce(m.name, m.filename, m.code),
       ref: coalesce(m.profile_id, m.clip_id, m.cluster_id),
       type: type(r), outgoing: startNode(r) = n,
       weight: coalesce(r.n_clips, r.n, r.speech_s, 1)
     } END)[..60] AS neighbours
OPTIONAL MATCH (u:Utterance)-[:SPOKEN_BY|IN_CLIP|MENTIONS]->(n)
WITH n, neighbours, u ORDER BY u.start_s
WITH n, neighbours, collect(u {.speech_id, .text, .start_s})[..5] AS lines
RETURN neighbours, lines
"""


async def neighbourhood(node_id: str) -> dict:
    rows = await graph.run(NEIGHBOURHOOD_CYPHER, id=node_id)
    if not rows:
        return {"neighbours": [], "lines": []}
    r = rows[0]
    return {"neighbours": [{**m, "weight": float(m["weight"] or 1)} for m in r["neighbours"]],
            "lines": r["lines"]}
