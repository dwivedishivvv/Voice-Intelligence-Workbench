"""Graph-RAG retrieval: anchor in Postgres, expand in Neo4j, render for a reader.

The split is deliberate (GRAPH_RAG_PLAN.md, Phase 3). Ranking stays in Postgres, where
the hybrid index already lives; the graph is used only for what it is actually better at,
pulling the neighbourhood of an already-chosen node in one round trip. Embeddings are
never copied into Neo4j — that would be the same vectors in two stores with no added
capability.

`render_context` is the part that decides whether any of this is useful. A JSON dump of
the expansion is expensive to read and easy to misread; short natural-language lines are
not. It is a pure function so it can be tested without either database.
"""
from common import db, graph

# One round trip. Every relationship is OPTIONAL: a manually uploaded clip has no session
# and no lap, an unenrolled voice has no speaker, and a partial neighbourhood is the
# normal case rather than an error.
#
# collect() drops nulls, and the CASE keeps that true for the map projections -- without
# it a speech with no lap would collect a map full of nulls instead of an empty list.
EXPAND_CYPHER = """
UNWIND $ids AS sid
MATCH (sp:Speech {speech_id: sid})
OPTIONAL MATCH (sp)-[:SPOKEN_BY]->(spk:Speaker)
OPTIONAL MATCH (sp)-[:IN_CLIP]->(clip:Clip)
OPTIONAL MATCH (sp)-[:BY_DRIVER]->(drv:Driver)
OPTIONAL MATCH (sp)-[:IN_SESSION]->(sess:Session)
OPTIONAL MATCH (prev:Speech)-[:NEXT]->(sp)
OPTIONAL MATCH (sp)-[:NEXT]->(nxt:Speech)
OPTIONAL MATCH (sp)-[:DURING_LAP]->(lap:Lap)
OPTIONAL MATCH (sp)-[:MENTIONS]->(ent)
RETURN sid AS speech_id,
       sp.text AS text, sp.mood AS mood, sp.start_s AS start_s,
       sp.sentiment AS sentiment, sp.sentiment_score AS sentiment_score,
       sp.text_sentiment AS text_sentiment,
       spk.name AS speaker,
       clip.clip_id AS clip_id, clip.filename AS filename,
       coalesce(drv.code, drv.full_name) AS driver, drv.full_name AS driver_name,
       drv.team AS team,
       sess.name AS session, sess.year AS year, sess.circuit AS circuit,
       prev.text AS prev_text, nxt.text AS next_text,
       collect(DISTINCT CASE WHEN lap IS NULL THEN null ELSE {
         number: lap.number,
         duration_s: lap.duration_s,
         prev_s: head([(p:Lap)-[:NEXT]->(lap) | p.duration_s]),
         next_s: head([(lap)-[:NEXT]->(n:Lap) | n.duration_s])
       } END) AS laps,
       collect(DISTINCT CASE WHEN ent IS NULL THEN null ELSE {
         kind: head(labels(ent)),
         name: coalesce(ent.code, ent.full_name, ent.name)
       } END) AS mentions
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
# Speaker resolution mirrors the utterances view: clip_speakers is the authoritative
# identity row and utterances.profile_id can lag it, so either may carry the name.
RESOLVE_UTTERANCES_SQL = """
SELECT u.id::text AS speech_id, u.text, u.mood, u.start_s, u.end_s,
       u.sentiment, u.sentiment_score, u.text_sentiment,
       p.display_name AS speaker, u.clip_id::text AS clip_id, c.filename
  FROM utterances u
  JOIN clips c ON c.id = u.clip_id
  LEFT JOIN clip_speakers cs ON cs.clip_id = u.clip_id AND cs.local_label = u.local_label
  LEFT JOIN speaker_profiles p ON p.id = coalesce(u.profile_id, cs.profile_id)
 WHERE u.id = ANY($1::uuid[])
"""

RESOLVE_RADIO_SQL = """
SELECT rc.id::text AS speech_id, rc.text, rc.mood, rc.clip_id::text AS clip_id,
       d.name_acronym AS driver, d.full_name AS driver_name, d.team_name AS team,
       s.name AS session, s.year, s.circuit
  FROM radio_calls rc
  LEFT JOIN f1_drivers d
         ON d.driver_number = rc.driver_number AND d.session_key = rc.session_key
  LEFT JOIN f1_sessions s ON s.session_key = rc.session_key
 WHERE rc.id = ANY($1::uuid[])
"""


async def resolve(speech_ids: list[str]) -> list[dict]:
    """Everything known about each id, graph first, Postgres for the rest.

    The graph is a *derived* read model, rebuilt by an explicit sync — so it is routinely
    behind the corpus, and any id processed since the last sync expands to nothing. That
    is tolerable for the lap and entity edges, which only the graph has. It is not
    tolerable for the quote, the speaker and the clip offset: those are what turn a cited
    id into a link the reader can check, and resolving them only through the graph meant a
    stale projection silently produced an answer whose citations went nowhere.

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
        for sql in (RESOLVE_UTTERANCES_SQL, RESOLVE_RADIO_SQL):
            if not missing:
                break
            try:
                rows = await db.fetch(sql, missing)
            except Exception:
                continue
            for r in rows:
                found[r["speech_id"]] = {**dict(r), "laps": [], "mentions": []}
            missing = [sid for sid in missing if sid not in found]

    return [found[sid] for sid in speech_ids if sid in found]


def _tone_phrase(row: dict) -> str:
    """Tone and sentiment, phrased as readings rather than facts.

    worker/worker/audio/tone.py is a threshold heuristic over acoustic features, not a
    trained speech-emotion model, and the pipeline stores the text and acoustic reads
    separately precisely so they can disagree. Both of those get flattened into a
    confident-sounding assertion the moment this is worded as 'he was stressed', so the
    hedging lives here — in the one place every consumer, UI or model, reads through."""
    parts = []
    if row.get("mood"):
        parts.append(f"voice reads {row['mood']}")
    if row.get("sentiment"):
        score = row.get("sentiment_score")
        s = f"words read {row['sentiment']}"
        if score is not None:
            s += f" ({score:+.2f})"
        parts.append(s)
    # Surface disagreement rather than averaging it away -- a calm-sounding delivery of
    # negative words is a signal, and it is exactly what a single fused number destroys.
    text_s, fused = row.get("text_sentiment"), row.get("sentiment")
    if text_s and fused and text_s != fused:
        parts.append(f"text alone reads {text_s}")
    return ", ".join(parts)


def _lap_phrase(lap: dict) -> str:
    """One lap, with its delta to the previous lap — the comparison that makes a lap time
    mean anything without the reader having to hold the whole timing sheet in their head."""
    out = f"lap {lap['number']}"
    if lap.get("duration_s") is not None:
        out += f": {lap['duration_s']:.1f}s"
        if lap.get("prev_s") is not None:
            out += f" ({lap['duration_s'] - lap['prev_s']:+.1f}s vs previous)"
    return out


def render_context(rows: list[dict], max_chars: int | None = None) -> str:
    """Flatten the expansion into short lines a person (or a model) can read.

    Every block leads with its speech id: that is what makes a downstream claim citable
    back to source audio, and an answer that cannot cite is an answer that cannot be
    checked."""
    blocks: list[str] = []
    for r in rows:
        head_bits = []
        if r.get("session"):
            head_bits.append(" ".join(str(b) for b in (r.get("session"), r.get("year")) if b))
        elif r.get("circuit"):
            head_bits.append(r["circuit"])
        who = r.get("driver") or r.get("speaker") or "unidentified speaker"
        if r.get("team"):
            who += f" ({r['team']})"
        head_bits.append(who)

        lines = [f"[{r['speech_id']}] " + " | ".join(b for b in head_bits if b)]
        tone = _tone_phrase(r)
        if tone:
            lines.append(f"  tone: {tone}")
        if r.get("text"):
            lines.append(f'  "{r["text"].strip()}"')
        for lap in r.get("laps") or []:
            lines.append(f"  {_lap_phrase(lap)}")
        mentions = [m for m in (r.get("mentions") or []) if m.get("name")]
        if mentions:
            lines.append("  mentions: "
                         + ", ".join(f"{m['name']} ({str(m['kind']).lower()})" for m in mentions))
        if r.get("prev_text"):
            lines.append(f'  before: "{r["prev_text"].strip()}"')
        if r.get("next_text"):
            lines.append(f'  after: "{r["next_text"].strip()}"')
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


def render_brief(rows: list[dict], max_text: int = 160) -> str:
    """One line per search hit: id, who said it, where, and a clipped quote.

    Deliberately thinner than render_context - search is for choosing what to look at,
    and a caller that can answer straight from the search output will do exactly that,
    from a truncated quote with none of its surroundings.

    The speaker is the one piece of context that belongs here rather than behind an
    expand call. Without it a question of the form "what did X say" is unanswerable from
    search output: every hit looks equally plausible. Observed against real models - they
    picked the first result and answered from it, citing an utterance that merely
    *mentioned* the driver rather than one he spoke. It costs a few tokens a line and
    closes the most common way this retrieves the wrong thing."""
    lines = []
    for r in rows:
        text = " ".join((r.get("text") or "").split())
        if len(text) > max_text:
            text = text[:max_text - 1].rstrip() + "…"
        who = r.get("driver") or r.get("speaker") or "unidentified speaker"
        where = f", {r['session']}" if r.get("session") else ""
        laps = r.get("laps") or []
        lap = f", lap {laps[0]['number']}" if laps and laps[0].get("number") is not None else ""
        # The tone reading rides along on the search line, hedged the same way it is
        # everywhere else. Without it a tone-filtered search returns hits that look like
        # plain quotes, and the model reports them with the qualifier dropped.
        tone = f", voice reads {r['mood']}" if r.get("mood") else ""
        lines.append(f'[{r["speech_id"]}] {who}{where}{lap}{tone}: "{text}"')
    return chr(10).join(lines)


DRIVER_LAPS_CYPHER = """
MATCH (l:Lap {session_key: $session_key, driver_number: $driver_number})
OPTIONAL MATCH (sp:Speech)-[:DURING_LAP]->(l)
WITH l, collect(DISTINCT CASE WHEN sp IS NULL THEN null ELSE {
       speech_id: sp.speech_id, text: sp.text, mood: sp.mood, sentiment: sp.sentiment
     } END) AS speech
WHERE ($from_lap IS NULL OR l.number >= $from_lap)
  AND ($to_lap   IS NULL OR l.number <= $to_lap)
RETURN l.number AS number, l.duration_s AS duration_s, l.is_pit_out AS is_pit_out, speech
ORDER BY l.number
"""


async def driver_laps(session_key: int, driver_number: int,
                      from_lap: int | None = None, to_lap: int | None = None) -> list[dict]:
    return await graph.run(DRIVER_LAPS_CYPHER, session_key=session_key,
                           driver_number=driver_number, from_lap=from_lap, to_lap=to_lap)


def render_timeline(laps: list[dict], from_lap: int | None = None,
                    to_lap: int | None = None) -> str:
    """A driver's session as lap times with the speech that landed on each lap.

    With no explicit range, only laps carrying speech are listed. A full race is 50-70 laps
    and most of them are silent; printing them all buries the handful that matter and costs
    the reader (or the context window) far more than it informs. The pace summary is what
    keeps the surviving laps interpretable without the rest."""
    if not laps:
        return ""
    timed = [l["duration_s"] for l in laps if l.get("duration_s") is not None]
    header = f"{len(laps)} laps"
    if timed:
        fastest = min(timed)
        header += (f" | fastest {fastest:.1f}s"
                   f" | median {sorted(timed)[len(timed) // 2]:.1f}s"
                   f" | slowest {max(timed):.1f}s")

    explicit_range = from_lap is not None or to_lap is not None
    shown = laps if explicit_range else [l for l in laps if l.get("speech")]
    if not shown:
        return header + "\n  (no radio landed on any lap in this session)"

    prev_by_number = {l["number"]: l.get("duration_s") for l in laps}
    lines = [header]
    for l in shown:
        line = f"  lap {l['number']}"
        if l.get("duration_s") is not None:
            line += f": {l['duration_s']:.1f}s"
            prev = prev_by_number.get(l["number"] - 1)
            if prev is not None:
                line += f" ({l['duration_s'] - prev:+.1f}s)"
        if l.get("is_pit_out"):
            line += " [out of pits]"
        lines.append(line)
        for s in l.get("speech") or []:
            tone = f" ({s['mood']})" if s.get("mood") else ""
            lines.append(f'    [{s["speech_id"]}]{tone} "{" ".join((s.get("text") or "").split())}"')
    return "\n".join(lines)
