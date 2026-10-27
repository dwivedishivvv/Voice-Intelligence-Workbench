import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import {
  forceCenter, forceCollide, forceLink, forceManyBody, forceSimulation, forceX, forceY,
  type Simulation, type SimulationLinkDatum, type SimulationNodeDatum,
} from "d3-force";
import { api } from "@/api/client";
import { ACCENT, AMBER, AMBER_INK, GREEN, VOICE_A, VOICE_B, fmtSpeech, fmtTime, muted } from "@/lib/ui";

type RawNode = {
  id: string; label: string; name: string | null; ref: string | null; degree: number;
  props: Record<string, unknown>;
};
type RawLink = { source: string; target: string; type: string; weight: number };
type Node = RawNode & SimulationNodeDatum & { r: number };
type Link = SimulationLinkDatum<Node> & { type: string; weight: number };
type Neighbour = {
  id: string; label: string; name: string; ref: string | null; type: string;
  outgoing: boolean; weight: number;
};
type Detail = { neighbours: Neighbour[]; lines: { speech_id: string; text: string; start_s: number }[] };

/** One entry per node label the projection creates (common/graph_sync.py). Colours come
 *  from the same signal tokens the rest of the workbench uses, so a speaker is the same
 *  blue here as on a clip page, and both themes follow automatically. */
const KINDS: Record<string, { color: string; shape: "circle" | "square" | "diamond" | "ring"; noun: string; hint: string }> = {
  Speaker: { color: VOICE_A, shape: "circle", noun: "People", hint: "Enrolled speakers" },
  Voice: { color: VOICE_B, shape: "ring", noun: "Unnamed voices", hint: "Recurring voices nobody has named" },
  Clip: { color: ACCENT, shape: "square", noun: "Recordings", hint: "Processed clips" },
  Topic: { color: AMBER, shape: "diamond", noun: "Topics", hint: "Automatic topic tags" },
  Entity: { color: GREEN, shape: "circle", noun: "Named things", hint: "People, companies and places named in the transcripts" },
  Tag: { color: "var(--color-neutral-500)", shape: "diamond", noun: "Your tags", hint: "Tags you added" },
  Language: { color: "var(--color-neutral-700)", shape: "square", noun: "Languages", hint: "Detected languages" },
};
const kindOf = (label: string) => KINDS[label] || { color: muted(50), shape: "circle" as const, noun: label, hint: label };

/** How an edge reads in the side panel, from the selected node's point of view. */
const REL_WORDS: Record<string, [string, string]> = {
  APPEARS_IN: ["appears in", "features"],
  HEARD_IN: ["heard in", "has the voice"],
  TALKED_WITH: ["shares recordings with", "shares recordings with"],
  DISCUSSED: ["discusses", "discussed by"],
  ABOUT: ["is about", "topic of"],
  TAGGED: ["tagged", "tag on"],
  IN_LANGUAGE: ["in", "language of"],
  MENTIONS: ["mentions", "mentioned in"],
  MENTIONED: ["names", "named by"],
  CO_MENTIONED: ["named alongside", "named alongside"],
};

const W = 1200, H = 760;

function Shape({ n, fill, stroke, sw }: { n: Node; fill: string; stroke: string; sw: number }) {
  const k = kindOf(n.label);
  const r = n.r;
  if (k.shape === "square") return <rect x={-r} y={-r} width={r * 2} height={r * 2} fill={fill} stroke={stroke} strokeWidth={sw} />;
  if (k.shape === "diamond") return <rect x={-r * 0.9} y={-r * 0.9} width={r * 1.8} height={r * 1.8} transform="rotate(45)" fill={fill} stroke={stroke} strokeWidth={sw} />;
  if (k.shape === "ring") return <circle r={r} fill="var(--color-bg)" stroke={fill} strokeWidth={Math.max(2, sw + 1.5)} />;
  return <circle r={r} fill={fill} stroke={stroke} strokeWidth={sw} />;
}

export default function Graph() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const [raw, setRaw] = useState<{ nodes: RawNode[]; links: RawLink[] } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [query, setQuery] = useState("");
  const [hover, setHover] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [, setTick] = useState(0);
  const [view, setView] = useState({ x: 0, y: 0, k: 1 });

  const nodesRef = useRef<Node[]>([]);
  const linksRef = useRef<Link[]>([]);
  const simRef = useRef<Simulation<Node, Link> | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const drag = useRef<{ kind: "pan" | "node"; id?: string; sx: number; sy: number; vx: number; vy: number; moved: boolean } | null>(null);

  async function load() {
    setError(null);
    try {
      setRaw(await api.graphMap(800));
    } catch (e: any) {
      setRaw(null);
      setError(e?.status === 503
        ? "The graph database is not reachable. Start Neo4j (GRAPH_ENABLED=true) and sync."
        : String(e?.message || e));
    }
  }
  useEffect(() => { load(); }, []);

  async function sync() {
    setSyncing(true);
    try {
      const r = await api.graphSync();
      const n = Object.values(r.nodes || {}).reduce((a: number, b: any) => a + Number(b), 0);
      toast.success("Graph rebuilt", { description: `${n} nodes from the current corpus.` });
      await load();
    } catch (e) {
      toast.error("Sync failed", { description: String((e as Error).message) });
    } finally { setSyncing(false); }
  }

  // Build the simulation whenever the data or the visible kinds change. Positions of nodes
  // that survive a filter change are kept, so toggling a kind does not reshuffle the map.
  useEffect(() => {
    if (!raw) return;
    const prev = new Map(nodesRef.current.map((n) => [n.id, n]));
    const maxDeg = Math.max(1, ...raw.nodes.map((n) => n.degree));
    const nodes: Node[] = raw.nodes.filter((n) => !hidden.has(n.label)).map((n) => {
      const p = prev.get(n.id);
      const base = n.label === "Speaker" || n.label === "Clip" ? 9 : n.label === "Voice" ? 7 : 5;
      return { ...n, r: base + 11 * Math.sqrt(n.degree / maxDeg), x: p?.x, y: p?.y, vx: p?.vx, vy: p?.vy };
    });
    const ids = new Set(nodes.map((n) => n.id));
    const links: Link[] = raw.links.filter((l) => ids.has(l.source) && ids.has(l.target))
      .map((l) => ({ ...l }));
    nodesRef.current = nodes;
    linksRef.current = links;

    simRef.current?.stop();
    // CO_MENTIONED is a clique per recording (every name against every other), so it is
    // kept long and weak: at full strength it collapses the whole map into one knot.
    const sim = forceSimulation<Node, Link>(nodes)
      .force("link", forceLink<Node, Link>(links).id((d) => d.id)
        .distance((l) => (l.type === "CO_MENTIONED" ? 150 : l.type === "TALKED_WITH" ? 140 : 105))
        .strength((l) => (l.type === "CO_MENTIONED" ? 0.03 : 0.45)))
      .force("charge", forceManyBody<Node>().strength((d) => -180 - d.r * 22).distanceMax(700))
      .force("collide", forceCollide<Node>().radius((d) => d.r + 16).strength(0.9))
      .force("center", forceCenter(W / 2, H / 2))
      .force("x", forceX<Node>(W / 2).strength(0.025))
      .force("y", forceY<Node>(H / 2).strength(0.04))
      .stop();
    // Settle the layout before the first paint rather than animating it in: a map that
    // jiggles for two seconds is a map nobody can read yet. A few hundred ticks over a few
    // hundred nodes is well under a frame budget's worth of work on any machine.
    sim.alpha(prev.size ? 0.3 : 1).tick(prev.size ? 80 : 320);
    fit();
    // From here the simulation only runs while a node is being dragged.
    sim.alpha(0).on("tick", () => setTick((t) => t + 1));
    simRef.current = sim;
    return () => { sim.stop(); };
  }, [raw, hidden]);

  // ?focus=<profile_id | clip_id | cluster_id> opens that node, so other pages can link here.
  useEffect(() => {
    const focus = params.get("focus");
    if (!focus || !raw) return;
    const n = raw.nodes.find((x) => x.ref === focus);
    if (n) { setSelected(n.id); setTimeout(() => centreOn(n.id), 30); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [raw]);

  useEffect(() => {
    setDetail(null);
    if (!selected) return;
    api.graphNode(selected).then(setDetail).catch(() => setDetail({ neighbours: [], lines: [] }));
  }, [selected]);

  const nodes = nodesRef.current;
  const links = linksRef.current;
  const byId = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);

  // The neighbourhood of whatever is hovered (or else selected) stays lit; the rest dims.
  const focusId = hover || selected;
  const litEdges = useMemo(() => focusId ? links.filter((l) =>
    (l.source as Node).id === focusId || (l.target as Node).id === focusId).length : 0,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [focusId, links, nodes.length]);
  const lit = useMemo(() => {
    if (!focusId) return null;
    const s = new Set<string>([focusId]);
    for (const l of links) {
      const a = (l.source as Node).id, b = (l.target as Node).id;
      if (a === focusId) s.add(b);
      if (b === focusId) s.add(a);
    }
    return s;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [focusId, links, nodes.length]);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [] as RawNode[];
    return (raw?.nodes || []).filter((n) => (n.name || "").toLowerCase().includes(q)).slice(0, 8);
  }, [query, raw]);

  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const n of raw?.nodes || []) c[n.label] = (c[n.label] || 0) + 1;
    return c;
  }, [raw]);

  function fit() {
    const ns = nodesRef.current.filter((n) => n.x != null && n.y != null);
    if (!ns.length) return;
    const pad = 70;
    const x0 = Math.min(...ns.map((n) => n.x! - n.r)), x1 = Math.max(...ns.map((n) => n.x! + n.r));
    const y0 = Math.min(...ns.map((n) => n.y! - n.r)), y1 = Math.max(...ns.map((n) => n.y! + n.r));
    const k = Math.min(2.2, Math.max(0.25, Math.min((W - pad * 2) / (x1 - x0 || 1), (H - pad * 2) / (y1 - y0 || 1))));
    setView({ k, x: W / 2 - ((x0 + x1) / 2) * k, y: H / 2 - ((y0 + y1) / 2) * k });
  }

  function centreOn(id: string) {
    // read from the ref, not the render-time map: this runs from timers scheduled in the
    // same commit that built the nodes
    const n = nodesRef.current.find((x) => x.id === id);
    if (!n || n.x == null || n.y == null) return;
    setView((v) => ({ ...v, x: W / 2 - n.x! * v.k, y: H / 2 - n.y! * v.k }));
  }

  function pick(id: string) {
    setHidden((h) => {
      const label = raw?.nodes.find((n) => n.id === id)?.label;
      if (label && h.has(label)) { const n = new Set(h); n.delete(label); return n; }
      return h;
    });
    setSelected(id);
    setQuery("");
    setTimeout(() => centreOn(id), 30);
  }

  // ── pointer: pan the canvas, drag a node, wheel to zoom about the cursor ──
  const toLocal = (e: { clientX: number; clientY: number }) => {
    const r = svgRef.current!.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * W, y: ((e.clientY - r.top) / r.height) * H };
  };

  function onWheel(e: React.WheelEvent) {
    const p = toLocal(e);
    setView((v) => {
      const k = Math.min(4, Math.max(0.25, v.k * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
      return { k, x: p.x - ((p.x - v.x) / v.k) * k, y: p.y - ((p.y - v.y) / v.k) * k };
    });
  }

  function onPointerDown(e: React.PointerEvent, id?: string) {
    e.stopPropagation();
    (e.target as Element).setPointerCapture?.(e.pointerId);
    const p = toLocal(e);
    drag.current = { kind: id ? "node" : "pan", id, sx: p.x, sy: p.y, vx: view.x, vy: view.y, moved: false };
    if (id) {
      const n = byId.get(id);
      if (n) { n.fx = n.x; n.fy = n.y; }
      simRef.current?.alphaTarget(0.25).restart();
    }
  }

  function onPointerMove(e: React.PointerEvent) {
    const d = drag.current;
    if (!d) return;
    const p = toLocal(e);
    if (Math.abs(p.x - d.sx) + Math.abs(p.y - d.sy) > 3) d.moved = true;
    if (d.kind === "pan") {
      setView((v) => ({ ...v, x: d.vx + (p.x - d.sx), y: d.vy + (p.y - d.sy) }));
    } else if (d.id) {
      const n = byId.get(d.id);
      if (n) { n.fx = (p.x - view.x) / view.k; n.fy = (p.y - view.y) / view.k; }
    }
  }

  function onPointerUp() {
    const d = drag.current;
    drag.current = null;
    if (!d) return;
    if (d.kind === "node" && d.id) {
      const n = byId.get(d.id);
      // a click selects; a drag pins the node where it was dropped until double-clicked
      if (!d.moved && n) { n.fx = null; n.fy = null; setSelected(d.id === selected ? null : d.id); }
      simRef.current?.alphaTarget(0);
    } else if (!d.moved) {
      setSelected(null);
    }
  }

  const sel = selected ? (byId.get(selected) || raw?.nodes.find((n) => n.id === selected)) : null;
  const showLabel = (n: Node) =>
    view.k > 1.4 || n.label === "Speaker" || n.label === "Clip" || n.degree >= 4 || lit?.has(n.id) || n.id === selected;
  const linkPage = (n: { label: string; ref: string | null }) =>
    n.label === "Speaker" && n.ref ? `/speakers?id=${n.ref}`
    : n.label === "Clip" && n.ref ? `/clips/${n.ref}`
    : n.label === "Voice" ? "/speakers" : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <header style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 20, flexWrap: "wrap" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <span className="kicker">Knowledge graph</span>
          <h2 style={{ margin: 0, fontSize: 30 }}>Who, what, and where it was said</h2>
          <p style={{ margin: 0, maxWidth: "66ch", fontSize: 14, color: muted(68) }}>
            Every person, unnamed voice, recording, topic and named thing in the corpus, and how
            they connect. Drag to pan, scroll to zoom, click anything to see its neighbourhood.
            Rebuilt from Postgres — never the source of truth.
          </p>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <div style={{ position: "relative" }}>
            <input className="input" placeholder="Find a person, company, recording…" value={query}
                   onChange={(e) => setQuery(e.target.value)} style={{ width: 260 }}
                   onKeyDown={(e) => { if (e.key === "Enter" && matches[0]) pick(matches[0].id); }} />
            {matches.length > 0 && (
              <div style={{ position: "absolute", top: "100%", left: 0, right: 0, zIndex: 10, background: "var(--color-bg)",
                            border: "1px solid var(--color-text)", boxShadow: "var(--shadow-md)" }}>
                {matches.map((m) => (
                  <button key={m.id} className="row-hover" onClick={() => pick(m.id)}
                          style={{ display: "flex", gap: 8, alignItems: "center", width: "100%", padding: "7px 10px",
                                   border: 0, background: "none", font: "inherit", cursor: "pointer", textAlign: "left", color: "inherit" }}>
                    <span style={{ width: 8, height: 8, flex: "none", background: kindOf(m.label).color }} />
                    <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontSize: 13 }}>{m.name}</span>
                    <span className="mono" style={{ fontSize: 10.5, color: muted(50) }}>{m.label}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
          <button className="btn btn-secondary" style={{ fontSize: 12, padding: "6px 11px" }} disabled={syncing} onClick={sync}>
            {syncing ? "Rebuilding…" : "Rebuild graph"}
          </button>
        </div>
      </header>

      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        {Object.entries(KINDS).filter(([k]) => counts[k]).map(([k, v]) => {
          const off = hidden.has(k);
          return (
            <button key={k} title={v.hint} onClick={() => setHidden((h) => { const n = new Set(h); off ? n.delete(k) : n.add(k); return n; })}
                    className="tag" style={{ display: "inline-flex", alignItems: "center", gap: 7, cursor: "pointer", font: "inherit",
                                             fontSize: 12, padding: "4px 9px", background: off ? "transparent" : "var(--color-surface)",
                                             border: `1px solid ${off ? "var(--color-divider)" : v.color}`, opacity: off ? 0.5 : 1, color: "inherit" }}>
              <svg width="12" height="12" viewBox="-7 -7 14 14" aria-hidden>
                <Shape n={{ label: k, r: 5 } as Node} fill={v.color} stroke="none" sw={0} />
              </svg>
              {v.noun}
              <span className="mono" style={{ fontSize: 10.5, color: muted(55) }}>{counts[k]}</span>
            </button>
          );
        })}
        <span style={{ flex: 1 }} />
        <span className="mono" style={{ fontSize: 11, color: muted(50), alignSelf: "center" }}>
          {raw ? `${nodes.length} nodes · ${links.length} links` : ""}
        </span>
      </div>

      <section style={{ display: "grid", gridTemplateColumns: sel ? "minmax(0,1fr) 340px" : "1fr", gap: 16, alignItems: "start" }}>
        <div className="blueprint" style={{ position: "relative", overflow: "hidden", aspectRatio: `${W} / ${H}`,
                                             background: "var(--color-surface)" }}>
          {error && (
            <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", padding: 30, textAlign: "center" }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 10, maxWidth: 420 }}>
                <span className="kicker" style={{ color: AMBER_INK }}>Graph unavailable</span>
                <span style={{ fontSize: 14, color: muted(70) }}>{error}</span>
                <button className="btn btn-secondary" style={{ alignSelf: "center" }} onClick={load}>Try again</button>
              </div>
            </div>
          )}
          {raw && raw.nodes.length === 0 && (
            <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", padding: 30, textAlign: "center" }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 10, maxWidth: 420 }}>
                <span className="kicker">Nothing projected yet</span>
                <span style={{ fontSize: 14, color: muted(70) }}>
                  Process a recording, then rebuild — the graph fills in from what the pipeline found.
                </span>
                <button className="btn btn-primary" style={{ alignSelf: "center" }} disabled={syncing} onClick={sync}>Rebuild graph</button>
              </div>
            </div>
          )}
          {!raw && !error && (
            <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", color: muted(55), fontSize: 13 }}>
              Reading the graph…
            </div>
          )}
          <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Knowledge graph of the corpus"
               style={{ width: "100%", height: "100%", display: "block", cursor: drag.current?.kind === "pan" ? "grabbing" : "grab", touchAction: "none" }}
               onWheel={onWheel} onPointerDown={(e) => onPointerDown(e)} onPointerMove={onPointerMove}
               onPointerUp={onPointerUp} onPointerLeave={onPointerUp}>
            <defs>
              <pattern id="graph-grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="var(--color-divider)" strokeWidth="0.6" />
              </pattern>
            </defs>
            <rect width={W} height={H} fill="url(#graph-grid)" />
            <g transform={`translate(${view.x},${view.y}) scale(${view.k})`}>
              {links.map((l, i) => {
                const a = l.source as Node, b = l.target as Node;
                if (a.x == null || b.x == null) return null;
                const on = lit ? lit.has(a.id) && lit.has(b.id) && (a.id === focusId || b.id === focusId) : false;
                return (
                  <g key={i}>
                    <line x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                          stroke={on ? "var(--color-accent)" : "var(--color-text)"}
                          strokeOpacity={lit ? (on ? 0.75 : 0.04) : 0.16}
                          strokeWidth={(on ? 1.6 : 1) * Math.min(3, 0.8 + Math.log2(1 + l.weight) * 0.5) / Math.sqrt(view.k)}
                          strokeDasharray={l.type === "CO_MENTIONED" ? "3 3" : undefined} />
                    {/* edge names only while they stay legible; past a handful they are noise */}
                    {on && litEdges <= 6 && (
                      <text x={(a.x + b.x!) / 2} y={(a.y! + b.y!) / 2 - 4} textAnchor="middle" className="mono"
                            style={{ fontSize: 9 / view.k, fill: "var(--color-accent-700)", pointerEvents: "none" }}>
                        {l.type}
                      </text>
                    )}
                  </g>
                );
              })}
              {nodes.map((n) => {
                if (n.x == null || n.y == null) return null;
                const k = kindOf(n.label);
                const dim = lit && !lit.has(n.id);
                const isSel = n.id === selected;
                return (
                  <g key={n.id} transform={`translate(${n.x},${n.y})`} style={{ cursor: "pointer", opacity: dim ? 0.15 : 1, transition: "opacity .15s" }}
                     onPointerDown={(e) => onPointerDown(e, n.id)}
                     onPointerEnter={() => !drag.current && setHover(n.id)} onPointerLeave={() => setHover(null)}
                     onDoubleClick={(e) => { e.stopPropagation(); n.fx = null; n.fy = null; simRef.current?.alpha(0.3).restart(); }}>
                    {isSel && <circle r={n.r + 7} fill="none" stroke="var(--color-accent)" strokeWidth={2 / view.k} />}
                    <Shape n={n} fill={k.color} stroke="var(--color-bg)" sw={1.5 / view.k} />
                    {showLabel(n) && (
                      <text y={n.r + 12 / view.k} textAnchor="middle"
                            style={{ fontSize: (n.label === "Speaker" ? 12.5 : 11) / view.k, pointerEvents: "none",
                                     fontFamily: n.label === "Speaker" || n.label === "Clip" ? "var(--font-heading)" : "var(--font-body)",
                                     fill: "var(--color-text)", paintOrder: "stroke", stroke: "var(--color-surface)",
                                     strokeWidth: 3 / view.k }}>
                        {(n.name || "").length > 34 ? `${(n.name || "").slice(0, 32)}…` : n.name}
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          </svg>
          <div style={{ position: "absolute", right: 10, bottom: 10, display: "flex", gap: 4 }}>
            {([["+", 1.25], ["−", 0.8]] as const).map(([l, f]) => (
              <button key={l} className="btn btn-secondary btn-icon" style={{ width: 28, height: 28 }} aria-label={l === "+" ? "Zoom in" : "Zoom out"}
                      onClick={() => setView((v) => {
                        const k = Math.min(4, Math.max(0.25, v.k * f));
                        return { k, x: W / 2 - ((W / 2 - v.x) / v.k) * k, y: H / 2 - ((H / 2 - v.y) / v.k) * k };
                      })}>{l}</button>
            ))}
            <button className="btn btn-secondary" style={{ height: 28, fontSize: 11.5, padding: "0 9px" }}
                    onClick={fit}>Fit</button>
          </div>
        </div>

        {sel && (
          <aside className="blueprint" style={{ padding: 16, display: "flex", flexDirection: "column", gap: 14, position: "sticky", top: 16,
                                                maxHeight: "calc(100vh - 40px)", overflowY: "auto" }}>
            <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
              <svg width="22" height="22" viewBox="-11 -11 22 22" aria-hidden style={{ flex: "none", marginTop: 3 }}>
                <Shape n={{ label: sel.label, r: 8 } as Node} fill={kindOf(sel.label).color} stroke="none" sw={0} />
              </svg>
              <div style={{ flex: 1, minWidth: 0 }}>
                <span className="kicker-sm" style={{ color: muted(55) }}>
                  {sel.label}{sel.props?.kind && sel.props.kind !== "name" ? ` · ${String(sel.props.kind)}` : ""}
                </span>
                <h3 style={{ margin: "2px 0 0", fontSize: 21, lineHeight: 1.15, wordBreak: "break-word" }}>{sel.name}</h3>
              </div>
              <button className="btn btn-ghost btn-icon" aria-label="Close" style={{ width: 26, height: 26 }}
                      onClick={() => { setSelected(null); if (params.get("focus")) { params.delete("focus"); setParams(params, { replace: true }); } }}>✕</button>
            </div>

            <div className="mono" style={{ display: "flex", flexWrap: "wrap", gap: "4px 14px", fontSize: 11.5, color: muted(60) }}>
              <span>{sel.degree} connection{sel.degree === 1 ? "" : "s"}</span>
              {sel.props?.duration_s != null && <span>{fmtSpeech(Number(sel.props.duration_s))} long</span>}
              {sel.props?.total_speech_s != null && <span>{fmtSpeech(Number(sel.props.total_speech_s))} of speech</span>}
              {sel.props?.n_mentions != null && <span>named {String(sel.props.n_mentions)}×</span>}
              {sel.props?.status != null && <span>{String(sel.props.status)}</span>}
              {sel.props?.language != null && <span>{String(sel.props.language).toUpperCase()}</span>}
              {sel.props?.quality != null && <span>audio {String(sel.props.quality)}</span>}
            </div>
            {typeof sel.props?.notes === "string" && sel.props.notes && (
              <p style={{ margin: 0, fontSize: 13, color: muted(72), whiteSpace: "pre-wrap" }}>{sel.props.notes}</p>
            )}

            {linkPage(sel) && (
              <button className="btn btn-primary" style={{ fontSize: 12.5, alignSelf: "flex-start" }}
                      onClick={() => navigate(linkPage(sel)!)}>
                {sel.label === "Clip" ? "Open recording" : sel.label === "Speaker" ? "Open speaker" : "Name this voice"} →
              </button>
            )}

            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="kicker-sm" style={{ color: muted(55) }}>Connections</span>
              {!detail && <span style={{ fontSize: 12.5, color: muted(55) }}>Loading…</span>}
              {detail?.neighbours.length === 0 && <span style={{ fontSize: 12.5, color: muted(55) }}>None.</span>}
              {detail?.neighbours.map((m) => {
                const words = REL_WORDS[m.type] || [m.type.toLowerCase(), m.type.toLowerCase()];
                return (
                  <button key={m.id + m.type} className="row-hover" onClick={() => pick(m.id)}
                          style={{ display: "grid", gridTemplateColumns: "10px 1fr auto", gap: 8, alignItems: "baseline",
                                   padding: "5px 2px", border: 0, borderBottom: `1px solid ${muted(8)}`, background: "none",
                                   font: "inherit", textAlign: "left", cursor: "pointer", color: "inherit" }}>
                    <span style={{ width: 8, height: 8, background: kindOf(m.label).color, alignSelf: "center" }} />
                    <span style={{ minWidth: 0 }}>
                      <span style={{ fontSize: 11, color: muted(50) }}>{m.outgoing ? words[0] : words[1]} </span>
                      <span style={{ fontSize: 13, wordBreak: "break-word" }}>{m.name}</span>
                    </span>
                    <span className="mono" style={{ fontSize: 10.5, color: muted(50) }}>
                      {m.type === "APPEARS_IN" || m.type === "HEARD_IN" ? fmtSpeech(m.weight) : m.weight > 1 ? `${Math.round(m.weight)}×` : ""}
                    </span>
                  </button>
                );
              })}
            </div>

            {detail && detail.lines.length > 0 && (
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <span className="kicker-sm" style={{ color: muted(55) }}>
                  {sel.label === "Clip" ? "Opening lines" : sel.label === "Speaker" ? "Said" : "Where it comes up"}
                </span>
                {detail.lines.map((l) => (
                  <p key={l.speech_id} style={{ margin: 0, fontSize: 12.5, lineHeight: 1.5, color: muted(78),
                                                borderLeft: "2px solid var(--color-divider)", paddingLeft: 9 }}>
                    <span className="mono" style={{ fontSize: 10.5, color: "var(--color-accent-700)" }}>{fmtTime(l.start_s)} </span>
                    {l.text.length > 220 ? `${l.text.slice(0, 218)}…` : l.text}
                  </p>
                ))}
              </div>
            )}

            <button className="btn btn-secondary" style={{ fontSize: 12, alignSelf: "flex-start" }}
                    onClick={() => navigate(`/ask`, { state: { prefill: `What connects ${sel.name} to the rest of the corpus?` } })}>
              Ask about {sel.label === "Clip" ? "this recording" : (sel.name || "").split(" ")[0]}
            </button>
          </aside>
        )}
      </section>
    </div>
  );
}
