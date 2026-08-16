import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/api/client";
import { Dialog } from "@/components/dialog";
import { useSidebarTakeover } from "@/components/layout";
import { AMBER_INK, MOOD_COLOR, RED, muted } from "@/lib/ui";

type Lap = { number: number; duration_s: number | null; prev_s: number | null };
type Citation = {
  speech_id: string; text: string | null; mood: string | null;
  sentiment: string | null; sentiment_score: number | null; text_sentiment: string | null;
  speaker: string | null; driver: string | null; driver_name: string | null;
  team: string | null; session: string | null; year: number | null;
  clip_id: string | null; start_s: number | null; laps: Lap[] | null;
  mentions: { kind: string; name: string }[] | null;
};
type Turn = {
  question: string; answer: string; citations: string[]; sources: string[];
  citation_details: Citation[]; tools_used: string[];
  usage: { input_tokens: number; output_tokens: number };
  model: string; provider: string; refused: boolean; detail: string | null;
};
type Thread = { id: string; title: string; created: number; turns: Turn[] };

const STORE = "ask_threads";

/** Threads live in the browser. The API answers one question at a time and keeps no
 *  conversation of its own, so the history in the sidebar is this machine's, and deleting
 *  a thread deletes the conversation — never the clips it cited. */
function loadThreads(): Thread[] {
  try { return JSON.parse(localStorage.getItem(STORE) || "[]"); } catch { return []; }
}
function saveThreads(t: Thread[]) {
  localStorage.setItem(STORE, JSON.stringify(t.slice(0, 60)));
}

const TOOL_LABEL: Record<string, string> = {
  search_speech: "searched the corpus",
  expand_speech: "pulled surrounding context",
  driver_timeline: "walked the lap timeline",
  compare_speakers: "compared speakers",
};

const EXAMPLES = [
  "Which radio calls sound stressed, and what was happening?",
  "Where do the text and voice readings disagree?",
  "Find anything about tyre wear and who said it",
];

const fmtAt = (s: number | null) =>
  s == null ? "the clip" : `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/** The answer with its cited ids turned into numbered chips that jump to the evidence. */
function AnswerBody({ text, order, onJump }: {
  text: string; order: string[]; onJump: (id: string) => void;
}) {
  const parts = useMemo(() => {
    const re = /\[?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\]?/gi;
    const out: React.ReactNode[] = [];
    let last = 0, key = 0, m: RegExpExecArray | null;
    while ((m = re.exec(text))) {
      if (m.index > last) out.push(<span key={key++}>{text.slice(last, m.index)}</span>);
      const id = m[1].toLowerCase();
      const idx = order.indexOf(id);
      out.push(
        <a key={key++} href="#" title={id}
           onClick={(e) => { e.preventDefault(); onJump(id); }}
           className="mono"
           style={{ fontSize: 11, padding: "1px 5px", border: "1px solid var(--color-accent)" }}>
          {idx >= 0 ? idx + 1 : "·"}
        </a>);
      last = m.index + m[0].length;
    }
    if (last < text.length) out.push(<span key={key++}>{text.slice(last)}</span>);
    return out;
  }, [text, order, onJump]);

  return (
    <p style={{ margin: 0, fontSize: 16, lineHeight: 1.65, textWrap: "pretty" } as React.CSSProperties}>
      {parts}
    </p>
  );
}

function EvidenceCard({ c, n, refFn }: {
  c: Citation; n: number; refFn: (el: HTMLDivElement | null) => void;
}) {
  const navigate = useNavigate();
  const who = c.driver_name || c.driver || c.speaker || "unidentified speaker";
  const lap = c.laps?.[0];
  const delta = lap && lap.duration_s != null && lap.prev_s != null
    ? lap.duration_s - lap.prev_s : null;
  const disagrees = c.text_sentiment && c.sentiment && c.text_sentiment !== c.sentiment;

  return (
    <article ref={refFn} className="blueprint" style={{
      padding: "15px 16px", display: "grid", gridTemplateColumns: "28px 1fr",
      gap: 14, scrollMarginTop: 24,
    }}>
      <span className="mono" style={{
        fontSize: 12, color: "var(--color-accent)", border: "1px solid var(--color-accent)",
        height: 24, display: "grid", placeItems: "center",
      }}>
        {n > 0 ? n : "·"}
      </span>
      <div style={{ display: "flex", flexDirection: "column", gap: 9, minWidth: 0 }}>
        <div className="mono" style={{
          display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap",
          fontSize: 11.5, color: muted(60),
        }}>
          <span style={{ fontFamily: "var(--font-heading)", fontSize: 15, color: "var(--color-text)" }}>
            {who}
          </span>
          {c.team && <span>{c.team}</span>}
          {c.session && <span>{c.session}{c.year ? ` ${c.year}` : ""}</span>}
          {lap && (
            <span>
              lap {lap.number}
              {lap.duration_s != null && ` · ${lap.duration_s.toFixed(1)}s`}
              {delta != null && ` (${delta >= 0 ? "+" : ""}${delta.toFixed(1)})`}
            </span>
          )}
        </div>
        {c.text && (
          <p style={{
            margin: 0, fontSize: 15, lineHeight: 1.6,
            borderLeft: "2px solid var(--color-divider)", paddingLeft: 12,
          }}>
            {c.text}
          </p>
        )}
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          {c.mood && (
            <span className="tag mono" style={{
              border: `1px solid ${MOOD_COLOR[c.mood] || "var(--color-neutral-300)"}`,
              color: MOOD_COLOR[c.mood] || undefined,
            }}>
              voice · {c.mood}
            </span>
          )}
          {c.sentiment && (
            <span className="tag tag-neutral mono" style={{ border: "1px solid var(--color-neutral-300)" }}>
              text · {c.sentiment}
              {c.sentiment_score != null && ` ${c.sentiment_score > 0 ? "+" : ""}${c.sentiment_score.toFixed(2)}`}
            </span>
          )}
          {disagrees && (
            <span style={{ fontSize: 11.5, color: AMBER_INK }}>
              text and voice disagree here
            </span>
          )}
          {(c.mentions ?? []).filter((m) => m.name).map((m) => (
            <span key={`${m.kind}-${m.name}`} className="tag tag-neutral" style={{ border: "1px solid var(--color-neutral-300)" }}>
              mentions {m.name}
            </span>
          ))}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          {c.clip_id ? (
            <a href={`/clips/${c.clip_id}`} style={{ fontSize: 13 }}
               onClick={(e) => {
                 e.preventDefault();
                 navigate(`/clips/${c.clip_id}${c.start_s ? `?t=${c.start_s}` : ""}`);
               }}>
              ▶ Listen at {fmtAt(c.start_s)}
            </a>
          ) : (
            <span style={{ fontSize: 12.5, color: muted(55) }}>
              Team radio · no processed clip to open
            </span>
          )}
          <span className="mono" style={{ fontSize: 11.5, color: muted(45) }}>
            {c.speech_id.slice(0, 8)}
          </span>
        </div>
      </div>
    </article>
  );
}

export default function Ask() {
  const navigate = useNavigate();
  const [threads, setThreads] = useState<Thread[]>(loadThreads);
  const [activeId, setActiveId] = useState<string | null>(threads[0]?.id ?? null);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [activity, setActivity] = useState<string[]>([]);
  const [disabled, setDisabled] = useState<string | null>(null);
  const [renaming, setRenaming] = useState(false);
  const [renameDraft, setRenameDraft] = useState("");
  const [deleting, setDeleting] = useState<Thread | null>(null);
  const [corpus, setCorpus] = useState<{ clips: number; utterances: number } | null>(null);
  const cardRefs = useRef(new Map<string, HTMLDivElement>());
  const bottomRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const active = threads.find((t) => t.id === activeId) || null;

  useEffect(() => () => wsRef.current?.close(), []);
  useEffect(() => { saveThreads(threads); }, [threads]);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [active?.turns, activity]);
  useEffect(() => {
    api.stats().then((s) => setCorpus({ clips: s.total_clips, utterances: 0 })).catch(() => {});
  }, []);

  function newThread() { setActiveId(null); setQuestion(""); }

  function removeThread(t: Thread) {
    setThreads((prev) => prev.filter((x) => x.id !== t.id));
    if (activeId === t.id) setActiveId(null);
    setDeleting(null);
  }

  function exportThread() {
    if (!active) return;
    const body = active.turns.map((t) =>
      `## ${t.question}\n\n${t.answer}\n\n${t.citation_details.map((c, i) =>
        `[${i + 1}] ${c.speaker || c.driver_name || "unknown"} — “${c.text || ""}”`).join("\n")}`
    ).join("\n\n---\n\n");
    const url = URL.createObjectURL(new Blob([`# ${active.title}\n\n${body}\n`], { type: "text/markdown" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `${active.title.replace(/[^\w-]+/g, "_")}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function jumpTo(id: string) {
    cardRefs.current.get(id)?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  async function ask(q: string) {
    if (!q.trim() || busy) return;
    setBusy(true);
    setActivity([]);
    setDisabled(null);

    // The socket opens before the request so no early tool event is missed — redis pub/sub
    // has no replay. It carries progress only; the answer comes back in the response body.
    const conversationId = crypto.randomUUID();
    const ws = new WebSocket(`${location.origin.replace("http", "ws")}/v1/ws/jobs/${conversationId}`);
    wsRef.current = ws;
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type !== "agent" || msg.kind !== "tool_use") return;
      setActivity((prev) => [...prev, TOOL_LABEL[msg.tool] || msg.tool]);
    };

    try {
      const history = active?.turns.map((t) => ({ question: t.question, answer: t.answer })) ?? null;
      const res = await api.agentAsk(q, history, conversationId);
      const turn: Turn = { question: q, ...res };
      setThreads((prev) => {
        if (active) {
          return prev.map((t) => t.id === active.id ? { ...t, turns: [...t.turns, turn] } : t);
        }
        const t: Thread = {
          id: conversationId,
          title: q.length > 48 ? `${q.slice(0, 48)}…` : q,
          created: Date.now(), turns: [turn],
        };
        setActiveId(t.id);
        return [t, ...prev];
      });
      setQuestion("");
    } catch (err) {
      const detail = String(err);
      if (detail.includes("disabled") || detail.includes("not set")) setDisabled(detail);
      else toast.error("Could not answer", { description: detail });
    } finally {
      ws.close();
      wsRef.current = null;
      setBusy(false);
      setActivity([]);
    }
  }

  // ── sidebar: thread history replaces the nav while Ask is open ──────────
  const groups = useMemo(() => {
    const day = 86_400_000;
    const now = Date.now();
    const bucket = (t: Thread) =>
      now - t.created < day ? "Today" : now - t.created < 2 * day ? "Yesterday" : "Earlier";
    const out: { label: string; items: Thread[] }[] = [];
    for (const label of ["Today", "Yesterday", "Earlier"]) {
      const items = threads.filter((t) => bucket(t) === label);
      if (items.length) out.push({ label, items });
    }
    return out;
  }, [threads]);

  useSidebarTakeover(
    <div style={{ display: "flex", flexDirection: "column", minHeight: 0, flex: 1, padding: "0 8px" }}>
      <button className="btn btn-primary" style={{ margin: "0 4px 12px", fontSize: 13 }} onClick={newThread}>
        New chat
      </button>
      <button
        onClick={() => navigate("/")}
        style={{
          display: "flex", alignItems: "center", gap: 8, background: "transparent", border: "none",
          font: "inherit", fontSize: 12.5, color: muted(60), cursor: "pointer",
          padding: "4px 14px 12px", textAlign: "left",
        }}
      >
        ← Back to workspace
      </button>
      <div style={{ flex: 1, minHeight: 0, overflow: "auto", display: "flex", flexDirection: "column", gap: 2 }}>
        {groups.map((g) => (
          <div key={g.label} style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            <span className="kicker" style={{ letterSpacing: ".14em", padding: "12px 6px 4px", color: muted(42) }}>
              {g.label}
            </span>
            {g.items.map((t) => (
              <div
                key={t.id}
                className="row-hover"
                onClick={() => setActiveId(t.id)}
                style={{
                  display: "flex", alignItems: "flex-start", gap: 8, padding: "7px 8px", cursor: "pointer",
                  background: t.id === activeId ? "color-mix(in srgb, var(--color-accent) 12%, transparent)" : "transparent",
                  borderLeft: `2px solid ${t.id === activeId ? "var(--color-accent)" : "transparent"}`,
                }}
              >
                <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 2 }}>
                  <span style={{
                    fontSize: 13, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                    color: t.id === activeId ? "var(--color-text)" : muted(78),
                  }}>
                    {t.title}
                  </span>
                  <span className="mono" style={{ fontSize: 10.5, color: muted(42) }}>
                    {t.turns.length} exchange{t.turns.length === 1 ? "" : "s"} ·{" "}
                    {new Date(t.created).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}
                  </span>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); setDeleting(t); }}
                  aria-label="Delete thread"
                  style={{
                    background: "transparent", border: "none", font: "inherit", fontSize: 13,
                    lineHeight: 1, color: muted(35), cursor: "pointer", padding: "2px 0",
                  }}
                >×</button>
              </div>
            ))}
          </div>
        ))}
        {threads.length === 0 && (
          <span style={{ padding: "12px 8px", fontSize: 12, color: muted(45) }}>No threads yet.</span>
        )}
      </div>
    </div>,
    [threads, activeId],
  );

  const lastTurn = active?.turns[active.turns.length - 1];
  const empty = !active && !busy;

  const composer = (
    <>
      <div className="blueprint" style={{ padding: "10px 12px", display: "flex", alignItems: "flex-end", gap: 10 }}>
        <textarea
          className="input"
          rows={2}
          value={question}
          disabled={busy}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(question); }
          }}
          placeholder={empty
            ? "Ask anything about the recordings. Answers cite the clips they came from."
            : "Ask a follow-up. Answers cite clips; nothing is asserted without a citation."}
          style={{
            flex: 1, resize: "none", border: "none", background: "transparent",
            padding: "2px 0", fontSize: 15, lineHeight: 1.5, minHeight: 0,
          }}
        />
        <button className="btn btn-primary" style={{ flex: "none" }}
                disabled={busy || !question.trim()} onClick={() => ask(question)}>
          {busy ? "Asking…" : "Ask"}
        </button>
      </div>
      <span className="mono" style={{ fontSize: 11, color: muted(45) }}>
        retrieval local · answer generation off-box
      </span>
    </>
  );

  return (
    // Ask is a single 940 column — header rule, transcript and composer share one width
    // and one centre line, so the input never drifts off to the side of a wide window.
    <div style={{
      height: "calc(100vh - 86px)", display: "flex", flexDirection: "column",
      alignItems: "center", margin: "0 0 -60px",
    }}>
      {/* header, thread and composer all sit in the same 940 column, so the page reads as
          one column of rules rather than three of different widths on a wide screen */}
      <header style={{
        flex: "none", display: "flex", alignItems: "center", justifyContent: "space-between",
        gap: 20, padding: "0 0 12px", width: "100%", maxWidth: 940,
        borderBottom: "1px solid var(--color-divider)",
      }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, minWidth: 0 }}>
          <h3 style={{
            margin: 0, fontSize: 19, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
          }}>
            {active?.title || "New question"}
          </h3>
          <span className="mono" style={{ fontSize: 11.5, color: muted(50), whiteSpace: "nowrap" }}>
            {corpus ? `${corpus.clips} clips` : "corpus"}
            {lastTurn ? ` · ${lastTurn.provider}/${lastTurn.model}` : ""}
          </span>
        </div>
        {/* Nothing to rename, export or delete until a thread exists — three greyed-out
            buttons are just furniture on an empty screen. */}
        {active && (
          <div style={{ display: "flex", gap: 6, flex: "none" }}>
            <button className="btn btn-ghost" style={{ fontSize: 12, padding: "5px 10px" }}
                    onClick={() => { setRenameDraft(active.title); setRenaming(true); }}>
              Rename
            </button>
            <button className="btn btn-ghost" style={{ fontSize: 12, padding: "5px 10px" }}
                    onClick={exportThread}>
              Export thread
            </button>
            <button className="btn btn-ghost" style={{ fontSize: 12, padding: "5px 10px", color: RED }}
                    onClick={() => setDeleting(active)}>
              Delete
            </button>
          </div>
        )}
      </header>

      <div style={{
        flex: 1, minHeight: 0, overflow: "auto", padding: "18px 0 24px",
        display: "flex", flexDirection: "column", gap: 20, width: "100%", maxWidth: 940,
      }}>
        {disabled && (
          <div style={{ border: `1px solid ${AMBER_INK}`, padding: "12px 14px", display: "flex", flexDirection: "column", gap: 6 }}>
            <span className="kicker-sm" style={{ color: AMBER_INK }}>The agent is turned off</span>
            <span style={{ fontSize: 13, color: muted(70) }}>{disabled}</span>
            <span style={{ fontSize: 13, color: muted(70) }}>
              This is the one feature that sends transcript text off the box, so it is disabled
              by default and set in <span className="mono">.env</span>, not from Settings. Set{" "}
              <span className="mono">LLM_ENABLED=true</span> with a provider key, then restart the API.
            </span>
          </div>
        )}

        {/* Nothing to scroll on an empty thread, so the invitation and the composer sit
            together in the middle of the space rather than at opposite ends of it. */}
        {empty && (
          <div style={{
            flex: 1, display: "flex", flexDirection: "column", justifyContent: "center", gap: 14,
          }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <span className="kicker">Ask the corpus</span>
              <p style={{ margin: 0, fontSize: 16, lineHeight: 1.6, color: muted(72), maxWidth: "58ch" }}>
                Retrieval runs on the box; only the final wording is composed off-box, and every
                claim comes back with the clip it came from.
              </p>
            </div>
            {composer}
            <div style={{ display: "flex", flexDirection: "column", gap: 8, paddingTop: 4 }}>
              {EXAMPLES.map((ex) => (
                <button key={ex} className="btn btn-secondary"
                        style={{ justifyContent: "space-between", fontSize: 13, textAlign: "left" }}
                        onClick={() => ask(ex)}>
                  {ex} <span style={{ opacity: 0.5 }}>↵</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {(active?.turns || []).map((turn, i) => {
          const cited = turn.citation_details.filter((c) => turn.citations.includes(c.speech_id));
          const consulted = turn.citation_details.filter((c) => !turn.citations.includes(c.speech_id));
          return (
            <div key={i} style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              <div style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
                <span className="mono" style={{ fontSize: 11, color: muted(45), width: 44, flex: "none", paddingTop: 4 }}>
                  you
                </span>
                <p style={{
                  margin: 0, fontSize: 16, lineHeight: 1.55,
                  borderLeft: "2px solid var(--color-divider)", paddingLeft: 14,
                }}>
                  {turn.question}
                </p>
              </div>

              {turn.tools_used?.length > 0 && (
                <div className="mono" style={{
                  display: "flex", gap: 14, alignItems: "center", fontSize: 11.5,
                  color: muted(55), flexWrap: "wrap",
                }}>
                  {turn.tools_used.map((t, j) => (
                    <span key={j}>✓ {TOOL_LABEL[t] || t}</span>
                  ))}
                </div>
              )}

              <div style={{
                borderLeft: "2px solid var(--color-accent)", padding: "2px 0 2px 18px",
                display: "flex", flexDirection: "column", gap: 10,
              }}>
                {turn.refused ? (
                  <p style={{ margin: 0, fontSize: 15, color: muted(70) }}>
                    The model declined this one.{turn.detail ? ` ${turn.detail}` : ""}
                  </p>
                ) : turn.answer ? (
                  <AnswerBody text={turn.answer} order={turn.citations} onJump={jumpTo} />
                ) : (
                  <p style={{ margin: 0, fontSize: 15, color: muted(70) }}>
                    No answer came back.{turn.detail ? ` ${turn.detail}` : ""}
                  </p>
                )}
              </div>

              {cited.length > 0 && (
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  <span className="kicker-sm" style={{ color: muted(55) }}>
                    Evidence · {cited.length} cited
                  </span>
                  {cited.map((c) => (
                    <EvidenceCard
                      key={c.speech_id} c={c} n={turn.citations.indexOf(c.speech_id) + 1}
                      refFn={(el) => { el ? cardRefs.current.set(c.speech_id, el) : cardRefs.current.delete(c.speech_id); }}
                    />
                  ))}
                </div>
              )}

              {consulted.length > 0 && (
                <div className="blueprint hatch" style={{
                  padding: "13px 16px", display: "flex", flexDirection: "column", gap: 6,
                }}>
                  <span className="kicker-sm" style={{ color: AMBER_INK }}>
                    Sources consulted — the model did not cite these directly
                  </span>
                  <span style={{ fontSize: 13, color: muted(70) }}>
                    {consulted.length} further passage{consulted.length === 1 ? " was" : "s were"} retrieved
                    and read but not referenced in the answer. They carry no citation authority.
                  </span>
                </div>
              )}

              <div className="mono" style={{
                display: "flex", gap: 16, paddingTop: 12, borderTop: "1px solid var(--color-divider)",
                fontSize: 11, color: muted(50), flexWrap: "wrap",
              }}>
                {turn.tools_used?.length > 0 && <span>tools: {turn.tools_used.join(", ")}</span>}
                <span>model: {turn.provider}/{turn.model}</span>
                {turn.usage && (
                  <span>
                    {turn.usage.input_tokens.toLocaleString()} in / {turn.usage.output_tokens.toLocaleString()} out
                  </span>
                )}
              </div>
            </div>
          );
        })}

        {busy && (
          <div className="mono" style={{ display: "flex", flexDirection: "column", gap: 6, fontSize: 11.5, color: muted(55) }}>
            <span className="pulse">· thinking</span>
            {activity.map((a, i) => <span key={i}>✓ {a}</span>)}
          </div>
        )}

        {active && (
          <p style={{ margin: 0, fontSize: 11, color: muted(50) }}>
            Tone labels are heuristic readings of the audio, not measurements of how anyone felt.
          </p>
        )}
        <div ref={bottomRef} />
      </div>

      {/* With a thread open the composer is pinned under the scrolling transcript. On an
          empty screen there is nothing to scroll, so it travels up with the invitation
          instead of sitting alone at the foot of a blank page. */}
      {!empty && (
        <div style={{
          flex: "none", padding: "14px 0 18px", borderTop: "1px solid var(--color-divider)",
          background: "var(--color-bg)", display: "flex", flexDirection: "column",
          gap: 9, width: "100%", maxWidth: 940,
        }}>
          {composer}
        </div>
      )}

      <Dialog
        open={renaming}
        onClose={() => setRenaming(false)}
        kicker="Ask"
        title="Rename thread"
        subject={active ? `created ${new Date(active.created).toLocaleString()}` : ""}
        confirm="Rename"
        width="440px"
        onConfirm={() => {
          setThreads((prev) => prev.map((t) => t.id === activeId ? { ...t, title: renameDraft } : t));
          setRenaming(false);
        }}
      >
        <div className="field">
          <label>Thread name</label>
          <input className="input" value={renameDraft} onChange={(e) => setRenameDraft(e.target.value)} />
        </div>
      </Dialog>

      <Dialog
        open={deleting != null}
        onClose={() => setDeleting(null)}
        kicker="Ask · destructive"
        title="Delete this thread?"
        subject={deleting ? `${deleting.title} · ${deleting.turns.length} exchanges` : ""}
        consequence="Not reversible."
        confirm="Delete thread"
        cancel="Keep it"
        danger
        width="470px"
        onConfirm={() => deleting && removeThread(deleting)}
      >
        <p style={{ margin: 0, fontSize: 14.5, lineHeight: 1.6 }}>
          The thread and its {deleting?.turns.length} exchange{deleting?.turns.length === 1 ? "" : "s"} go.
          The clips it cited are untouched — this deletes the conversation, not the evidence.
        </p>
      </Dialog>
    </div>
  );
}
