import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/api/client";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { MOOD_STYLE, type Mood } from "@/lib/mood";
import { cn } from "@/lib/utils";
import {
  Sparkles, Search, Radar, Route, Users, Send, Loader2, PlayCircle,
  AlertTriangle, Quote, CornerDownLeft,
} from "lucide-react";

type Lap = { number: number; duration_s: number | null; prev_s: number | null };
type Citation = {
  speech_id: string;
  text: string | null;
  mood: Mood | null;
  sentiment: string | null;
  sentiment_score: number | null;
  text_sentiment: string | null;
  speaker: string | null;
  driver: string | null;
  driver_name: string | null;
  team: string | null;
  session: string | null;
  year: number | null;
  clip_id: string | null;
  start_s: number | null;
  laps: Lap[] | null;
  mentions: { kind: string; name: string }[] | null;
};
type Turn = {
  question: string;
  answer: string;
  citations: string[];
  sources: string[];
  citation_details: Citation[];
  tools_used: string[];
  usage: { input_tokens: number; output_tokens: number };
  model: string;
  provider: string;
  refused: boolean;
  detail: string | null;
};

const TOOL_META: Record<string, { label: string; icon: typeof Search }> = {
  search_speech: { label: "Searching the corpus", icon: Search },
  expand_speech: { label: "Pulling the surrounding context", icon: Radar },
  driver_timeline: { label: "Walking the lap timeline", icon: Route },
  compare_speakers: { label: "Comparing speakers", icon: Users },
};

const EXAMPLES = [
  "What did Albon say about power, and which lap was it on?",
  "Which radio calls sound stressed, and what was happening?",
  "Find anything about tyre wear and who said it",
];

/** A cited id, rendered inline as a chip that scrolls to its evidence card. */
function CiteChip({ id, index, onJump }: { id: string; index: number; onJump: (id: string) => void }) {
  return (
    <button
      onClick={() => onJump(id)}
      title={id}
      className="mx-0.5 inline-flex translate-y-[-1px] items-center rounded-md border border-primary/30 bg-primary/10 px-1.5 py-0 text-[11px] font-medium text-primary transition-colors hover:bg-primary/20"
    >
      {index}
    </button>
  );
}

/** Split the answer on cited ids so each becomes a chip, leaving the prose intact. */
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
      out.push(<CiteChip key={key++} id={id} index={idx >= 0 ? idx + 1 : 0} onJump={onJump} />);
      last = m.index + m[0].length;
    }
    if (last < text.length) out.push(<span key={key++}>{text.slice(last)}</span>);
    return out;
  }, [text, order, onJump]);

  return <p className="whitespace-pre-wrap text-[15px] leading-relaxed text-foreground">{parts}</p>;
}

function EvidenceCard({ c, index, refFn }: {
  c: Citation; index: number; refFn: (el: HTMLDivElement | null) => void;
}) {
  const who = c.driver_name || c.driver || c.speaker || "unidentified speaker";
  const lap = c.laps?.[0];
  const delta = lap && lap.duration_s != null && lap.prev_s != null ? lap.duration_s - lap.prev_s : null;
  // Only the fused read differing from the text read is worth calling out — that
  // disagreement is signal the pipeline stores separately on purpose.
  const disagrees = c.text_sentiment && c.sentiment && c.text_sentiment !== c.sentiment;

  return (
    <div ref={refFn} className="scroll-mt-24">
      <Card className="border-border/70 transition-colors hover:border-primary/40">
        <CardContent className="space-y-2.5 p-4">
          <div className="flex flex-wrap items-center gap-2">
            <span className="flex size-5 shrink-0 items-center justify-center rounded-md bg-primary/15 text-[11px] font-semibold text-primary">
              {index}
            </span>
            <span className="text-sm font-medium">{who}</span>
            {c.team && <span className="text-xs text-muted-foreground">{c.team}</span>}
            {c.session && (
              <Badge variant="outline" className="text-[11px]">
                {c.session}{c.year ? ` ${c.year}` : ""}
              </Badge>
            )}
            {lap && (
              <Badge variant="outline" className="font-mono text-[11px]">
                lap {lap.number}
                {lap.duration_s != null && ` · ${lap.duration_s.toFixed(1)}s`}
                {delta != null && ` (${delta >= 0 ? "+" : ""}${delta.toFixed(1)})`}
              </Badge>
            )}
          </div>

          {c.text && (
            <div className="flex gap-2">
              <Quote className="mt-1 size-3.5 shrink-0 text-muted-foreground/60" />
              <p className="text-sm leading-relaxed text-foreground/90">{c.text}</p>
            </div>
          )}

          <div className="flex flex-wrap items-center gap-1.5">
            {/* Phrased as readings, never as facts: tone is a threshold heuristic over
                acoustic features, not a trained emotion model. */}
            {c.mood && (
              <Badge variant="outline" className={cn("text-[11px]", MOOD_STYLE[c.mood])}>
                voice reads {c.mood}
              </Badge>
            )}
            {c.sentiment && (
              <Badge variant="outline" className="text-[11px]">
                words read {c.sentiment}
                {c.sentiment_score != null && ` (${c.sentiment_score > 0 ? "+" : ""}${c.sentiment_score.toFixed(2)})`}
              </Badge>
            )}
            {disagrees && (
              <Badge variant="outline" className="border-warning/30 bg-warning/10 text-[11px] text-warning">
                text alone reads {c.text_sentiment}
              </Badge>
            )}
            {(c.mentions ?? []).filter((m) => m.name).map((m) => (
              <Badge key={`${m.kind}-${m.name}`} variant="secondary" className="text-[11px]">
                mentions {m.name}
              </Badge>
            ))}
          </div>

          {c.clip_id ? (
            <Link
              to={`/clips/${c.clip_id}${c.start_s ? `?t=${c.start_s}` : ""}`}
              className="inline-flex items-center gap-1.5 text-xs font-medium text-primary hover:underline"
            >
              <PlayCircle className="size-3.5" />
              Listen at {c.start_s != null ? `${Math.floor(c.start_s / 60)}:${String(Math.floor(c.start_s % 60)).padStart(2, "0")}` : "the clip"}
            </Link>
          ) : (
            // Radio calls have no clip row unless they are put through the full pipeline.
            <span className="text-xs text-muted-foreground">Team radio · no processed clip to open</span>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

export default function Ask() {
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [activity, setActivity] = useState<string[]>([]);
  const [disabled, setDisabled] = useState<string | null>(null);
  const cardRefs = useRef(new Map<string, HTMLDivElement>());
  const wsRef = useRef<WebSocket | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => () => wsRef.current?.close(), []);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: "smooth" }); }, [turns, activity]);

  function jumpTo(id: string) {
    cardRefs.current.get(id)?.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  async function ask(q: string) {
    if (!q.trim() || busy) return;
    setBusy(true);
    setActivity([]);
    setDisabled(null);

    // The websocket is opened *before* the request so no early tool event is missed —
    // redis pub/sub has no replay. It carries progress only; the answer comes back in the
    // response body, so a dropped socket costs the live view, never the result.
    const conversationId = crypto.randomUUID();
    const ws = new WebSocket(`${location.origin.replace("http", "ws")}/v1/ws/jobs/${conversationId}`);
    wsRef.current = ws;
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.type !== "agent") return;
      if (msg.kind === "tool_use") {
        const meta = TOOL_META[msg.tool];
        setActivity((prev) => [...prev, meta ? meta.label : msg.tool]);
      }
    };

    try {
      const res = await api.agentAsk(q, null, conversationId);
      setTurns((prev) => [...prev, { question: q, ...res }]);
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

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        title="Ask"
        description="Question the corpus. Every claim is cited back to the recording it came from."
      />

      <div className="flex-1 space-y-6 overflow-y-auto px-8 py-6">
        {disabled && (
          <Alert className="border-warning/30 bg-warning/10">
            <AlertTriangle className="size-4 text-warning" />
            <AlertTitle>The agent is turned off</AlertTitle>
            <AlertDescription className="space-y-1.5 text-sm">
              <p className="text-muted-foreground">{disabled}</p>
              <p className="text-muted-foreground">
                This is the one feature that sends transcript text off the box, so it is
                disabled by default and set in <code className="text-foreground">.env</code>,
                not from the Settings page. Set <code className="text-foreground">LLM_ENABLED=true</code>
                {" "}with a provider key, then restart the API.
              </p>
            </AlertDescription>
          </Alert>
        )}

        {turns.length === 0 && !busy && (
          <div className="mx-auto max-w-2xl space-y-6 pt-10 text-center">
            <div className="mx-auto flex size-12 items-center justify-center rounded-2xl bg-gradient-to-br from-primary to-chart-5 shadow-lg shadow-primary/20">
              <Sparkles className="size-6 text-primary-foreground" strokeWidth={2} />
            </div>
            <div className="space-y-1.5">
              <h2 className="text-lg font-semibold">Ask about anything in the corpus</h2>
              <p className="text-sm text-muted-foreground">
                Searches transcripts and team radio, follows the graph to laps, speakers and
                mentions, and cites what it used.
              </p>
            </div>
            <div className="space-y-2">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  onClick={() => ask(ex)}
                  className="flex w-full items-center justify-between gap-3 rounded-lg border border-border bg-card px-4 py-2.5 text-left text-sm text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground"
                >
                  {ex}
                  <CornerDownLeft className="size-3.5 shrink-0 opacity-50" />
                </button>
              ))}
            </div>
          </div>
        )}

        {turns.map((turn, i) => (
          <div key={i} className="mx-auto max-w-3xl space-y-4">
            <div className="flex justify-end">
              <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-primary/10 px-4 py-2.5 text-sm text-foreground">
                {turn.question}
              </div>
            </div>

            <Card className="border-border/70">
              <CardContent className="space-y-4 p-5">
                {turn.refused ? (
                  <p className="text-sm text-muted-foreground">
                    The model declined this one.{turn.detail ? ` ${turn.detail}` : ""}
                  </p>
                ) : turn.answer ? (
                  <AnswerBody text={turn.answer} order={turn.citations} onJump={jumpTo} />
                ) : (
                  <p className="text-sm text-muted-foreground">
                    No answer came back.{turn.detail ? ` ${turn.detail}` : ""}
                  </p>
                )}

                {turn.citation_details?.length > 0 && (
                  <div className="space-y-2 border-t border-border pt-4">
                    {/* "Evidence" only when the model actually cited. When it answered
                        without ids, these are what its tools showed it — related, but not
                        a claim-level citation, and labelled so nobody reads it as one. */}
                    <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                      {turn.citations?.length ? "Evidence" : "Sources consulted (the model did not cite directly)"}
                    </p>
                    {turn.citation_details.map((c) => (
                      <EvidenceCard
                        key={c.speech_id}
                        c={c}
                        index={turn.citations.indexOf(c.speech_id) + 1}
                        refFn={(el) => {
                          if (el) cardRefs.current.set(c.speech_id, el);
                          else cardRefs.current.delete(c.speech_id);
                        }}
                      />
                    ))}
                  </div>
                )}

                {/* Provenance strip: which tools ran, on which model, at what cost. The
                    agent is the one component that leaves the box, so what it did is not
                    hidden behind a spinner. */}
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-border pt-3 text-[11px] text-muted-foreground">
                  {turn.tools_used?.length > 0 && (
                    <span>{turn.tools_used.length} tool call{turn.tools_used.length > 1 ? "s" : ""}: {turn.tools_used.join(" → ")}</span>
                  )}
                  <span className="font-mono">{turn.provider}/{turn.model}</span>
                  {turn.usage && (
                    <span className="font-mono">
                      {turn.usage.input_tokens.toLocaleString()} in / {turn.usage.output_tokens.toLocaleString()} out
                    </span>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        ))}

        {busy && (
          <div className="mx-auto max-w-3xl">
            <Card className="border-border/70">
              <CardContent className="space-y-2 p-5">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="size-4 animate-spin text-primary" />
                  Thinking
                </div>
                {activity.map((a, i) => {
                  const meta = Object.values(TOOL_META).find((m) => m.label === a);
                  const Icon = meta?.icon ?? Search;
                  return (
                    <div key={i} className="flex items-center gap-2 pl-6 text-xs text-muted-foreground animate-slide-up">
                      <Icon className="size-3.5 text-primary/70" />
                      {a}
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-border bg-background px-8 py-4">
        <div className="mx-auto flex max-w-3xl items-end gap-2">
          <Textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(question); }
            }}
            placeholder="Ask about the recordings…  (Enter to send, Shift+Enter for a new line)"
            rows={1}
            disabled={busy}
            className="max-h-40 min-h-[42px] resize-none"
          />
          <Button onClick={() => ask(question)} disabled={busy || !question.trim()} size="icon" className="size-[42px] shrink-0">
            {busy ? <Loader2 className="size-4 animate-spin" /> : <Send className="size-4" />}
          </Button>
        </div>
        <p className="mx-auto mt-2 max-w-3xl text-[11px] text-muted-foreground">
          Tone labels are heuristic readings of the audio, not measurements of how anyone felt.
        </p>
      </div>
    </div>
  );
}
