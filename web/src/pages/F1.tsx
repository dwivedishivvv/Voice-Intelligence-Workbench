import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { api } from "@/api/client";
import { PageHeader } from "@/components/page-header";
import { AudioPlayer } from "@/components/audio-player";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select, SelectTrigger, SelectValue, SelectContent, SelectItem,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { type Mood, MOOD_STYLE, moodDot } from "@/lib/mood";
import { Loader2, Radio, Flag, Sparkles, TrendingUp } from "lucide-react";

type Session = { session_key: number; session_name: string; country_name: string; circuit_short_name: string; date_start: string; year: number };
type Driver = { driver_number: number; broadcast_name: string; team_name: string; team_colour: string };
type Lap = { lap_number: number; lap_duration: number | null; date_start: string };
type RadioClip = { driver_number: number; date: string; recording_url: string };
type Analysis = { status: "pending" | "done" | "error"; text?: string; mood?: Mood; features?: Record<string, number> };

// SVG can't use the tailwind mood classes, so mirror them onto the same CSS variables
const MOOD_VAR: Record<Mood, string> = {
  calm: "var(--success)",
  stressed: "var(--destructive)",
  tired: "var(--warning)",
};

function fmtLap(s: number | null) {
  if (!s) return "—";
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(3);
  return m > 0 ? `${m}:${sec.padStart(6, "0")}` : `${sec}s`;
}

function fmtDelta(d: number) {
  return `${d >= 0 ? "+" : ""}${d.toFixed(3)}s`;
}

// nearest lap to a radio call's timestamp, so a stressed/tired call can be pinned to the lap it happened on
function nearestLap(laps: Lap[], radioDate: string): Lap | null {
  const t = new Date(radioDate).getTime();
  let best: Lap | null = null, bestDiff = Infinity;
  for (const l of laps) {
    const diff = Math.abs(new Date(l.date_start).getTime() - t);
    if (diff < bestDiff) { bestDiff = diff; best = l; }
  }
  return best;
}

function quantile(sorted: number[], q: number) {
  const i = (sorted.length - 1) * q;
  const lo = Math.floor(i), hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

type Pin = { lap: number; mood: Mood | null; date: string; pending: boolean };

const W = 1000, H = 240, PAD_L = 52, PAD_R = 12, PAD_T = 26, PAD_B = 26;

function LapTrace({
  laps, pins, teamColor, selected, onSelect,
}: {
  laps: Lap[]; pins: Pin[]; teamColor: string;
  selected: string | null; onSelect: (date: string | null) => void;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const valid = laps.filter((l) => l.lap_duration != null);
  if (valid.length < 2) return <p className="text-sm text-muted-foreground">Not enough lap data to chart.</p>;

  const times = valid.map((l) => l.lap_duration!);
  const sorted = [...times].sort((a, b) => a - b);
  const best = sorted[0];
  // pit stops and safety cars run 20-40s off the pace; letting them set the domain squashes
  // every green-flag lap into a flat line, so clamp the top and pin outliers to the ceiling
  const top = Math.min(sorted[sorted.length - 1], quantile(sorted, 0.9) + (quantile(sorted, 0.9) - best) * 0.5 + 0.5);
  const lo = best - (top - best) * 0.08;

  const x = (i: number) => PAD_L + (i / (valid.length - 1)) * (W - PAD_L - PAD_R);
  const y = (t: number) => PAD_T + (1 - (Math.min(t, top) - lo) / Math.max(top - lo, 0.001)) * (H - PAD_T - PAD_B);
  const xByLap = new Map(valid.map((l, i) => [l.lap_number, x(i)]));

  const line = valid.map((l, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(l.lap_duration!).toFixed(1)}`).join(" ");
  const area = `${line} L${x(valid.length - 1).toFixed(1)},${H - PAD_B} L${PAD_L},${H - PAD_B} Z`;
  const ticks = [best, (best + top) / 2, top];
  const hoveredLap = hover != null ? valid[hover] : null;

  return (
    <div className="relative rounded-xl border border-border bg-card p-1">
      <svg
        viewBox={`0 0 ${W} ${H}`} className="w-full"
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          const px = ((e.clientX - r.left) / r.width) * W;
          const i = Math.round(((px - PAD_L) / (W - PAD_L - PAD_R)) * (valid.length - 1));
          setHover(Math.max(0, Math.min(valid.length - 1, i)));
        }}
      >
        <defs>
          <linearGradient id="lapfill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={teamColor} stopOpacity="0.28" />
            <stop offset="100%" stopColor={teamColor} stopOpacity="0" />
          </linearGradient>
        </defs>

        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={PAD_L} x2={W - PAD_R} y1={y(t)} y2={y(t)} stroke="currentColor" strokeOpacity="0.08" />
            <text x={PAD_L - 8} y={y(t) + 4} textAnchor="end" className="fill-muted-foreground" fontSize="11">
              {i === 0 ? fmtLap(t) : fmtDelta(t - best)}
            </text>
          </g>
        ))}

        <path d={area} fill="url(#lapfill)" />
        <path d={line} fill="none" stroke={teamColor} strokeWidth="2" strokeLinejoin="round" />

        {/* fastest lap */}
        <circle cx={xByLap.get(valid[times.indexOf(best)]?.lap_number ?? 0) ?? 0} cy={y(best)} r="4"
          fill="var(--background)" stroke={teamColor} strokeWidth="2" />

        {/* radio calls, pinned to the lap they happened on */}
        {pins.map((p) => {
          const px = xByLap.get(p.lap);
          if (px == null) return null;
          const c = p.mood ? MOOD_VAR[p.mood] : "var(--muted-foreground)";
          const on = selected === p.date;
          return (
            <g key={p.date} onClick={() => onSelect(on ? null : p.date)} className="cursor-pointer">
              <line x1={px} x2={px} y1={PAD_T - 12} y2={H - PAD_B} stroke={c}
                strokeOpacity={on ? 0.7 : 0.25} strokeWidth={on ? 2 : 1} strokeDasharray="3 3" />
              <circle cx={px} cy={PAD_T - 14} r={on ? 6 : 4.5} fill={c}
                className={p.pending ? "animate-pulse" : undefined} />
              <circle cx={px} cy={PAD_T - 14} r="12" fill="transparent" />
            </g>
          );
        })}

        {hoveredLap && (
          <line x1={x(hover!)} x2={x(hover!)} y1={PAD_T} y2={H - PAD_B}
            stroke="currentColor" strokeOpacity="0.25" strokeWidth="1" />
        )}
        {hoveredLap && (
          <circle cx={x(hover!)} cy={y(hoveredLap.lap_duration!)} r="4" fill={teamColor} />
        )}

        <text x={PAD_L} y={H - 6} className="fill-muted-foreground" fontSize="11">Lap {valid[0].lap_number}</text>
        <text x={W - PAD_R} y={H - 6} textAnchor="end" className="fill-muted-foreground" fontSize="11">
          Lap {valid[valid.length - 1].lap_number}
        </text>
      </svg>

      {hoveredLap && (
        <div
          className="pointer-events-none absolute top-2 rounded-lg border border-border bg-popover px-2.5 py-1.5 text-xs shadow-lg"
          style={{ left: `${(x(hover!) / W) * 100}%`, transform: "translateX(-50%)" }}
        >
          <span className="font-medium">Lap {hoveredLap.lap_number}</span>
          <span className="ml-2 tabular-nums">{fmtLap(hoveredLap.lap_duration)}</span>
          <span className="ml-2 tabular-nums text-muted-foreground">{fmtDelta(hoveredLap.lap_duration! - best)}</span>
        </div>
      )}
    </div>
  );
}

export default function F1() {
  const [year, setYear] = useState(new Date().getFullYear());
  const [sessions, setSessions] = useState<Session[] | null>(null);
  const [sessionKey, setSessionKey] = useState<number | null>(null);
  const [drivers, setDrivers] = useState<Driver[] | null>(null);
  const [driverNumber, setDriverNumber] = useState<number | null>(null);
  const [laps, setLaps] = useState<Lap[] | null>(null);
  const [radio, setRadio] = useState<RadioClip[] | null>(null);
  const [radioCounts, setRadioCounts] = useState<Map<number, number>>(new Map());
  const [analyses, setAnalyses] = useState<Map<string, Analysis>>(new Map());
  const [selected, setSelected] = useState<string | null>(null);
  const [analyzingAll, setAnalyzingAll] = useState(false);
  const wsRefs = useRef<Map<string, WebSocket>>(new Map());
  const cardRefs = useRef<Map<string, HTMLDivElement>>(new Map());

  useEffect(() => {
    setSessions(null);
    api.f1Sessions(year).then((r) => setSessions(r)).catch(() => setSessions([]));
  }, [year]);

  useEffect(() => {
    if (!sessionKey) return;
    setDrivers(null); setDriverNumber(null); setLaps(null); setRadio(null); setRadioCounts(new Map());
    api.f1Drivers(sessionKey).then((r) => setDrivers(r)).catch(() => setDrivers([]));
    // team radio coverage varies a lot by session — count clips per driver up front so the
    // picker doesn't send someone straight into a driver with zero recorded radio calls
    api.f1TeamRadioAll(sessionKey).then((clips: RadioClip[]) => {
      const counts = new Map<number, number>();
      for (const c of clips) counts.set(c.driver_number, (counts.get(c.driver_number) || 0) + 1);
      setRadioCounts(counts);
    }).catch(() => setRadioCounts(new Map()));
  }, [sessionKey]);

  useEffect(() => {
    if (!sessionKey || !driverNumber) return;
    setLaps(null); setRadio(null); setAnalyses(new Map()); setSelected(null);
    api.f1Laps(sessionKey, driverNumber).then((r) => setLaps(r)).catch(() => setLaps([]));
    // Seed from analyses already stored server-side, so calls analyzed in an earlier
    // visit come back showing their transcript instead of an empty "Analyze" button.
    // Keyed by clip.date like every other per-call map on this page; the stored rows
    // are matched to the fetched clips by recording_url, which is their natural key.
    api.f1TeamRadio(sessionKey, driverNumber).then(async (r: RadioClip[]) => {
      setRadio(r);
      const stored = await api.f1Analyses(sessionKey).catch(() => []);
      const byUrl = new Map<string, any>(stored.map((s: any) => [s.recording_url, s]));
      setAnalyses((prev) => {
        const next = new Map(prev);
        for (const c of r) {
          const s = byUrl.get(c.recording_url);
          if (s) next.set(c.date, {
            status: s.error ? "error" : "done", text: s.text, mood: s.mood, features: s.features,
          });
        }
        return next;
      });
    }).catch(() => setRadio([]));
  }, [sessionKey, driverNumber]);

  useEffect(() => () => wsRefs.current.forEach((ws) => ws.close()), []);

  // resolves when the worker reports back, so "Analyze all" can run one clip at a time
  // instead of dumping the whole session's radio onto the queue at once
  function analyze(clip: RadioClip): Promise<void> {
    const key = clip.date;
    setAnalyses((prev) => new Map(prev).set(key, { status: "pending" }));
    return new Promise<void>((resolve) => {
      api.f1Ingest(clip.recording_url, sessionKey ?? undefined, clip.driver_number).then((res) => {
        // Already analyzed server-side — no job was queued and no websocket will fire.
        if (res.cached) {
          setAnalyses((prev) => new Map(prev).set(key, {
            status: "done", text: res.text, mood: res.mood, features: res.features,
          }));
          resolve();
          return;
        }
        const ws = new WebSocket(`${location.origin.replace("http", "ws")}${res.ws_url}`);
        wsRefs.current.set(key, ws);
        ws.onmessage = (e) => {
          const msg = JSON.parse(e.data);
          if (msg.type !== "f1_result") return;
          setAnalyses((prev) => new Map(prev).set(key, {
            status: msg.error ? "error" : "done", text: msg.text, mood: msg.mood, features: msg.features,
          }));
          ws.close();
          wsRefs.current.delete(key);
          resolve();
        };
        ws.onerror = () => {
          setAnalyses((prev) => new Map(prev).set(key, { status: "error" }));
          resolve();
        };
      }).catch((e) => {
        setAnalyses((prev) => new Map(prev).set(key, { status: "error" }));
        toast.error("Ingest failed", { description: String(e) });
        resolve();
      });
    });
  }

  async function analyzeAll() {
    if (!radio) return;
    setAnalyzingAll(true);
    for (const clip of radio) {
      if (analyses.get(clip.date)?.status === "done") continue;
      await analyze(clip);
    }
    setAnalyzingAll(false);
  }

  const driver = drivers?.find((d) => d.driver_number === driverNumber);
  const teamColor = driver?.team_colour ? `#${driver.team_colour}` : "var(--primary)";

  const lapByClip = useMemo(() => {
    const m = new Map<string, number>();
    if (!laps || !radio) return m;
    for (const c of radio) {
      const l = nearestLap(laps, c.date);
      if (l) m.set(c.date, l.lap_number);
    }
    return m;
  }, [laps, radio]);

  const pins: Pin[] = useMemo(() => {
    if (!radio) return [];
    return radio.flatMap((c) => {
      const lap = lapByClip.get(c.date);
      if (lap == null) return [];
      const a = analyses.get(c.date);
      return [{ lap, date: c.date, mood: a?.status === "done" ? a.mood ?? null : null, pending: a?.status === "pending" }];
    });
  }, [radio, lapByClip, analyses]);

  // the point of the whole page: do the stressed calls sit on slower laps? compare each
  // mood's laps against the driver's own median lap so pace differences between races wash out
  const verdict = useMemo(() => {
    if (!laps) return null;
    const durByLap = new Map(laps.filter((l) => l.lap_duration != null).map((l) => [l.lap_number, l.lap_duration!]));
    const all = [...durByLap.values()].sort((a, b) => a - b);
    if (all.length < 5) return null;
    const median = quantile(all, 0.5);
    const byMood: Record<string, number[]> = {};
    for (const [date, a] of analyses) {
      if (a.status !== "done" || !a.mood) continue;
      const lap = lapByClip.get(date);
      const dur = lap != null ? durByLap.get(lap) : undefined;
      if (dur == null) continue;
      (byMood[a.mood] ||= []).push(dur - median);
    }
    const rows = (["calm", "stressed", "tired"] as Mood[])
      .filter((m) => byMood[m]?.length)
      .map((m) => ({ mood: m, n: byMood[m].length, delta: byMood[m].reduce((a, b) => a + b, 0) / byMood[m].length }));
    return rows.length ? { rows, median } : null;
  }, [analyses, laps, lapByClip]);

  const done = radio ? radio.filter((c) => analyses.get(c.date)?.status === "done").length : 0;

  return (
    <div>
      <PageHeader
        title="Race Radio"
        description="The Silent Co-Driver — transcribe driver radio calls, read the tone, and see if stress lines up with slower laps."
      />

      <div className="mx-auto max-w-5xl space-y-6 px-8 py-6">
        <div className="flex flex-wrap items-center gap-3">
          <Select value={String(year)} onValueChange={(v) => setYear(Number(v))}>
            <SelectTrigger className="w-28"><SelectValue /></SelectTrigger>
            <SelectContent>
              {[2024, 2025, 2026].map((y) => <SelectItem key={y} value={String(y)}>{y}</SelectItem>)}
            </SelectContent>
          </Select>

          {sessions === null ? (
            <Skeleton className="h-9 w-64" />
          ) : (
            <Select value={sessionKey ? String(sessionKey) : undefined} onValueChange={(v) => setSessionKey(Number(v))}>
              <SelectTrigger className="w-64"><SelectValue placeholder="Select a race…" /></SelectTrigger>
              <SelectContent>
                {sessions.map((s) => (
                  <SelectItem key={s.session_key} value={String(s.session_key)}>
                    {s.country_name} — {s.circuit_short_name} ({new Date(s.date_start).toLocaleDateString()})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}

          {sessionKey && (drivers === null ? (
            <Skeleton className="h-9 w-56" />
          ) : (
            <Select value={driverNumber ? String(driverNumber) : undefined} onValueChange={(v) => setDriverNumber(Number(v))}>
              <SelectTrigger className="w-64"><SelectValue placeholder="Select a driver…" /></SelectTrigger>
              <SelectContent>
                {[...drivers].sort((a, b) => (radioCounts.get(b.driver_number) || 0) - (radioCounts.get(a.driver_number) || 0)).map((d) => {
                  const count = radioCounts.get(d.driver_number) || 0;
                  return (
                    <SelectItem key={d.driver_number} value={String(d.driver_number)}>
                      <span className="size-2 rounded-full" style={{ background: `#${d.team_colour}` }} />
                      {d.broadcast_name} · {d.team_name}
                      <span className="text-muted-foreground">{count > 0 ? `(${count} radio)` : "(no radio)"}</span>
                    </SelectItem>
                  );
                })}
              </SelectContent>
            </Select>
          ))}
        </div>

        {driver && (
          <div className="animate-slide-up space-y-6">
            {/* driver identity — team colour carries through the chart and pins below */}
            <div className="flex items-center gap-4 overflow-hidden rounded-xl border border-border bg-card">
              <div className="h-16 w-1.5 shrink-0" style={{ background: teamColor }} />
              <div className="flex flex-1 flex-wrap items-baseline gap-x-3 gap-y-1 py-3 pr-4">
                <span className="text-2xl font-semibold tabular-nums" style={{ color: teamColor }}>
                  {driver.driver_number}
                </span>
                <span className="text-lg font-medium">{driver.broadcast_name}</span>
                <span className="text-sm text-muted-foreground">{driver.team_name}</span>
                <span className="ml-auto flex items-center gap-4 text-sm text-muted-foreground">
                  <span>{laps?.filter((l) => l.lap_duration).length ?? "—"} laps</span>
                  <span>{radio?.length ?? "—"} radio calls</span>
                  <span>{done} analyzed</span>
                </span>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm font-medium">
                <Flag className="size-4 text-muted-foreground" /> Lap times
                <span className="text-xs font-normal text-muted-foreground">
                  · pins mark radio calls — click one to jump to it
                </span>
              </div>
              {laps === null ? <Skeleton className="h-60 w-full rounded-xl" /> : (
                <LapTrace
                  laps={laps} pins={pins} teamColor={teamColor} selected={selected}
                  onSelect={(d) => {
                    setSelected(d);
                    if (d) cardRefs.current.get(d)?.scrollIntoView({ behavior: "smooth", block: "center" });
                  }}
                />
              )}
              <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
                {(["calm", "stressed", "tired"] as Mood[]).map((m) => (
                  <span key={m} className="flex items-center gap-1.5">
                    <span className={cn("size-2 rounded-full", moodDot(m))} /> {m}
                  </span>
                ))}
                <span className="flex items-center gap-1.5">
                  <span className="size-2 rounded-full bg-muted-foreground" /> not analyzed
                </span>
              </div>
            </div>

            {verdict && (
              <div className="rounded-xl border border-border bg-card p-4">
                <div className="mb-3 flex items-center gap-2 text-sm font-medium">
                  <TrendingUp className="size-4 text-muted-foreground" /> Tone vs pace
                  <span className="text-xs font-normal text-muted-foreground">
                    · lap time on each call's lap, against this driver's median lap ({fmtLap(verdict.median)})
                  </span>
                </div>
                <div className="grid gap-3 sm:grid-cols-3">
                  {verdict.rows.map((r) => (
                    <div key={r.mood} className="rounded-lg border border-border/60 p-3">
                      <Badge variant="outline" className={cn("capitalize", MOOD_STYLE[r.mood])}>{r.mood}</Badge>
                      <div className="mt-2 text-2xl font-semibold tabular-nums" style={{ color: MOOD_VAR[r.mood] }}>
                        {fmtDelta(r.delta)}
                      </div>
                      <div className="text-xs text-muted-foreground">{r.n} call{r.n === 1 ? "" : "s"}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm font-medium">
                <Radio className="size-4 text-muted-foreground" /> Team radio
                {radio && radio.length > 0 && (
                  <Button size="sm" variant="secondary" className="ml-auto" disabled={analyzingAll} onClick={analyzeAll}>
                    {analyzingAll
                      ? <><Loader2 className="size-3.5 animate-spin" /> Analyzing {Math.min(done + 1, radio.length)}/{radio.length}</>
                      : <><Sparkles className="size-3.5" /> Analyze all ({radio.length})</>}
                  </Button>
                )}
              </div>
              {radio === null ? (
                <Skeleton className="h-24 w-full rounded-lg" />
              ) : radio.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No radio calls recorded for this driver in this session — OpenF1's team radio
                  coverage varies a lot by session. Try a driver marked with a radio count above.
                </p>
              ) : (
                <div className="space-y-3">
                  {radio.map((clip) => {
                    const a = analyses.get(clip.date);
                    const lap = lapByClip.get(clip.date);
                    const on = selected === clip.date;
                    return (
                      <div
                        key={clip.date}
                        ref={(el) => { el ? cardRefs.current.set(clip.date, el) : cardRefs.current.delete(clip.date); }}
                        onMouseEnter={() => setSelected(clip.date)}
                        onMouseLeave={() => setSelected((s) => (s === clip.date ? null : s))}
                        className={cn(
                          "rounded-xl border border-l-4 border-border bg-card p-3 transition-colors",
                          on && "border-border/80 bg-accent/40"
                        )}
                        style={{ borderLeftColor: a?.status === "done" && a.mood ? MOOD_VAR[a.mood] : "transparent" }}
                      >
                        <div className="mb-2 flex items-center justify-between gap-2">
                          <span className="flex items-center gap-2 text-xs text-muted-foreground">
                            {lap != null && <span className="rounded bg-secondary px-1.5 py-0.5 font-medium tabular-nums text-secondary-foreground">Lap {lap}</span>}
                            {new Date(clip.date).toLocaleTimeString()}
                          </span>
                          <div className="flex items-center gap-2">
                            {a?.status === "done" && a.mood && (
                              <Badge variant="outline" className={cn("capitalize", MOOD_STYLE[a.mood])}>{a.mood}</Badge>
                            )}
                            <Button
                              size="sm" variant="outline"
                              disabled={a?.status === "pending"}
                              onClick={() => analyze(clip)}
                            >
                              {a?.status === "pending" ? <Loader2 className="size-3.5 animate-spin" /> : null}
                              {a ? "Re-analyze" : "Transcribe & analyze"}
                            </Button>
                          </div>
                        </div>
                        <AudioPlayer src={clip.recording_url} />
                        {a?.status === "done" && (
                          <>
                            <p className="mt-2 text-sm text-foreground/90">{a.text || <span className="text-muted-foreground">(no speech detected)</span>}</p>
                            {a.features && (
                              <div className="mt-2 flex flex-wrap gap-3 text-xs tabular-nums text-muted-foreground">
                                <span>pitch {a.features.pitch_mean_hz || 0} Hz</span>
                                <span>variation ±{a.features.pitch_std_hz || 0} Hz</span>
                                <span>{a.features.speech_rate_wps || 0} words/s</span>
                                <span>voiced {Math.round((a.features.voiced_ratio || 0) * 100)}%</span>
                              </div>
                            )}
                          </>
                        )}
                        {a?.status === "error" && (
                          <p className="mt-2 text-sm text-destructive">Analysis failed.</p>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
