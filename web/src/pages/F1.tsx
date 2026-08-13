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
import { Loader2, Radio, Flag } from "lucide-react";

type Session = { session_key: number; session_name: string; country_name: string; circuit_short_name: string; date_start: string; year: number };
type Driver = { driver_number: number; broadcast_name: string; team_name: string; team_colour: string };
type Lap = { lap_number: number; lap_duration: number | null; date_start: string };
type RadioClip = { driver_number: number; date: string; recording_url: string };
type Analysis = { status: "pending" | "done" | "error"; text?: string; mood?: Mood; features?: Record<string, number> };

function fmtLap(s: number | null) {
  if (!s) return "—";
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(3);
  return m > 0 ? `${m}:${sec.padStart(6, "0")}` : `${sec}s`;
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

function LapChart({ laps, highlightLap, moodByLap }: { laps: Lap[]; highlightLap?: number; moodByLap: Map<number, Mood> }) {
  const valid = laps.filter((l) => l.lap_duration != null);
  if (valid.length === 0) return <p className="text-sm text-muted-foreground">No lap data yet.</p>;
  const max = Math.max(...valid.map((l) => l.lap_duration!));
  const min = Math.min(...valid.map((l) => l.lap_duration!));
  const w = 100 / valid.length;

  return (
    <div className="flex h-40 items-end gap-px rounded-lg border border-border bg-card p-3">
      {valid.map((l) => {
        const heightPct = 15 + ((l.lap_duration! - min) / Math.max(max - min, 0.001)) * 85;
        const mood = moodByLap.get(l.lap_number);
        return (
          <div
            key={l.lap_number}
            title={`Lap ${l.lap_number}: ${fmtLap(l.lap_duration)}`}
            className={cn(
              "group relative flex-1 rounded-t-sm transition-all",
              mood ? moodDot(mood) : "bg-primary/30 hover:bg-primary/50",
              highlightLap === l.lap_number && "ring-2 ring-white"
            )}
            style={{ height: `${heightPct}%`, minWidth: Math.max(w, 0.5) + "%" }}
          />
        );
      })}
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
  const wsRefs = useRef<Map<string, WebSocket>>(new Map());

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
    setLaps(null); setRadio(null); setAnalyses(new Map());
    api.f1Laps(sessionKey, driverNumber).then((r) => setLaps(r)).catch(() => setLaps([]));
    api.f1TeamRadio(sessionKey, driverNumber).then((r) => setRadio(r)).catch(() => setRadio([]));
  }, [sessionKey, driverNumber]);

  useEffect(() => () => wsRefs.current.forEach((ws) => ws.close()), []);

  async function analyze(clip: RadioClip) {
    const key = clip.date;
    setAnalyses((prev) => new Map(prev).set(key, { status: "pending" }));
    try {
      const res = await api.f1Ingest(clip.recording_url, sessionKey ?? undefined, clip.driver_number);
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
      };
      ws.onerror = () => {
        setAnalyses((prev) => new Map(prev).set(key, { status: "error" }));
      };
    } catch (e) {
      setAnalyses((prev) => new Map(prev).set(key, { status: "error" }));
      toast.error("Ingest failed", { description: String(e) });
    }
  }

  const moodByLap = useMemo(() => {
    const m = new Map<number, Mood>();
    if (!laps) return m;
    for (const [date, a] of analyses) {
      if (a.status !== "done" || !a.mood) continue;
      const clip = radio?.find((r) => r.date === date);
      if (!clip) continue;
      const lap = nearestLap(laps, clip.date);
      if (lap) m.set(lap.lap_number, a.mood);
    }
    return m;
  }, [analyses, laps, radio]);

  return (
    <div>
      <PageHeader
        title="Race Radio"
        description="The Silent Co-Driver — transcribe driver radio calls, read the tone, and see if stress lines up with slower laps."
      />

      <div className="mx-auto max-w-4xl space-y-6 px-8 py-6">
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

        {driverNumber && (
          <div className="animate-slide-up space-y-6">
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm font-medium">
                <Flag className="size-4 text-muted-foreground" /> Lap times
              </div>
              {laps === null ? <Skeleton className="h-40 w-full rounded-lg" /> : (
                <LapChart laps={laps} moodByLap={moodByLap} />
              )}
              {moodByLap.size > 0 && (
                <div className="flex gap-3 text-xs text-muted-foreground">
                  {(["calm", "stressed", "tired"] as Mood[]).map((m) => (
                    <span key={m} className="flex items-center gap-1.5">
                      <span className={cn("size-2 rounded-full", moodDot(m))} /> {m}
                    </span>
                  ))}
                </div>
              )}
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm font-medium">
                <Radio className="size-4 text-muted-foreground" /> Team radio
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
                    return (
                      <div key={clip.date} className="rounded-xl border border-border bg-card p-3">
                        <div className="mb-2 flex items-center justify-between">
                          <span className="text-xs text-muted-foreground">
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
                          <p className="mt-2 text-sm text-foreground/90">{a.text || <span className="text-muted-foreground">(no speech detected)</span>}</p>
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
