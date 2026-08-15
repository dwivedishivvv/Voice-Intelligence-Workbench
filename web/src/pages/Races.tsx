import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/api/client";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Select, SelectTrigger, SelectValue, SelectContent, SelectItem,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { speakerColor } from "@/lib/speaker-color";
import {
  Flag, Plus, Upload as UploadIcon, Loader2, Users, ArrowLeft, Trash2, FileAudio,
  Search, X, Pencil, Check, ArrowUpRight,
} from "lucide-react";

type Race = {
  id: string; name: string; circuit: string | null; race_date: string | null;
  session_key: number | null; svg_path: string | null; notes: string | null;
  n_clips?: number; n_complete?: number; avg_sentiment?: number | null;
};
type Sentiment = "negative" | "neutral" | "positive";
type RaceClip = {
  id: string; filename: string; status: string; duration_s: number | null;
  n_speakers: number | null; language: string | null; sentiment: Sentiment | null;
  sentiment_score: number | null; mood: string | null; transcript: string | null;
};
type RaceSpeaker = { id: string; display_name: string; n_clips: number; speech_s: number | null };
type Utterance = {
  id: string; clip_id: string; filename: string; start_s: number; end_s: number; text: string;
  local_label: string | null; display_name: string | null; cluster_id: string | null;
  sentiment: Sentiment | null; sentiment_score: number | null;
  text_sentiment: Sentiment | null; mood: string | null;
};

const SENTIMENT_STYLE: Record<Sentiment, string> = {
  positive: "bg-success/15 text-success border-success/30",
  negative: "bg-destructive/15 text-destructive border-destructive/30",
  neutral: "bg-muted text-muted-foreground border-border",
};

const TERMINAL = ["COMPLETE", "FAILED", "REJECTED", "DEAD"];

function fmtDur(s: number | null) {
  if (!s) return "—";
  return s >= 60 ? `${Math.floor(s / 60)}:${Math.floor(s % 60).toString().padStart(2, "0")}` : `${s.toFixed(1)}s`;
}

/** Track outline. Fetched rather than <img src> so it inlines into the DOM and inherits
 *  currentColor — an <img> would render it as an opaque bitmap that can't be themed. */
function TrackOutline({ raceId }: { raceId: string }) {
  const [svg, setSvg] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let alive = true;
    fetch(api.raceSvgUrl(raceId))
      .then((r) => (r.ok ? r.text() : Promise.reject()))
      .then((t) => alive && setSvg(t))
      .catch(() => alive && setFailed(true));
    return () => { alive = false; };
  }, [raceId]);
  if (failed || !svg) return null;
  // The wrapper needs a definite size of its own: with an auto-height parent, h-full on the
  // <svg> resolves against nothing and the viewBox renders at intrinsic scale, overflowing
  // and cropping the outline. h-full w-full here gives preserveAspectRatio something to fit
  // against, so the track letterboxes instead.
  return (
    <div
      className="pointer-events-none h-full w-full [&_svg]:h-full [&_svg]:w-full [&_path]:fill-none [&_path]:stroke-current [&_path]:stroke-[6] [&_polyline]:fill-none [&_polyline]:stroke-current [&_polyline]:stroke-[6]"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}

function SentimentBar({ clips }: { clips: RaceClip[] }) {
  const counts = useMemo(() => {
    const c = { positive: 0, neutral: 0, negative: 0 };
    for (const x of clips) if (x.sentiment) c[x.sentiment]++;
    return c;
  }, [clips]);
  const total = counts.positive + counts.neutral + counts.negative;
  if (!total) return null;
  return (
    <div className="space-y-1.5">
      <div className="flex h-2 overflow-hidden rounded-full">
        <div className="bg-destructive" style={{ width: `${(counts.negative / total) * 100}%` }} />
        {/* bg-muted is nearly the card's own colour on a dark theme — the neutral share read
            as an empty gap rather than a segment, so this needs the foreground tint */}
        <div className="bg-muted-foreground/40" style={{ width: `${(counts.neutral / total) * 100}%` }} />
        <div className="bg-success" style={{ width: `${(counts.positive / total) * 100}%` }} />
      </div>
      <div className="flex gap-3 text-xs text-muted-foreground">
        <span>{counts.negative} negative</span>
        <span>{counts.neutral} neutral</span>
        <span>{counts.positive} positive</span>
      </div>
    </div>
  );
}

const MOODS = ["calm", "stressed", "tired"] as const;
const MOOD_BAR: Record<string, string> = { calm: "bg-success", stressed: "bg-destructive", tired: "bg-warning" };
// SVG can't read the tailwind colour classes, so mirror them onto the same CSS variables
const SENT_VAR: Record<Sentiment, string> = {
  positive: "var(--success)", neutral: "var(--muted-foreground)", negative: "var(--destructive)",
};

/** Column chart of per-recording sentiment in upload order. viewBox + w-full only: the
 *  intrinsic aspect ratio then gives the <svg> a definite height, which h-full inside an
 *  auto-height parent would not. */
function SentimentTrace({ clips, onJump }: { clips: RaceClip[]; onJump: (id: string) => void }) {
  const pts = clips.filter((c) => c.sentiment_score != null && c.sentiment);
  if (pts.length < 2) return <p className="text-sm text-muted-foreground">Not enough scored recordings to chart.</p>;
  const W = 1000, H = 180, PAD_L = 34, PAD_R = 8, PAD_T = 10, PAD_B = 18;
  const step = (W - PAD_L - PAD_R) / pts.length;
  const bw = Math.max(2, Math.min(step * 0.7, 22));
  const y = (v: number) => PAD_T + (1 - (v + 1) / 2) * (H - PAD_T - PAD_B);
  const zero = y(0);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[1, 0, -1].map((t) => (
        <g key={t}>
          <line x1={PAD_L} x2={W - PAD_R} y1={y(t)} y2={y(t)} stroke="currentColor" strokeOpacity={t === 0 ? 0.25 : 0.08} />
          <text x={PAD_L - 6} y={y(t) + 4} textAnchor="end" className="fill-muted-foreground" fontSize="11">
            {t > 0 ? "+1" : t}
          </text>
        </g>
      ))}
      {pts.map((c, i) => {
        const v = c.sentiment_score!;
        const cx = PAD_L + step * i + step / 2;
        return (
          <rect
            key={c.id} x={cx - bw / 2} y={Math.min(y(v), zero)} width={bw}
            height={Math.max(1.5, Math.abs(zero - y(v)))} rx="2"
            fill={SENT_VAR[c.sentiment!]} fillOpacity="0.85"
            className="cursor-pointer"
            onClick={() => onJump(c.id)}
          >
            <title>{`${c.filename} · ${c.sentiment} ${v > 0 ? "+" : ""}${v.toFixed(2)}`}</title>
          </rect>
        );
      })}
      <text x={PAD_L} y={H - 4} className="fill-muted-foreground" fontSize="11">first recording</text>
      <text x={W - PAD_R} y={H - 4} textAnchor="end" className="fill-muted-foreground" fontSize="11">last</text>
    </svg>
  );
}

function Bar({ pct, className }: { pct: number; className: string }) {
  return <div className={cn("h-full", className)} style={{ width: `${pct}%` }} />;
}

function AnalysisPanel({
  utterances, clips, voiceOf, onJump,
}: {
  utterances: Utterance[]; clips: RaceClip[];
  voiceOf: (u: Utterance) => { key: string; name: string } | null;
  onJump: (clipId: string) => void;
}) {
  const voices = useMemo(() => {
    const m = new Map<string, { name: string; secs: number; n: number; moods: Record<string, number> }>();
    for (const u of utterances) {
      const v = voiceOf(u);
      if (!v) continue;
      const e = m.get(v.key) ?? { name: v.name, secs: 0, n: 0, moods: { calm: 0, stressed: 0, tired: 0 } };
      e.secs += Math.max(0, u.end_s - u.start_s);
      e.n++;
      if (u.mood && e.moods[u.mood] != null) e.moods[u.mood]++;
      m.set(v.key, e);
    }
    return [...m.values()].sort((a, b) => b.secs - a.secs);
  }, [utterances, voiceOf]);

  const totalSecs = voices.reduce((s, v) => s + v.secs, 0) || 1;

  const extremes = useMemo(() => {
    const scored = utterances.filter((u) => u.sentiment_score != null && u.text?.trim());
    const byScore = [...scored].sort((a, b) => a.sentiment_score! - b.sentiment_score!);
    return { neg: byScore.slice(0, 5), pos: byScore.slice(-5).reverse() };
  }, [utterances]);

  const langs = useMemo(() => {
    const m = new Map<string, number>();
    for (const c of clips) if (c.language) m.set(c.language, (m.get(c.language) ?? 0) + 1);
    return [...m].sort((a, b) => b[1] - a[1]);
  }, [clips]);

  if (!utterances.length) {
    return <p className="py-10 text-center text-sm text-muted-foreground">Nothing analysed yet — recordings need to finish processing first.</p>;
  }

  const moment = (u: Utterance) => {
    const v = voiceOf(u);
    return (
      <button key={u.id} onClick={() => onJump(u.clip_id)}
              className="flex w-full items-baseline gap-2 rounded-lg px-2 py-1.5 text-left text-sm hover:bg-muted/50">
        <Badge variant="outline" className={cn("shrink-0 rounded-full px-1.5 py-0 text-[10px]",
               v ? speakerColor(v.name).chip : "text-muted-foreground")}>
          {v?.name ?? "Unknown voice"}
        </Badge>
        <span className="flex-1 line-clamp-2">{u.text}</span>
        <span className={cn("shrink-0 tabular-nums text-xs",
              u.sentiment_score! > 0 ? "text-success" : "text-destructive")}>
          {u.sentiment_score! > 0 ? "+" : ""}{u.sentiment_score!.toFixed(2)}
        </span>
      </button>
    );
  };

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-border bg-card p-5">
          <p className="mb-3 text-xs font-medium text-muted-foreground">Talk share</p>
          <div className="space-y-2">
            {voices.map((v) => (
              <div key={v.name} className="space-y-1">
                <div className="flex items-baseline justify-between text-xs">
                  <span className="flex items-center gap-1.5">
                    <span className={cn("size-1.5 rounded-full", speakerColor(v.name).dot)} />{v.name}
                  </span>
                  <span className="tabular-nums text-muted-foreground">
                    {fmtDur(v.secs)} · {((v.secs / totalSecs) * 100).toFixed(0)}% · {v.n} utt
                  </span>
                </div>
                <div className="flex h-1.5 overflow-hidden rounded-full bg-muted">
                  <Bar pct={(v.secs / totalSecs) * 100} className={speakerColor(v.name).dot} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card p-5">
          <p className="mb-3 text-xs font-medium text-muted-foreground">Mood per voice</p>
          <div className="space-y-2">
            {voices.map((v) => {
              const tot = MOODS.reduce((s, m) => s + v.moods[m], 0);
              return (
                <div key={v.name} className="space-y-1">
                  <div className="flex items-baseline justify-between text-xs">
                    <span>{v.name}</span>
                    <span className="tabular-nums text-muted-foreground">
                      {tot ? MOODS.filter((m) => v.moods[m]).map((m) => `${v.moods[m]} ${m}`).join(" · ") : "—"}
                    </span>
                  </div>
                  <div className="flex h-1.5 overflow-hidden rounded-full bg-muted">
                    {tot > 0 && MOODS.map((m) => <Bar key={m} pct={(v.moods[m] / tot) * 100} className={MOOD_BAR[m]} />)}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="rounded-xl border border-border bg-card p-5">
        <p className="mb-3 text-xs font-medium text-muted-foreground">Sentiment over the race</p>
        <SentimentTrace clips={clips} onJump={onJump} />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-xl border border-border bg-card p-4">
          <p className="mb-2 px-2 text-xs font-medium text-muted-foreground">Most negative moments</p>
          {extremes.neg.map(moment)}
        </div>
        <div className="rounded-xl border border-border bg-card p-4">
          <p className="mb-2 px-2 text-xs font-medium text-muted-foreground">Most positive moments</p>
          {extremes.pos.map(moment)}
        </div>
      </div>

      {langs.length > 1 && (
        <div className="rounded-xl border border-border bg-card p-5">
          <p className="mb-3 text-xs font-medium text-muted-foreground">Languages</p>
          <div className="flex flex-wrap gap-2">
            {langs.map(([l, n]) => (
              <Badge key={l} variant="outline" className="rounded-full">
                <span className="uppercase">{l}</span>
                <span className="ml-1.5 tabular-nums opacity-60">{n}</span>
              </Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function CreateRaceForm({ onCreated, onCancel }: { onCreated: (r: Race) => void; onCancel: () => void }) {
  const [busy, setBusy] = useState(false);
  const [svg, setSvg] = useState<File | null>(null);
  const [fields, setFields] = useState({ name: "", circuit: "", race_date: "", session_key: "", notes: "" });
  const set = (k: string) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setFields((f) => ({ ...f, [k]: e.target.value }));

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!fields.name.trim()) return toast.error("Name is required");
    setBusy(true);
    try {
      onCreated(await api.createRace(fields, svg));
    } catch (err) {
      toast.error(String((err as Error).message));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="mx-auto max-w-xl space-y-5 px-8 py-8">
      <div className="space-y-2">
        <Label htmlFor="name">Race name</Label>
        <Input id="name" value={fields.name} onChange={set("name")} placeholder="2018 Australian Grand Prix" autoFocus />
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label htmlFor="circuit">Circuit</Label>
          <Input id="circuit" value={fields.circuit} onChange={set("circuit")} placeholder="Albert Park" />
        </div>
        <div className="space-y-2">
          <Label htmlFor="race_date">Date</Label>
          {/* native date input: no picker dependency, correct locale + keyboard handling free */}
          <Input id="race_date" type="date" value={fields.race_date} onChange={set("race_date")} />
        </div>
      </div>
      <div className="space-y-2">
        <Label htmlFor="session_key">OpenF1 session key <span className="text-muted-foreground">(optional)</span></Label>
        <Input id="session_key" inputMode="numeric" value={fields.session_key} onChange={set("session_key")} placeholder="9158" />
        <p className="text-xs text-muted-foreground">Links this race to live OpenF1 data. Leave blank to keep it standalone.</p>
      </div>
      <div className="space-y-2">
        <Label htmlFor="svg">Track layout <span className="text-muted-foreground">(SVG outline)</span></Label>
        <Input id="svg" type="file" accept=".svg,image/svg+xml"
               onChange={(e) => setSvg(e.target.files?.[0] ?? null)} />
        <p className="text-xs text-muted-foreground">
          Outline only. Scripts and external references are stripped on upload.
        </p>
      </div>
      <div className="flex gap-2 pt-2">
        <Button type="submit" disabled={busy}>
          {busy && <Loader2 className="mr-2 size-4 animate-spin" />}Create race
        </Button>
        <Button type="button" variant="ghost" onClick={onCancel}>Cancel</Button>
      </div>
    </form>
  );
}

function RaceDetail({ raceId, onBack }: { raceId: string; onBack: () => void }) {
  const [race, setRace] = useState<(Race & { clips: RaceClip[]; speakers: RaceSpeaker[] }) | null>(null);
  const [utterances, setUtterances] = useState<Utterance[]>([]);
  const [uploading, setUploading] = useState(0);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const [tab, setTab] = useState("recordings");
  const [voice, setVoice] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [sentiment, setSentiment] = useState("all");
  const [mood, setMood] = useState("all");
  const [status, setStatus] = useState("all");
  const [naming, setNaming] = useState<string | null>(null);
  const [nameDraft, setNameDraft] = useState("");
  const [highlight, setHighlight] = useState<string | null>(null);

  const load = useCallback(async () => {
    const r = await api.getRace(raceId);
    setRace(r);
    if (r.clips.some((c: RaceClip) => c.status === "COMPLETE")) {
      setUtterances((await api.raceUtterances(raceId)).items);
    }
    return r;
  }, [raceId]);

  useEffect(() => { load().catch((e) => toast.error(String(e.message))); }, [load]);

  // poll only while something is still in flight — a race whose clips are all terminal
  // is a static page, and polling it forever is pure waste
  const pending = race?.clips.filter((c) => !TERMINAL.includes(c.status)).length ?? 0;
  useEffect(() => {
    if (!pending) return;
    const t = setInterval(() => load().catch(() => {}), 3000);
    return () => clearInterval(t);
  }, [pending, load]);

  async function uploadFiles(files: File[]) {
    const audio = files.filter((f) => f.type.startsWith("audio/") || /\.(mp3|wav|m4a|ogg|flac|webm|aac)$/i.test(f.name));
    if (!audio.length) return toast.error("No audio files in that drop");
    setUploading(audio.length);
    let ok = 0, failed = 0;
    // bounded concurrency: the API streams each file to disk, and firing 200 at once just
    // buries the event loop and the browser's own connection pool for no extra throughput
    const queue = [...audio];
    await Promise.all(Array.from({ length: 6 }, async () => {
      while (queue.length) {
        const f = queue.shift()!;
        try { await api.uploadRaceClip(raceId, f); ok++; } catch { failed++; }
        setUploading((n) => n - 1);
      }
    }));
    toast[failed ? "warning" : "success"](`${ok} uploaded${failed ? `, ${failed} failed` : ""}`);
    load().catch(() => {});
  }

  // "Voice N", numbered by first appearance across the race. This is keyed on cluster_id,
  // NOT local_label: local_label is pyannote's per-clip speaker index, so SPEAKER_00 in one
  // recording and SPEAKER_00 in the next are unrelated people. The cluster is what actually
  // ties one voice together across clips, so it's the only honest label to show here.
  const voiceNames = useMemo(() => {
    const m = new Map<string, string>();
    for (const u of utterances) {
      if (u.cluster_id && !m.has(u.cluster_id)) m.set(u.cluster_id, `Voice ${m.size + 1}`);
    }
    return m;
  }, [utterances]);

  // one identity key per utterance. An enrolled profile wins over the cluster because it is
  // the stronger claim; local_label is deliberately never part of the key.
  const voiceOf = useCallback((u: Utterance) => {
    if (u.display_name) return { key: `name:${u.display_name}`, name: u.display_name };
    if (u.cluster_id) return { key: `cluster:${u.cluster_id}`, name: voiceNames.get(u.cluster_id) ?? "Voice" };
    return null;
  }, [voiceNames]);

  const uttFiltersOn = voice != null || q.trim() !== "" || sentiment !== "all" || mood !== "all";

  const shown = useMemo(() => utterances.filter((u) => {
    if (voice && voiceOf(u)?.key !== voice) return false;
    if (sentiment !== "all" && u.sentiment !== sentiment) return false;
    if (mood !== "all" && u.mood !== mood) return false;
    if (q.trim() && !(u.text ?? "").toLowerCase().includes(q.trim().toLowerCase())) return false;
    return true;
  }), [utterances, voice, sentiment, mood, q, voiceOf]);

  const visibleClips = useMemo(() => {
    const withHit = new Set(shown.map((u) => u.clip_id));
    return (race?.clips ?? []).filter((c) => {
      if (status === "complete" && c.status !== "COMPLETE") return false;
      if (status === "processing" && TERMINAL.includes(c.status)) return false;
      if (status === "review" && !["REJECTED", "FAILED", "DEAD"].includes(c.status)) return false;
      return !uttFiltersOn || withHit.has(c.id);
    });
  }, [race?.clips, shown, status, uttFiltersOn]);

  const clearAll = () => { setVoice(null); setQ(""); setSentiment("all"); setMood("all"); setStatus("all"); };

  // jumping from the Analysis tab: switch tabs first, then scroll once the list is mounted
  const jumpToClip = (clipId: string) => {
    setTab("recordings");
    setHighlight(clipId);
    requestAnimationFrame(() =>
      document.getElementById(`clip-${clipId}`)?.scrollIntoView({ behavior: "smooth", block: "center" }));
  };

  async function submitName(clusterId: string) {
    const name = nameDraft.trim();
    if (!name) return;
    try {
      await api.nameRaceVoice(raceId, clusterId, name);
      toast.success(`Voice named “${name}”`);
      setNaming(null); setNameDraft("");
      load().catch(() => {});
    } catch (err) {
      toast.error(String((err as Error).message));
    }
  }

  if (!race) return <div className="space-y-3 p-8"><Skeleton className="h-8 w-64" /><Skeleton className="h-32 w-full" /></div>;

  const complete = race.clips.filter((c) => c.status === "COMPLETE").length;
  const anyFilter = uttFiltersOn || status !== "all";
  const voiceLabel = !voice ? null
    : voice.startsWith("name:") ? voice.slice(5) : voiceNames.get(voice.slice(8)) ?? "voice";

  return (
    <div>
      <PageHeader
        title={race.name}
        description={[race.circuit, race.race_date].filter(Boolean).join(" · ") || undefined}
        actions={
          <>
            <Button variant="ghost" size="sm" onClick={onBack}><ArrowLeft className="mr-1.5 size-4" />All races</Button>
            <Button size="sm" onClick={() => inputRef.current?.click()}>
              <UploadIcon className="mr-1.5 size-4" />Add recordings
            </Button>
          </>
        }
      />

      <input ref={inputRef} type="file" multiple accept="audio/*,.mp3,.wav,.m4a,.ogg,.flac"
             className="hidden"
             onChange={(e) => { uploadFiles(Array.from(e.target.files ?? [])); e.target.value = ""; }} />

      <div className="space-y-6 px-8 py-6">
        <div className="grid gap-6 md:grid-cols-[260px_1fr]">
          <div className="flex h-56 items-center justify-center rounded-xl border border-border bg-card p-4 text-muted-foreground/50">
            {race.svg_path ? <TrackOutline raceId={race.id} />
                           : <span className="text-xs">No track layout</span>}
          </div>

          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); uploadFiles(Array.from(e.dataTransfer.files)); }}
            className={cn(
              "flex h-56 flex-col items-center justify-center gap-3 rounded-xl border border-dashed p-6 text-center transition-colors",
              dragging ? "border-primary bg-primary/5" : "border-border"
            )}
          >
            {uploading > 0 ? (
              <><Loader2 className="size-6 animate-spin text-primary" />
                <p className="text-sm">Uploading {uploading} file{uploading === 1 ? "" : "s"}…</p></>
            ) : (
              <><FileAudio className="size-7 text-muted-foreground" />
                <div>
                  <p className="text-sm font-medium">Drop team radio recordings here</p>
                  <p className="text-xs text-muted-foreground">
                    Bulk drop a whole folder. Each file is transcribed, diarized, fingerprinted and scored for sentiment.
                  </p>
                </div></>
            )}
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-4">
          {[
            ["Recordings", String(race.clips.length)],
            ["Processed", `${complete}/${race.clips.length}`],
            ["Speakers identified", String(race.speakers.length)],
            ["Utterances", String(utterances.length)],
          ].map(([k, v]) => (
            <div key={k} className="rounded-xl border border-border bg-card px-4 py-3">
              <p className="text-xs text-muted-foreground">{k}</p>
              <p className="mt-0.5 text-xl font-semibold tabular-nums">{v}</p>
            </div>
          ))}
        </div>

        {complete > 0 && (
          <div className="rounded-xl border border-border bg-card p-5">
            <p className="mb-3 text-xs font-medium text-muted-foreground">Sentiment across the race</p>
            <SentimentBar clips={race.clips} />
          </div>
        )}

        {(race.speakers.length > 0 || voiceNames.size > 0) && (
          <div className="flex flex-wrap items-center gap-3 rounded-xl border border-border bg-card px-4 py-3">
            <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
              <Users className="size-3.5" />Voices
            </span>
            {race.speakers.map((s) => {
              const key = `name:${s.display_name}`;
              return (
                <button key={s.id} onClick={() => setVoice((v) => (v === key ? null : key))}
                        className={cn("flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs transition-shadow",
                          speakerColor(s.display_name).chip,
                          voice === key ? "ring-2 ring-primary ring-offset-1 ring-offset-card" : "hover:brightness-125")}>
                  <span className={cn("size-1.5 rounded-full", speakerColor(s.display_name).dot)} />
                  {s.display_name}
                  <span className="opacity-60">{s.n_clips} clip{s.n_clips === 1 ? "" : "s"}</span>
                </button>
              );
            })}
            {/* unenrolled voices, grouped by fingerprint. Showing how many clips each one
                spans is the whole point — it's the evidence the same voice was recognised
                across recordings, and what makes "name this voice" worth doing. */}
            {[...voiceNames].map(([cid, name]) => {
              const nClips = new Set(utterances.filter((u) => u.cluster_id === cid).map((u) => u.clip_id)).size;
              const key = `cluster:${cid}`;
              if (naming === cid) {
                return (
                  <span key={cid} className="flex items-center gap-1 rounded-full border border-border px-1.5 py-0.5">
                    <Input autoFocus value={nameDraft} onChange={(e) => setNameDraft(e.target.value)}
                           onKeyDown={(e) => {
                             if (e.key === "Enter") submitName(cid);
                             if (e.key === "Escape") { setNaming(null); setNameDraft(""); }
                           }}
                           placeholder={`Name ${name}`} className="h-6 w-36 border-0 bg-transparent px-1 text-xs shadow-none focus-visible:ring-0" />
                    <button onClick={() => submitName(cid)} className="rounded p-1 hover:bg-muted" aria-label="Save name">
                      <Check className="size-3" />
                    </button>
                    <button onClick={() => { setNaming(null); setNameDraft(""); }} className="rounded p-1 hover:bg-muted" aria-label="Cancel">
                      <X className="size-3" />
                    </button>
                  </span>
                );
              }
              return (
                <span key={cid} className={cn("flex items-center gap-1.5 rounded-full py-1 pl-2.5 pr-1.5 text-xs",
                       speakerColor(name).chip, voice === key && "ring-2 ring-primary ring-offset-1 ring-offset-card")}>
                  <button onClick={() => setVoice((v) => (v === key ? null : key))} className="flex items-center gap-1.5">
                    <span className={cn("size-1.5 rounded-full", speakerColor(name).dot)} />
                    {name}
                    <span className="opacity-60">{nClips} clip{nClips === 1 ? "" : "s"}</span>
                  </button>
                  <button onClick={() => { setNaming(cid); setNameDraft(""); }}
                          className="rounded-full p-0.5 opacity-50 hover:opacity-100" aria-label={`Name ${name}`}>
                    <Pencil className="size-3" />
                  </button>
                </span>
              );
            })}
          </div>
        )}

        {pending > 0 && (
          <p className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="size-3.5 animate-spin" />{pending} recording{pending === 1 ? "" : "s"} still processing…
          </p>
        )}

        <Tabs value={tab} onValueChange={setTab} className="gap-4">
          <TabsList>
            <TabsTrigger value="recordings">Recordings</TabsTrigger>
            <TabsTrigger value="analysis">Analysis</TabsTrigger>
          </TabsList>

          <TabsContent value="analysis">
            <AnalysisPanel utterances={utterances} clips={race.clips} voiceOf={voiceOf} onJump={jumpToClip} />
          </TabsContent>

          <TabsContent value="recordings" className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <div className="relative min-w-56 flex-1">
                <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
                <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search transcripts…"
                       className="h-8 pl-8 text-sm" />
              </div>
              <Select value={sentiment} onValueChange={setSentiment}>
                <SelectTrigger size="sm" className="w-36"><SelectValue placeholder="Sentiment" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Any sentiment</SelectItem>
                  <SelectItem value="negative">Negative</SelectItem>
                  <SelectItem value="neutral">Neutral</SelectItem>
                  <SelectItem value="positive">Positive</SelectItem>
                </SelectContent>
              </Select>
              <Select value={mood} onValueChange={setMood}>
                <SelectTrigger size="sm" className="w-32 capitalize"><SelectValue placeholder="Mood" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Any mood</SelectItem>
                  {MOODS.map((m) => <SelectItem key={m} value={m} className="capitalize">{m}</SelectItem>)}
                </SelectContent>
              </Select>
              <Select value={status} onValueChange={setStatus}>
                <SelectTrigger size="sm" className="w-36"><SelectValue placeholder="Status" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Any status</SelectItem>
                  <SelectItem value="complete">Processed</SelectItem>
                  <SelectItem value="processing">In flight</SelectItem>
                  <SelectItem value="review">Needs review</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {anyFilter && (
              <div className="flex flex-wrap items-center gap-2 rounded-lg border border-primary/30 bg-primary/5 px-3 py-2 text-xs">
                <span className="tabular-nums">
                  Showing {uttFiltersOn ? shown.length : utterances.length} of {utterances.length} utterances
                  {" · "}{visibleClips.length} of {race.clips.length} recordings
                </span>
                {voiceLabel && <Badge variant="outline" className="rounded-full">{voiceLabel}</Badge>}
                {q.trim() && <Badge variant="outline" className="rounded-full">“{q.trim()}”</Badge>}
                {sentiment !== "all" && <Badge variant="outline" className="rounded-full capitalize">{sentiment}</Badge>}
                {mood !== "all" && <Badge variant="outline" className="rounded-full capitalize">{mood}</Badge>}
                {status !== "all" && <Badge variant="outline" className="rounded-full capitalize">{status}</Badge>}
                <button onClick={clearAll} className="ml-auto flex items-center gap-1 text-primary hover:underline">
                  <X className="size-3" />clear all
                </button>
              </div>
            )}

            <div className="space-y-2">
          {visibleClips.map((c) => {
            const utts = shown.filter((u) => u.clip_id === c.id);
            return (
              <div key={c.id} id={`clip-${c.id}`}
                   className={cn("rounded-xl border bg-card px-4 py-3",
                     highlight === c.id ? "border-primary" : "border-border")}
                   onMouseEnter={() => highlight === c.id && setHighlight(null)}>
                <div className="flex items-center gap-3">
                  <span className={cn(
                    "size-1.5 shrink-0 rounded-full",
                    c.status === "COMPLETE" && "bg-success",
                    c.status === "REJECTED" && "bg-warning",
                    (c.status === "FAILED" || c.status === "DEAD") && "bg-destructive",
                    !TERMINAL.includes(c.status) && "animate-pulse bg-primary"
                  )} />
                  {/* the full clip view (audio player, quality metrics, per-speaker stats,
                      word-level seek, exports) already exists at /clips/:id — link to it
                      rather than growing a second, thinner copy of it here */}
                  <Link
                    to={`/clips/${c.id}`}
                    title="Open full clip analysis"
                    className="group/name flex min-w-0 flex-1 items-center gap-1.5 text-sm font-medium hover:text-primary"
                  >
                    <span className="truncate">{c.filename}</span>
                    <ArrowUpRight className="size-3.5 shrink-0 opacity-0 transition-opacity group-hover/name:opacity-100" />
                  </Link>
                  <span className="text-xs tabular-nums text-muted-foreground">{fmtDur(c.duration_s)}</span>
                  {c.n_speakers != null && c.n_speakers > 0 && (
                    <span className="text-xs tabular-nums text-muted-foreground">
                      {c.n_speakers} {c.n_speakers === 1 ? "voice" : "voices"}
                    </span>
                  )}
                  {c.language && <Badge variant="outline" className="rounded-full px-2 py-0 text-[10px] uppercase">{c.language}</Badge>}
                  {c.sentiment && (
                    <Badge variant="outline" className={cn("rounded-full px-2 py-0 text-[11px] capitalize", SENTIMENT_STYLE[c.sentiment])}>
                      {c.sentiment}{c.sentiment_score != null && <span className="ml-1 tabular-nums opacity-70">{c.sentiment_score > 0 ? "+" : ""}{c.sentiment_score.toFixed(2)}</span>}
                    </Badge>
                  )}
                  {!TERMINAL.includes(c.status) && <span className="text-xs text-muted-foreground">{c.status.toLowerCase()}…</span>}
                  {c.status === "REJECTED" && <span className="text-xs text-warning">no speech</span>}
                </div>

                {utts.length > 0 && (
                  <div className="mt-2.5 space-y-1 border-t border-border/60 pt-2.5">
                    {utts.map((u) => {
                      const who = u.display_name
                        || (u.cluster_id ? voiceNames.get(u.cluster_id) : null)
                        || "Unknown voice";
                      return (
                        <div key={u.id} className="flex items-baseline gap-2 text-sm">
                          <Badge variant="outline" className={cn("shrink-0 rounded-full px-1.5 py-0 text-[10px]",
                                 u.cluster_id || u.display_name ? speakerColor(who).chip : "text-muted-foreground")}>
                            {who}
                          </Badge>
                          <span className="flex-1">{u.text}</span>
                          {/* text and delivery shown separately: when they disagree, that
                              disagreement is the interesting signal, not noise to average away */}
                          {u.mood && u.mood !== "calm" && (
                            <span className="shrink-0 text-[10px] uppercase tracking-wide text-warning">{u.mood}</span>
                          )}
                          {u.sentiment && u.sentiment !== "neutral" && (
                            <span className={cn("shrink-0 text-[10px] tabular-nums",
                              u.sentiment === "positive" ? "text-success" : "text-destructive")}>
                              {u.sentiment_score != null && (u.sentiment_score > 0 ? "+" : "")}{u.sentiment_score?.toFixed(2)}
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}

                {utts.length === 0 && c.transcript && (
                  <p className="mt-2 border-t border-border/60 pt-2 text-sm text-foreground/90">{c.transcript}</p>
                )}
              </div>
            );
          })}
          {race.clips.length === 0 && (
            <p className="py-10 text-center text-sm text-muted-foreground">
              No recordings yet — drop some audio above to get started.
            </p>
          )}
          {race.clips.length > 0 && visibleClips.length === 0 && (
            <p className="py-10 text-center text-sm text-muted-foreground">
              No recordings match these filters.{" "}
              <button onClick={clearAll} className="text-primary hover:underline">Clear all</button>
            </p>
          )}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

export default function Races() {
  const [races, setRaces] = useState<Race[] | null>(null);
  const [creating, setCreating] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  const load = useCallback(() => {
    api.listRaces().then((r) => setRaces(r.items)).catch((e) => toast.error(String(e.message)));
  }, []);
  useEffect(load, [load]);

  if (selected) return <RaceDetail raceId={selected} onBack={() => { setSelected(null); load(); }} />;

  if (creating) {
    return (
      <div>
        <PageHeader title="New race" description="Name it, add a track outline, then drop in the recordings." />
        <CreateRaceForm
          onCancel={() => setCreating(false)}
          onCreated={(r) => { setCreating(false); load(); setSelected(r.id); }}
        />
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title="Races"
        description="Group team radio recordings by race. Each one is transcribed, speaker-tagged against enrolled profiles, and scored for sentiment."
        actions={<Button size="sm" onClick={() => setCreating(true)}><Plus className="mr-1.5 size-4" />New race</Button>}
      />
      <div className="px-8 py-6">
        {!races ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {[0, 1, 2].map((i) => <Skeleton key={i} className="h-40" />)}
          </div>
        ) : races.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-border py-20 text-center">
            <div className="flex size-12 items-center justify-center rounded-full bg-muted">
              <Flag className="size-6 text-muted-foreground" />
            </div>
            <div className="space-y-1">
              <p className="font-medium">No races yet</p>
              <p className="max-w-xs text-sm text-muted-foreground">
                Create a race, upload its track outline, then bulk-drop the team radio.
              </p>
            </div>
            <Button className="mt-2" onClick={() => setCreating(true)}><Plus className="mr-1.5 size-4" />New race</Button>
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {races.map((r) => (
              <button key={r.id} onClick={() => setSelected(r.id)}
                      className="group relative flex flex-col overflow-hidden rounded-xl border border-border bg-card text-left transition-colors hover:border-primary/40">
                <div className="flex h-28 shrink-0 items-center justify-center overflow-hidden bg-muted/30 p-3 text-muted-foreground/40">
                  {r.svg_path ? <TrackOutline raceId={r.id} /> : <Flag className="size-6" />}
                </div>
                <div className="flex-1 space-y-1 p-4">
                  <p className="font-medium leading-tight">{r.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {[r.circuit, r.race_date].filter(Boolean).join(" · ") || "—"}
                  </p>
                  <p className="pt-1 text-xs text-muted-foreground">
                    {r.n_clips ?? 0} recording{r.n_clips === 1 ? "" : "s"}
                    {(r.n_clips ?? 0) > 0 && ` · ${r.n_complete ?? 0} processed`}
                  </p>
                </div>
                <span
                  role="button"
                  tabIndex={0}
                  aria-label={`Delete ${r.name}`}
                  onClick={async (e) => {
                    e.stopPropagation();
                    // deliberately no confirm() dialog: the clips survive (race_id is set
                    // to NULL, they stay in the library), so this only unmakes the grouping
                    try { await api.deleteRace(r.id); toast.success("Race deleted — its recordings stay in the library"); load(); }
                    catch (err) { toast.error(String((err as Error).message)); }
                  }}
                  className="absolute right-2 top-2 rounded-md p-1.5 text-muted-foreground opacity-0 transition-opacity hover:bg-destructive/10 hover:text-destructive group-hover:opacity-100"
                >
                  <Trash2 className="size-3.5" />
                </span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
