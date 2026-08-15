import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/api/client";
import { StatusBadge } from "@/components/status-badge";
import { AudioPlayer } from "@/components/audio-player";
import { speakerColor } from "@/lib/speaker-color";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  ArrowLeft, Download, RefreshCw, AlertTriangle, Pencil, Check, X, ChevronDown,
} from "lucide-react";

const API_KEY = () => localStorage.getItem("api_key") || "change-me";
const EXPORT_FORMATS = ["srt", "vtt", "rttm", "json", "txt"];

const SENTIMENT_STYLE: Record<string, string> = {
  positive: "bg-success/15 text-success border-success/30",
  negative: "bg-destructive/15 text-destructive border-destructive/30",
  neutral: "bg-muted text-muted-foreground border-border",
};

export default function ClipDetail() {
  const { id } = useParams();
  const [result, setResult] = useState<any>(null);
  const [assigning, setAssigning] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [reprocessing, setReprocessing] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [variant, setVariant] = useState<"original" | "work">("work");
  // {nonce} forces the AudioPlayer's seek effect to re-fire even when clicking the same
  // word/utterance twice in a row, which a plain `number | null` state wouldn't (React
  // skips effects whose dependency didn't change value).
  const [seekTo, setSeekTo] = useState<{ time: number; nonce: number } | null>(null);
  const seek = (time: number) => setSeekTo((prev) => ({ time, nonce: (prev?.nonce ?? 0) + 1 }));
  const activeRowRef = useRef<HTMLDivElement>(null);
  const [searchParams] = useSearchParams();

  async function load() {
    if (id) setResult(await api.getResult(id));
  }
  useEffect(() => { load(); }, [id]);

  // ?t=<seconds> deep link. An agent answer cites a moment, not a file, so following a
  // citation has to land on the audio at that moment -- otherwise "verifiable" means
  // scrubbing a 90-second clip by hand. Runs once the result is loaded, because the
  // player does not exist before then.
  const seekedRef = useRef(false);
  useEffect(() => {
    const t = Number(searchParams.get("t"));
    if (!result || seekedRef.current || !Number.isFinite(t) || t <= 0) return;
    seekedRef.current = true;
    seek(t);
  }, [result, searchParams]);

  const wordsByUtterance = useMemo(() => {
    const map = new Map<string, any[]>();
    for (const w of result?.words || []) {
      if (!map.has(w.utterance_id)) map.set(w.utterance_id, []);
      map.get(w.utterance_id)!.push(w);
    }
    return map;
  }, [result]);

  // Sentiment per speaker, for a recording with more than one voice. The pipeline scores
  // every utterance individually and each utterance carries its speaker, so this is a
  // grouping, not a second analysis — one clip-level number would average a driver's
  // frustration together with their engineer's calm and report neither.
  // Weighted by utterance duration: a long angry sentence should outweigh a clipped "ok".
  const sentimentBySpeaker = useMemo(() => {
    const acc = new Map<string, {
      sum: number; weight: number; counts: Record<string, number>;
      moods: Record<string, number>; n: number;
      top: { text: string; score: number } | null;
      bottom: { text: string; score: number } | null;
    }>();
    for (const u of result?.utterances || []) {
      if (!u.local_label) continue;
      const e = acc.get(u.local_label) ?? {
        sum: 0, weight: 0, counts: { negative: 0, neutral: 0, positive: 0 },
        moods: { calm: 0, stressed: 0, tired: 0 }, n: 0, top: null, bottom: null,
      };
      if (u.sentiment_score != null) {
        const w = Math.max((u.end_s ?? 0) - (u.start_s ?? 0), 0.1);
        e.sum += u.sentiment_score * w;
        e.weight += w;
        e.n += 1;
        if (u.sentiment) e.counts[u.sentiment] = (e.counts[u.sentiment] ?? 0) + 1;
        if (!e.top || u.sentiment_score > e.top.score) e.top = { text: u.text, score: u.sentiment_score };
        if (!e.bottom || u.sentiment_score < e.bottom.score) e.bottom = { text: u.text, score: u.sentiment_score };
      }
      if (u.mood) e.moods[u.mood] = (e.moods[u.mood] ?? 0) + 1;
      acc.set(u.local_label, e);
    }
    return acc;
  }, [result]);

  const activeUtteranceId = useMemo(() => {
    for (const u of result?.utterances || []) {
      if (currentTime >= u.start_s && currentTime < u.end_s) return u.id;
    }
    return null;
  }, [result, currentTime]);

  useEffect(() => {
    activeRowRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [activeUtteranceId]);

  if (!result) {
    return (
      <div className="space-y-4 p-8">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-16 w-full rounded-xl" />
        <Skeleton className="h-40 w-full rounded-xl" />
      </div>
    );
  }

  const { clip, quality, speakers, utterances, turns, run } = result;
  const warnings: any[] = run?.warnings || [];
  const displayName = (label: string) =>
    speakers?.find((s: any) => s.local_label === label)?.display_name || label;

  async function assign(label: string, body: { enroll?: boolean; [k: string]: unknown }) {
    if (!id) return;
    const res = await api.assignSpeaker(id, label, body);
    setAssigning(null);
    setName("");
    if (body.enroll && !res.enrolled) {
      toast.warning("Speaker labeled, but not enrolled", {
        description: res.enroll_skip_reason || "This clip's audio wasn't usable for identification.",
      });
    } else {
      toast.success("Speaker updated");
    }
    load();
  }

  async function reprocess() {
    if (!id) return;
    setReprocessing(true);
    try {
      await api.reprocess(id);
      toast.success("Reprocessing queued");
      setTimeout(load, 1500);
    } finally {
      setReprocessing(false);
    }
  }

  return (
    <div>
      <div className="flex items-start justify-between border-b border-border px-8 py-6">
        <div className="space-y-2">
          <Button variant="ghost" size="sm" asChild className="-ml-2 h-7 gap-1 text-muted-foreground">
            <Link to="/"><ArrowLeft className="size-3.5" /> Library</Link>
          </Button>
          <h1 className="text-2xl font-semibold tracking-tight">{clip.filename}</h1>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <span className="tabular-nums">{clip.duration_s?.toFixed(1)}s</span>
            <Separator />
            <span className="uppercase">{clip.language || "—"}</span>
            <Separator />
            <StatusBadge status={clip.status} />
            {quality && (
              <>
                <Separator />
                <Badge variant="outline" className={
                  quality.grade === "good" ? "border-success/30 bg-success/10 text-success"
                    : quality.grade === "fair" ? "border-warning/30 bg-warning/10 text-warning"
                      : "border-destructive/30 bg-destructive/10 text-destructive"
                }>
                  {quality.grade} quality
                </Badge>
              </>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" className="gap-1.5" onClick={reprocess} disabled={reprocessing}>
            <RefreshCw className={`size-3.5 ${reprocessing ? "animate-spin" : ""}`} />
            Reprocess
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" className="gap-1.5">
                <Download className="size-3.5" /> Export <ChevronDown className="size-3" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {EXPORT_FORMATS.map((fmt) => (
                <DropdownMenuItem key={fmt} asChild>
                  <a href={`/v1/clips/${id}/export/${fmt}?key=${API_KEY()}`} className="uppercase">
                    {fmt}
                  </a>
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      <div className="mx-auto max-w-4xl space-y-6 px-8 py-6">
        {clip.work_path && (
          <div className="flex items-center gap-1 rounded-lg bg-muted p-1 text-xs">
            {(["original", "work"] as const).map((v) => (
              <button
                key={v}
                onClick={() => setVariant(v)}
                className={cn(
                  "rounded-md px-3 py-1.5 font-medium transition-colors",
                  variant === v ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground"
                )}
              >
                {v === "original" ? "Original upload" : "Processed (denoised · normalized)"}
              </button>
            ))}
          </div>
        )}
        <AudioPlayer
          key={variant}
          src={`/v1/clips/${id}/audio?variant=${variant}&key=${API_KEY()}`}
          duration={clip.duration_s || 0}
          turns={turns || []}
          speakers={speakers || []}
          onTimeUpdate={setCurrentTime}
          seekTo={seekTo}
        />

        {warnings.map((w, i) => (
          <Alert key={i} variant="destructive" className="border-warning/30 bg-warning/10 text-warning [&>svg]:text-warning">
            <AlertTriangle className="size-4" />
            <AlertTitle>{w.code.replaceAll("_", " ")}</AlertTitle>
            {Object.keys(w).length > 1 && (
              <AlertDescription className="text-warning/80">
                {Object.entries(w).filter(([k]) => k !== "code").map(([k, v]) => `${k}: ${v}`).join(" · ")}
              </AlertDescription>
            )}
          </Alert>
        ))}

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Speakers</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {(speakers || []).map((s: any) => (
              <div key={s.local_label} className="flex items-center gap-4 rounded-lg border border-border p-3">
                <span className={cn("size-2.5 shrink-0 rounded-full", speakerColor(s.local_label).dot)} />
                <div className="min-w-0 flex-1 space-y-1.5">
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{s.display_name || s.local_label}</span>
                    <Badge variant="secondary" className="text-xs">{s.match_result}</Badge>
                    {s.match_score != null && (
                      <span className="text-xs text-muted-foreground tabular-nums">score {s.match_score.toFixed(2)}</span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <Progress value={(s.talk_share || 0) * 100} className="h-1.5 w-40" />
                    <span className="text-xs tabular-nums text-muted-foreground">{((s.talk_share || 0) * 100).toFixed(0)}% talk</span>
                    <span className="text-xs tabular-nums text-muted-foreground">· reliability {(s.reliability || 0).toFixed(2)}</span>
                  </div>
                  {(() => {
                    const sen = sentimentBySpeaker.get(s.local_label);
                    if (!sen || !sen.weight) return null;
                    const avg = sen.sum / sen.weight;
                    const label = avg > 0.2 ? "positive" : avg < -0.2 ? "negative" : "neutral";
                    const total = sen.counts.negative + sen.counts.neutral + sen.counts.positive || 1;
                    const stressed = sen.moods.stressed || 0;
                    return (
                      <div className="space-y-1.5 pt-1">
                        <div className="flex items-center gap-2">
                          <Badge variant="outline" className={cn("rounded-full px-2 py-0 text-[11px] capitalize", SENTIMENT_STYLE[label])}>
                            {label} <span className="ml-1 tabular-nums opacity-70">{avg > 0 ? "+" : ""}{avg.toFixed(2)}</span>
                          </Badge>
                          <span className="text-xs text-muted-foreground">
                            over {sen.n} utterance{sen.n === 1 ? "" : "s"}
                          </span>
                          {stressed > 0 && (
                            <span className="text-[10px] uppercase tracking-wide text-warning">
                              {stressed} stressed
                            </span>
                          )}
                        </div>
                        <div className="flex h-1.5 w-56 overflow-hidden rounded-full">
                          <div className="bg-destructive" style={{ width: `${(sen.counts.negative / total) * 100}%` }} />
                          <div className="bg-muted-foreground/40" style={{ width: `${(sen.counts.neutral / total) * 100}%` }} />
                          <div className="bg-success" style={{ width: `${(sen.counts.positive / total) * 100}%` }} />
                        </div>
                        {/* the single strongest line each way says more about how this person
                            sounded than the average does, and it's evidence for the number */}
                        {sen.bottom && sen.bottom.score <= -0.2 && (
                          <p className="truncate text-xs text-destructive/90" title={sen.bottom.text}>
                            ↓ “{sen.bottom.text}”
                          </p>
                        )}
                        {sen.top && sen.top.score >= 0.2 && (
                          <p className="truncate text-xs text-success/90" title={sen.top.text}>
                            ↑ “{sen.top.text}”
                          </p>
                        )}
                      </div>
                    );
                  })()}
                </div>

                {assigning === s.local_label ? (
                  <div className="flex items-center gap-1.5">
                    <Input
                      autoFocus
                      placeholder="Speaker name"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      className="h-8 w-40"
                    />
                    <Button size="icon" className="size-8" onClick={() => assign(s.local_label, { create_profile: { display_name: name }, enroll: true })}>
                      <Check className="size-3.5" />
                    </Button>
                    <Button size="icon" variant="outline" className="size-8" onClick={() => assign(s.local_label, { mark_unknown: true })}>
                      <X className="size-3.5" />
                    </Button>
                    <Button size="icon" variant="ghost" className="size-8" onClick={() => setAssigning(null)}>
                      <X className="size-3.5 text-muted-foreground" />
                    </Button>
                  </div>
                ) : (
                  <Button variant="ghost" size="sm" className="gap-1.5 text-muted-foreground" onClick={() => setAssigning(s.local_label)}>
                    <Pencil className="size-3.5" /> Correct
                  </Button>
                )}
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Transcript</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            {(utterances || []).map((u: any) => {
              const active = u.id === activeUtteranceId;
              const words = wordsByUtterance.get(u.id);
              return (
                <div
                  key={u.id}
                  ref={active ? activeRowRef : undefined}
                  onClick={() => seek(u.start_s)}
                  className={cn(
                    "flex cursor-pointer gap-3 rounded-lg border border-transparent px-2 py-2 transition-colors hover:bg-white/[0.03]",
                    active && "border-primary/30 bg-primary/5"
                  )}
                >
                  <span className={cn(
                    "mt-0.5 h-fit shrink-0 rounded-full px-2 py-0.5 text-xs font-medium",
                    speakerColor(u.local_label).chip
                  )}>
                    {displayName(u.local_label)}
                  </span>
                  {/* text sentiment and acoustic mood side by side, never merged into one
                      number: "calm words, stressed delivery" is the interesting case and
                      averaging the two would hide exactly that */}
                  {(u.sentiment || u.mood) && (
                    <span className="mt-1 flex h-fit shrink-0 items-center gap-1.5">
                      {u.sentiment && u.sentiment !== "neutral" && (
                        <span className={cn("text-[10px] tabular-nums",
                          u.sentiment === "positive" ? "text-success" : "text-destructive")}>
                          {u.sentiment_score > 0 ? "+" : ""}{u.sentiment_score?.toFixed(2)}
                        </span>
                      )}
                      {u.mood && u.mood !== "calm" && (
                        <span className="text-[10px] uppercase tracking-wide text-warning">{u.mood}</span>
                      )}
                    </span>
                  )}
                  <p
                    className="flex-1 leading-relaxed"
                    title={`word conf ${u.mean_word_conf?.toFixed(2)} · speaker conf ${u.mean_speaker_conf?.toFixed(2)}`
                      + (u.text_sentiment ? ` · text ${u.text_sentiment} ${u.text_score?.toFixed(2)}` : "")
                      + (u.mood ? ` · voice ${u.mood}` : "")}
                  >
                    {words ? words.map((w: any, i: number) => {
                      const wordActive = active && currentTime >= w.start_s && currentTime < w.end_s;
                      const spoken = active && currentTime >= w.end_s;
                      return (
                        <span
                          key={i}
                          onClick={(e) => { e.stopPropagation(); seek(w.start_s); }}
                          className={cn(
                            "rounded transition-colors",
                            wordActive && "bg-primary/30 text-foreground",
                            spoken && !wordActive && "text-foreground/70",
                            !active && "text-foreground"
                          )}
                        >
                          {w.word}{" "}
                        </span>
                      );
                    }) : u.text}
                  </p>
                </div>
              );
            })}
            {(!utterances || utterances.length === 0) && (
              <p className="py-4 text-center text-sm text-muted-foreground">No transcript available.</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function Separator() {
  return <span className="text-border">·</span>;
}
