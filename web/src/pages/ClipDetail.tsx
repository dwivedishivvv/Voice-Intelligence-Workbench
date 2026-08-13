import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
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

  async function load() {
    if (id) setResult(await api.getResult(id));
  }
  useEffect(() => { load(); }, [id]);

  const wordsByUtterance = useMemo(() => {
    const map = new Map<string, any[]>();
    for (const w of result?.words || []) {
      if (!map.has(w.utterance_id)) map.set(w.utterance_id, []);
      map.get(w.utterance_id)!.push(w);
    }
    return map;
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
                  <p
                    className="leading-relaxed"
                    title={`word conf ${u.mean_word_conf?.toFixed(2)} · speaker conf ${u.mean_speaker_conf?.toFixed(2)}`}
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
