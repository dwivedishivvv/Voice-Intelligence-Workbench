import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/api/client";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { AudioPlayer } from "@/components/audio-player";
import { cn } from "@/lib/utils";
import {
  UploadCloud, FileAudio, CheckCircle2, XCircle, Loader2, X,
} from "lucide-react";

const API_KEY = () => localStorage.getItem("api_key") || "change-me";

type StageEvent = {
  type: string; stage?: string; state?: string; ms?: number; code?: string; message?: string;
  progress?: number; elapsed_ms?: number; total_ms?: number;
};

const STAGES = [
  "VALIDATING", "PREPROCESSING", "TRANSCRIBING", "DIARIZING", "RECONCILING",
  "EMBEDDING", "IDENTIFYING", "POSTPROCESSING", "INDEXING",
];

function fmtEta(ms: number) {
  const s = Math.ceil(ms / 1000);
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
}

export default function Upload() {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [events, setEvents] = useState<StageEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [clipId, setClipId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  // object URL so the raw upload is audible immediately, before any processing happens
  useEffect(() => {
    if (!file) { setPreviewUrl(null); return; }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setFile(f);
  }, []);

  async function submit() {
    if (!file) return;
    setBusy(true);
    setEvents([]);
    setClipId(null);
    try {
      const res = await api.uploadClip(file);
      setClipId(res.clip_id);
      if (res.duplicate) {
        toast.info("This clip was already processed — jumping to its results.");
        navigate(`/clips/${res.clip_id}`);
        return;
      }
      const ws = new WebSocket(`${location.origin.replace("http", "ws")}${res.ws_url}`);
      ws.onmessage = (e) => {
        // worker always emits {type:"stage", stage, state, ...} — terminal outcomes are
        // distinguished by stage/state, not a dedicated "complete"/"rejected" type.
        const msg = JSON.parse(e.data) as StageEvent;
        setEvents((prev) => [...prev, msg]);
        const done = msg.stage === "COMPLETE" || msg.stage === "FAILED" || msg.state === "rejected";
        if (done) {
          ws.close();
          setBusy(false);
          if (msg.stage === "COMPLETE") {
            toast.success("Processing complete", {
              description: msg.total_ms ? `Finished in ${(msg.total_ms / 1000).toFixed(1)}s` : undefined,
            });
            navigate(`/clips/${res.clip_id}`);
          } else if (msg.state === "rejected") {
            toast.error(`Rejected: ${msg.code}`, { description: msg.message });
          } else {
            toast.error("Processing failed", { description: msg.message });
          }
        }
      };
      ws.onerror = () => { setBusy(false); toast.error("Lost connection to the processing job."); };
    } catch (e) {
      setBusy(false);
      toast.error("Upload failed", { description: String(e) });
    }
  }

  const stageEvents = events.filter((e) => e.type === "stage");
  const stageStates = new Map(stageEvents.map((e) => [e.stage, e.state]));
  const currentStageIdx = STAGES.findIndex((s) => stageStates.get(s) === "started");
  const lastDone = [...stageEvents].reverse().find((e) => e.state === "done" && e.progress != null && e.elapsed_ms != null);
  const progress = lastDone?.progress ?? (stageEvents.length ? 1 : 0);
  // linear ETA from the rate observed so far — a rough estimate is far more useful than none
  const etaMs = lastDone && lastDone.progress! > 0 && lastDone.progress! < 100
    ? (lastDone.elapsed_ms! / lastDone.progress!) * (100 - lastDone.progress!)
    : null;

  return (
    <div>
      <PageHeader title="Upload a clip" description="Audio or video, up to 90 seconds. Processed fully offline." />

      <div className="mx-auto max-w-2xl px-8 py-8">
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => !busy && inputRef.current?.click()}
          className={cn(
            "flex cursor-pointer flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed p-14 text-center transition-all",
            dragging ? "border-primary bg-primary/5 scale-[1.01]" : "border-border hover:border-primary/40 hover:bg-white/[0.02]",
            busy && "pointer-events-none opacity-60"
          )}
        >
          <input
            ref={inputRef}
            type="file"
            accept="audio/*,video/*"
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          {file ? (
            <>
              <div className="flex size-14 items-center justify-center rounded-full bg-primary/10">
                <FileAudio className="size-7 text-primary" />
              </div>
              <div>
                <p className="font-medium">{file.name}</p>
                <p className="text-sm text-muted-foreground">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
              </div>
              {!busy && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="gap-1.5 text-muted-foreground"
                  onClick={(e) => { e.stopPropagation(); setFile(null); }}
                >
                  <X className="size-3.5" /> Remove
                </Button>
              )}
            </>
          ) : (
            <>
              <div className="flex size-14 items-center justify-center rounded-full bg-muted">
                <UploadCloud className="size-7 text-muted-foreground" />
              </div>
              <div>
                <p className="font-medium">Drop an audio or video file here</p>
                <p className="text-sm text-muted-foreground">or click to browse</p>
              </div>
            </>
          )}
        </div>

        {previewUrl && !busy && events.length === 0 && (
          <div className="mt-4 animate-slide-up">
            <p className="mb-1.5 text-xs font-medium text-muted-foreground">Preview before uploading</p>
            <AudioPlayer src={previewUrl} />
          </div>
        )}

        <Button onClick={submit} disabled={!file || busy} className="mt-6 w-full gap-2" size="lg">
          {busy && <Loader2 className="size-4 animate-spin" />}
          {busy ? "Processing…" : "Upload & process"}
        </Button>

        {events.length > 0 && (
          <div className="mt-8 space-y-6 animate-slide-up">
            <div className="space-y-2">
              <div className="flex items-baseline justify-between">
                <span className="text-3xl font-semibold tabular-nums tracking-tight">{Math.round(progress)}%</span>
                {etaMs != null && busy && (
                  <span className="text-sm text-muted-foreground">~{fmtEta(etaMs)} remaining</span>
                )}
              </div>
              <Progress value={progress} className="h-2" />
            </div>

            <div className="space-y-1">
              {STAGES.map((stage, i) => {
                const state = stageStates.get(stage);
                const ev = events.find((e) => e.stage === stage && e.state === "done");
                const done = state === "done";
                const active = i === currentStageIdx;
                const rejected = events.some((e) => e.stage === stage && e.state === "rejected");
                return (
                  <div
                    key={stage}
                    className={cn(
                      "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
                      active && "bg-primary/5"
                    )}
                  >
                    {rejected ? (
                      <XCircle className="size-4 shrink-0 text-destructive" />
                    ) : done ? (
                      <CheckCircle2 className="size-4 shrink-0 text-success" />
                    ) : active ? (
                      <Loader2 className="size-4 shrink-0 animate-spin text-primary" />
                    ) : (
                      <div className="size-4 shrink-0 rounded-full border-2 border-border" />
                    )}
                    <span className={cn(
                      "flex-1",
                      done ? "text-foreground" : active ? "font-medium text-foreground" : "text-muted-foreground"
                    )}>
                      {stage.charAt(0) + stage.slice(1).toLowerCase()}
                    </span>
                    {ev?.ms != null && <span className="tabular-nums text-xs text-muted-foreground">{ev.ms}ms</span>}
                  </div>
                );
              })}
            </div>

            {clipId && stageStates.get("PREPROCESSING") === "done" && (
              <div className="animate-slide-up">
                <p className="mb-1.5 text-xs font-medium text-muted-foreground">
                  Processed audio (denoised · normalized) — what the pipeline actually hears from here on
                </p>
                <AudioPlayer src={`/v1/clips/${clipId}/audio?variant=work&key=${API_KEY()}`} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
