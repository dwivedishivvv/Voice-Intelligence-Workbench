import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "@/api/client";
import { PageHeader } from "@/components/page-header";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { type Mood, MOOD_STYLE, moodDot } from "@/lib/mood";
import { speakerColor } from "@/lib/speaker-color";
import { Mic, Square, Loader2, AudioLines, Users } from "lucide-react";

// VAD-driven chunk boundaries: cut on a natural pause instead of an arbitrary timer, so
// chunks end at sentence/phrase breaks rather than mid-word — the single biggest source
// of garbled live text. Naive fixed-threshold energy VAD (not adaptive to noise floor or
// mic gain) — good enough to find pauses in a normal room; may need retuning on a very
// noisy input or a hot mic.
const SPEECH_RMS_THRESHOLD = 0.02;
const PAUSE_MS = 600; // silence this long after speech = natural end of phrase, cut here
const MIN_CHUNK_MS = 1200; // don't cut on a micro-pause before this much has been said
const MAX_CHUNK_MS = 12000; // hard cap so continuous talk without a pause can't grow forever
const LEVEL_NORMALIZER = 0.12; // rms value that reads as "full" on the level ring

type ChunkStatus = "pending" | "done" | "error";
type Segment = { label: string; is_known: boolean; start: number; end: number; text: string };
type Chunk = { seq: number; text: string; segments: Segment[]; status: ChunkStatus; mood?: Mood; t: number };
type VadState = { start: number; hasSpeech: boolean; silenceSince: number | null };

function fmtElapsed(ms: number) {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, "0")}`;
}

export default function Live() {
  const [recording, setRecording] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [elapsedMs, setElapsedMs] = useState(0);
  const sessionIdRef = useRef<string | null>(null);
  const sessionStartRef = useRef(0);
  const seqRef = useRef(0);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const stopRequestedRef = useRef(false);
  const elapsedTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const vadBufRef = useRef<Uint8Array<ArrayBuffer> | null>(null);
  const vadStateRef = useRef<VadState>({ start: 0, hasSpeech: false, silenceSince: null });
  const rafRef = useRef<number | null>(null);
  const levelRingRef = useRef<HTMLSpanElement>(null);

  useEffect(() => () => stop(), []); // eslint-disable-line react-hooks/exhaustive-deps

  function vadTick() {
    const analyser = analyserRef.current, buf = vadBufRef.current;
    if (stopRequestedRef.current || !analyser || !buf) return;

    analyser.getByteTimeDomainData(buf);
    let sumSquares = 0;
    for (let i = 0; i < buf.length; i++) {
      const v = (buf[i] - 128) / 128;
      sumSquares += v * v;
    }
    const rms = Math.sqrt(sumSquares / buf.length);

    // drive the level ring directly via the DOM, not React state — this runs every
    // animation frame and a state-driven re-render at 60fps would be wasteful
    if (levelRingRef.current) {
      const level = Math.min(1, rms / LEVEL_NORMALIZER);
      levelRingRef.current.style.transform = `scale(${1 + level * 0.5})`;
      levelRingRef.current.style.opacity = String(0.12 + level * 0.5);
    }

    const state = vadStateRef.current;
    const now = performance.now();
    const isSpeech = rms > SPEECH_RMS_THRESHOLD;
    if (isSpeech) {
      state.hasSpeech = true;
      state.silenceSince = null;
    } else if (state.silenceSince == null) {
      state.silenceSince = now;
    }
    setSpeaking((prev) => (prev !== isSpeech ? isSpeech : prev));

    const elapsed = now - state.start;
    const pausedLongEnough = state.hasSpeech && state.silenceSince != null && (now - state.silenceSince) >= PAUSE_MS;
    const shouldCut = (elapsed >= MIN_CHUNK_MS && pausedLongEnough) || elapsed >= MAX_CHUNK_MS;

    if (shouldCut && recorderRef.current?.state === "recording") {
      recorderRef.current.stop();
    }
    rafRef.current = requestAnimationFrame(vadTick);
  }

  // MediaRecorder chunks from a single continuous recording aren't independently
  // decodable (only the first has a full container header) — so each VAD-bounded
  // window is its own start/stop recorder instance, each a self-contained webm file.
  function recordOneChunk(stream: MediaStream) {
    if (stopRequestedRef.current) return;
    const recorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
    recorderRef.current = recorder;
    vadStateRef.current = { start: performance.now(), hasSpeech: false, silenceSince: null };
    const seq = seqRef.current++;
    const parts: BlobPart[] = [];

    recorder.ondataavailable = (e) => { if (e.data.size > 0) parts.push(e.data); };
    recorder.onstop = async () => {
      const hadSpeech = vadStateRef.current.hasSpeech;
      if (hadSpeech) {
        const blob = new Blob(parts, { type: "audio/webm" });
        setChunks((prev) => [...prev, { seq, text: "", segments: [], status: "pending", t: performance.now() - sessionStartRef.current }]);
        try {
          await api.uploadLiveChunk(sessionIdRef.current!, seq, blob);
        } catch {
          setChunks((prev) => prev.map((c) => (c.seq === seq ? { ...c, status: "error" } : c)));
        }
      }
      if (!stopRequestedRef.current && streamRef.current) recordOneChunk(streamRef.current);
    };
    recorder.start();
  }

  async function start() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;
    stopRequestedRef.current = false;
    seqRef.current = 0;
    sessionStartRef.current = performance.now();
    setChunks([]);
    setElapsedMs(0);

    const audioCtx = new AudioContext();
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    audioCtx.createMediaStreamSource(stream).connect(analyser);
    audioCtxRef.current = audioCtx;
    analyserRef.current = analyser;
    vadBufRef.current = new Uint8Array(new ArrayBuffer(analyser.fftSize));

    const sessionId = crypto.randomUUID();
    sessionIdRef.current = sessionId;

    const ws = new WebSocket(`${location.origin.replace("http", "ws")}/v1/ws/jobs/${sessionId}`);
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data) as {
        type: string; seq: number; text: string; mood?: Mood; segments?: Segment[]; error?: string;
      };
      if (msg.type !== "live_transcript") return;
      setChunks((prev) => prev.map((c) =>
        c.seq === msg.seq
          ? { ...c, text: msg.text, mood: msg.mood, segments: msg.segments || [], status: msg.error ? "error" : "done" }
          : c));
    };
    wsRef.current = ws;

    setRecording(true);
    recordOneChunk(stream);
    rafRef.current = requestAnimationFrame(vadTick);
    elapsedTimerRef.current = setInterval(
      () => setElapsedMs(performance.now() - sessionStartRef.current), 1000);
  }

  function stop() {
    stopRequestedRef.current = true;
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    wsRef.current?.close();
    if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
    if (elapsedTimerRef.current != null) clearInterval(elapsedTimerRef.current);
    audioCtxRef.current?.close();
    audioCtxRef.current = null;
    analyserRef.current = null;
    setRecording(false);
    setSpeaking(false);
  }

  const transcript = chunks.map((c) => c.text).filter(Boolean).join(" ");
  const pendingCount = chunks.filter((c) => c.status === "pending").length;
  const doneChunks = chunks.filter((c) => c.status === "done" && c.mood);
  const currentMood = doneChunks.length ? doneChunks[doneChunks.length - 1].mood : undefined;
  const wordCount = transcript ? transcript.trim().split(/\s+/).length : 0;

  // distinct speakers seen this session, in first-appearance order — diarization runs
  // per chunk (see worker/worker/live.py), so this is what stitches those chunk-scoped
  // reads into "who's been talking" for the whole session
  const speakers = useMemo(() => {
    const seen = new Map<string, boolean>();
    for (const c of chunks) for (const s of c.segments) if (!seen.has(s.label)) seen.set(s.label, s.is_known);
    return Array.from(seen, ([label, isKnown]) => ({ label, isKnown }));
  }, [chunks]);

  return (
    <div>
      <PageHeader
        title="Live transcription"
        description="Speak into your mic — each chunk cuts on your next pause, not a fixed timer, so words don't get split mid-sentence. Speakers are identified against enrolled profiles where possible, otherwise labeled for this session only. Full audio quality analysis still needs the recording uploaded for the full pipeline."
      />

      <div className="mx-auto max-w-2xl px-8 py-10">
        <div className="flex flex-col items-center">
          <div className="relative flex size-28 items-center justify-center">
            {recording && (
              <span
                ref={levelRingRef}
                className={cn(
                  "absolute inset-0 rounded-full transition-colors duration-300",
                  currentMood ? moodDot(currentMood) : "bg-primary"
                )}
                style={{ opacity: 0.12 }}
              />
            )}
            <Button
              onClick={recording ? stop : start}
              variant={recording ? "destructive" : "default"}
              className={cn(
                "relative z-10 size-20 rounded-full shadow-lg transition-transform hover:scale-105 active:scale-95",
                recording ? "shadow-destructive/20" : "shadow-primary/20"
              )}
            >
              {recording ? <Square className="size-6 fill-current" /> : <Mic className="size-7" />}
            </Button>
          </div>

          <div className="mt-4 flex items-center gap-2 text-sm">
            {recording ? (
              <span className="flex items-center gap-2 text-muted-foreground">
                <span className="relative flex size-2">
                  <span className={cn(
                    "absolute inline-flex h-full w-full animate-ping rounded-full opacity-75",
                    speaking ? "bg-primary" : "bg-destructive"
                  )} />
                  <span className={cn("relative inline-flex size-2 rounded-full", speaking ? "bg-primary" : "bg-destructive")} />
                </span>
                {pendingCount > 0 && <Loader2 className="size-3.5 animate-spin" />}
                {pendingCount > 0
                  ? `transcribing chunk ${chunks.length - pendingCount + 1}…`
                  : speaking ? "hearing you…" : "listening for a pause…"}
              </span>
            ) : (
              <span className="text-muted-foreground">Tap to start listening</span>
            )}
            {currentMood && (
              <Badge variant="outline" className={cn("rounded-full px-3 capitalize", MOOD_STYLE[currentMood])}>
                {currentMood}
              </Badge>
            )}
          </div>

          {(recording || chunks.length > 0) && (
            <div className="mt-5 flex items-center gap-5 text-xs tabular-nums text-muted-foreground">
              <span>{fmtElapsed(elapsedMs)}</span>
              <span className="h-3 w-px bg-border" />
              <span>{chunks.length} {chunks.length === 1 ? "chunk" : "chunks"}</span>
              <span className="h-3 w-px bg-border" />
              <span>{wordCount} {wordCount === 1 ? "word" : "words"}</span>
            </div>
          )}
        </div>

        {chunks.length === 0 && !recording ? (
          <div className="mt-10 flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-border py-16 text-center animate-fade-in">
            <div className="flex size-12 items-center justify-center rounded-full bg-muted">
              <AudioLines className="size-6 text-muted-foreground" />
            </div>
            <div className="space-y-1">
              <p className="font-medium">Nothing transcribed yet</p>
              <p className="max-w-xs text-sm text-muted-foreground">
                Start listening and speak naturally — text and a stress read appear after each pause.
              </p>
            </div>
          </div>
        ) : (
          <>
            <div className="mt-8 animate-slide-up rounded-xl border border-border bg-card p-5">
              {doneChunks.length > 0 && (
                <div className="mb-4 flex items-center gap-1">
                  {chunks.map((c) => (
                    <span
                      key={c.seq}
                      title={c.mood ? `chunk ${c.seq}: ${c.mood}` : `chunk ${c.seq}`}
                      className={cn("h-1.5 flex-1 rounded-full transition-colors", c.mood ? moodDot(c.mood) : "bg-muted")}
                    />
                  ))}
                </div>
              )}

              <p className="mb-2 text-xs font-medium text-muted-foreground">Live transcript</p>
              <p className="whitespace-pre-wrap text-lg leading-relaxed">
                {chunks.some((c) => c.segments.length > 0) ? (
                  chunks.map((c) =>
                    c.segments.length > 0 ? (
                      c.segments.map((s, i) => (
                        <span key={`${c.seq}-${i}`} className={cn("rounded px-0.5", speakerColor(s.label).chip)}>
                          {s.text}{" "}
                        </span>
                      ))
                    ) : c.text ? (
                      <span key={c.seq}>{c.text} </span>
                    ) : null
                  )
                ) : (
                  transcript || <span className="text-muted-foreground">Listening…</span>
                )}
                {pendingCount > 0 && <span className="animate-pulse text-muted-foreground"> …</span>}
              </p>
            </div>

            {speakers.length > 0 && (
              <div className="mt-4 flex flex-wrap items-center gap-3 rounded-xl border border-border bg-card px-4 py-3 animate-slide-up">
                <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                  <Users className="size-3.5" /> Speakers
                </span>
                {speakers.map((s) => (
                  <span key={s.label} className={cn("flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs", speakerColor(s.label).chip)}>
                    <span className={cn("size-1.5 rounded-full", speakerColor(s.label).dot)} />
                    {s.label}
                    {!s.isKnown && <span className="text-[10px] opacity-70">(this session)</span>}
                  </span>
                ))}
              </div>
            )}

            {chunks.length > 0 && (
              <div className="mt-4 animate-slide-up space-y-1.5">
                {[...chunks].reverse().map((c) => (
                  <div
                    key={c.seq}
                    className="flex items-start gap-3 rounded-lg border border-transparent px-3 py-2 text-sm transition-colors hover:border-border hover:bg-white/[0.02]"
                  >
                    <span className="mt-1 w-10 shrink-0 tabular-nums text-xs text-muted-foreground">
                      {fmtElapsed(c.t)}
                    </span>
                    <span className={cn(
                      "mt-1.5 size-1.5 shrink-0 rounded-full",
                      c.status === "pending" && "bg-warning animate-pulse",
                      c.status === "done" && "bg-success",
                      c.status === "error" && "bg-destructive"
                    )} />
                    <span className="flex-1 space-y-1 text-foreground/90">
                      {c.status === "pending" && <span className="text-muted-foreground">transcribing…</span>}
                      {c.status === "error" && <span className="text-destructive">failed to transcribe</span>}
                      {c.status === "done" && c.segments.length > 0 ? (
                        c.segments.map((s, i) => (
                          <div key={i} className="flex items-baseline gap-1.5">
                            <Badge variant="outline" className={cn("shrink-0 rounded-full px-1.5 py-0 text-[10px]", speakerColor(s.label).chip)}>
                              {s.label}
                            </Badge>
                            <span>{s.text}</span>
                          </div>
                        ))
                      ) : c.status === "done" ? (
                        c.text || <span className="text-muted-foreground">(no speech detected)</span>
                      ) : null}
                    </span>
                    {c.mood && c.status === "done" && (
                      <Badge variant="outline" className={cn("shrink-0 rounded-full px-2 py-0 text-[11px] capitalize", MOOD_STYLE[c.mood])}>
                        {c.mood}
                      </Badge>
                    )}
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
