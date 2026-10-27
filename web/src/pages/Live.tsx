import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { api } from "@/api/client";
import { RED, muted } from "@/lib/ui";

// Transcription only. Each chunk is cut at a natural pause (so lines don't split
// mid-word), uploaded, and transcribed on the worker at interactive priority; nothing else
// runs on it. Naive fixed-threshold energy VAD — good enough to find pauses in a normal
// room; may need retuning on a very noisy input or a hot mic.
const SPEECH_RMS_THRESHOLD = 0.02;
const PAUSE_MS = 600;       // silence this long after speech = end of phrase, cut here
const MIN_CHUNK_MS = 1200;  // don't cut on a micro-pause before this much has been said
const MAX_CHUNK_MS = 12000; // hard cap so continuous talk without a pause can't grow forever
const LEVEL_NORMALIZER = 0.12;
// Opus at a higher bitrate than the browser default: the encoder's artefacts are the one
// accuracy cost this page adds before the audio ever reaches the model.
const AUDIO_BPS = 128000;

type Line = { text: string; startS: number; endS: number };
type Chunk = {
  seq: number; status: "pending" | "done" | "error"; t: number; lines: Line[];
  sentAt: number; latencyMs?: number;
};
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
  const [startError, setStartError] = useState<string | null>(null);

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
  const levelRef = useRef<HTMLSpanElement>(null);

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
    // driven through the DOM, not React state — this runs every animation frame
    if (levelRef.current) levelRef.current.style.width = `${Math.round(Math.min(1, rms / LEVEL_NORMALIZER) * 100)}%`;

    const state = vadStateRef.current;
    const now = performance.now();
    const isSpeech = rms > SPEECH_RMS_THRESHOLD;
    if (isSpeech) { state.hasSpeech = true; state.silenceSince = null; }
    else if (state.silenceSince == null) state.silenceSince = now;
    setSpeaking((prev) => (prev !== isSpeech ? isSpeech : prev));

    const elapsed = now - state.start;
    const paused = state.hasSpeech && state.silenceSince != null && (now - state.silenceSince) >= PAUSE_MS;
    if (((elapsed >= MIN_CHUNK_MS && paused) || elapsed >= MAX_CHUNK_MS)
        && recorderRef.current?.state === "recording") recorderRef.current.stop();
    rafRef.current = requestAnimationFrame(vadTick);
  }

  // MediaRecorder chunks from one continuous recording aren't independently decodable
  // (only the first carries a container header) — so each pause-bounded window is its own
  // start/stop recorder, each a self-contained webm file.
  function recordOneChunk(stream: MediaStream) {
    if (stopRequestedRef.current) return;
    const recorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus", audioBitsPerSecond: AUDIO_BPS });
    recorderRef.current = recorder;
    vadStateRef.current = { start: performance.now(), hasSpeech: false, silenceSince: null };
    const seq = seqRef.current++;
    const parts: BlobPart[] = [];
    recorder.ondataavailable = (e) => { if (e.data.size > 0) parts.push(e.data); };
    recorder.onstop = async () => {
      if (vadStateRef.current.hasSpeech) {
        const blob = new Blob(parts, { type: "audio/webm" });
        const now = performance.now();
        setChunks((prev) => [...prev, { seq, status: "pending", t: now - sessionStartRef.current, lines: [], sentAt: now }]);
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
        type: string; seq: number; error?: string | null;
        turns?: { text: string; start_s: number; end_s: number }[];
      };
      if (msg.type !== "live_transcript") return;
      const now = performance.now();
      setChunks((prev) => prev.map((c) => (c.seq === msg.seq
        ? {
            ...c, status: msg.error ? "error" : "done", latencyMs: now - c.sentAt,
            lines: (msg.turns || []).map((t) => ({ text: t.text, startS: t.start_s, endS: t.end_s })),
          }
        : c)));
    };
    wsRef.current = ws;

    setRecording(true);
    recordOneChunk(stream);
    rafRef.current = requestAnimationFrame(vadTick);
    elapsedTimerRef.current = setInterval(() => setElapsedMs(performance.now() - sessionStartRef.current), 1000);
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

  async function toggle() {
    if (recording) { stop(); return; }
    setStartError(null);
    try { await start(); } catch (e) { setStartError((e as Error).message || String(e)); }
  }

  const done = chunks.filter((c) => c.status === "done");
  const pending = chunks.filter((c) => c.status === "pending").length;
  const transcript = done.flatMap((c) => c.lines.map((l) => l.text)).join(" ").trim();
  const words = transcript ? transcript.split(/\s+/).length : 0;
  const lat = done.map((c) => c.latencyMs!).filter((x) => x != null).sort((a, b) => a - b);
  const medianLat = lat.length ? lat[Math.floor(lat.length / 2)] : null;

  async function copy() {
    try {
      await navigator.clipboard.writeText(transcript);
      toast.success("Transcript copied");
    } catch (e) {
      toast.error("Could not copy", { description: String(e) });
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20, maxWidth: 860 }}>
      <header style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 20, flexWrap: "wrap" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
          <span className="kicker">Live transcription</span>
          <h2 style={{ margin: 0, fontSize: 30 }}>Microphone</h2>
          <p style={{ margin: 0, maxWidth: "62ch", fontSize: 14, color: muted(68) }}>
            Speak normally — each phrase is transcribed on this machine as soon as you pause.
          </p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          {recording && (
            <span className="mono" style={{
              display: "flex", alignItems: "center", gap: 8, fontSize: 12, padding: "6px 10px",
              border: `1px solid ${RED}`, color: RED,
            }}>
              <span style={{ width: 7, height: 7, background: RED, borderRadius: "50%" }} />
              REC {fmtElapsed(elapsedMs)}
            </span>
          )}
          <button className="btn btn-secondary" disabled={!transcript} onClick={copy}>Copy transcript</button>
          <button className={recording ? "btn btn-secondary" : "btn btn-primary"} onClick={toggle}>
            {recording ? "Stop" : "Start listening"}
          </button>
        </div>
      </header>

      <div className="blueprint" style={{ padding: 13, display: "flex", flexDirection: "column", gap: 8 }}>
        <div className="mono" style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5, color: muted(60) }}>
          <span>{recording ? (pending > 0 ? `transcribing ${pending}…` : speaking ? "hearing you" : "listening") : "idle"}</span>
          <span>{words} words{medianLat != null ? ` · ${(medianLat / 1000).toFixed(1)}s after each pause` : ""}</span>
        </div>
        <div style={{ height: 8, background: muted(10) }}>
          <span ref={levelRef} style={{ display: "block", width: 0, height: "100%", background: "var(--color-accent)" }} />
        </div>
        {startError && <span style={{ fontSize: 12, color: RED }}>{startError}</span>}
      </div>

      {/* Newest first: the line that just arrived is the one being looked for. */}
      <div style={{ display: "flex", flexDirection: "column", borderTop: "1px solid var(--color-divider)" }}>
        {[...chunks].reverse().map((c) => (
          <div key={c.seq} style={{
            display: "grid", gridTemplateColumns: "56px 1fr", gap: 14, padding: "11px 0",
            borderBottom: `1px solid ${muted(8)}`,
          }}>
            <span className="mono" style={{ fontSize: 12, color: muted(50) }}>{fmtElapsed(c.t)}</span>
            <p style={{ margin: 0, fontSize: 16, lineHeight: 1.55,
                        color: c.status === "done" && c.lines.length ? "var(--color-text)" : muted(55) }}>
              {c.status === "pending" ? "…"
                : c.status === "error" ? "this phrase could not be transcribed"
                : c.lines.length ? c.lines.map((l) => l.text).join(" ")
                : "(no speech)"}
            </p>
          </div>
        ))}
        {!recording && chunks.length === 0 && (
          <p style={{ padding: "24px 0", fontSize: 13, color: muted(55) }}>
            Nothing transcribed yet. Press Start listening and speak.
          </p>
        )}
      </div>
    </div>
  );
}
