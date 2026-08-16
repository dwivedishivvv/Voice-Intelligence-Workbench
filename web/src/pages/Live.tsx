import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "@/api/client";
import { AMBER, AMBER_INK, GREEN, MOOD_COLOR, MOOD_PCT, RED, muted, tagTone } from "@/lib/ui";

// VAD-driven chunk boundaries: cut on a natural pause instead of an arbitrary timer, so
// chunks end at sentence/phrase breaks rather than mid-word — the single biggest source of
// garbled live text. Naive fixed-threshold energy VAD (not adaptive to the noise floor or
// mic gain) — good enough to find pauses in a normal room; may need retuning on a very
// noisy input or a hot mic.
const SPEECH_RMS_THRESHOLD = 0.02;
const PAUSE_MS = 600;      // silence this long after speech = end of phrase, cut here
const MIN_CHUNK_MS = 1200; // don't cut on a micro-pause before this much has been said
const MAX_CHUNK_MS = 12000; // hard cap so continuous talk without a pause can't grow forever
const LEVEL_NORMALIZER = 0.12;

type ChunkStatus = "pending" | "done" | "error";
type Chunk = { seq: number; text: string; status: ChunkStatus; mood?: string; t: number };
type VadState = { start: number; hasSpeech: boolean; silenceSince: number | null };

const TOOLS: [string, "local" | "off-box"][] = [
  ["transcribe_chunk", "local"],
  ["tone_readings", "local"],
  ["voice_activity", "local"],
  ["store_session", "local"],
  ["search_corpus", "local"],
  ["compose_answer", "off-box"],
];

function fmtElapsed(ms: number) {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, "0")}`;
}

export default function Live() {
  const navigate = useNavigate();
  const [recording, setRecording] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [startedAt, setStartedAt] = useState<Date | null>(null);
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
  const liveBarsRef = useRef<HTMLDivElement>(null);

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
    const level = Math.min(1, rms / LEVEL_NORMALIZER);

    // driven through the DOM, not React state — this runs every animation frame and a
    // state-driven re-render at 60fps would be pure waste
    if (levelRef.current) levelRef.current.style.width = `${Math.round(level * 100)}%`;
    if (liveBarsRef.current) {
      const bars = liveBarsRef.current.children;
      for (let i = 0; i < bars.length; i++) {
        const jitter = 0.35 + 0.65 * Math.abs(Math.sin(performance.now() / 120 + i));
        (bars[i] as HTMLElement).style.height = `${Math.max(2, Math.round(level * 14 * jitter))}px`;
      }
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
    const pausedLongEnough = state.hasSpeech && state.silenceSince != null
      && (now - state.silenceSince) >= PAUSE_MS;
    const shouldCut = (elapsed >= MIN_CHUNK_MS && pausedLongEnough) || elapsed >= MAX_CHUNK_MS;
    if (shouldCut && recorderRef.current?.state === "recording") recorderRef.current.stop();

    rafRef.current = requestAnimationFrame(vadTick);
  }

  // MediaRecorder chunks from a single continuous recording aren't independently decodable
  // (only the first carries a container header) — so each VAD-bounded window is its own
  // start/stop recorder instance, each a self-contained webm file.
  function recordOneChunk(stream: MediaStream) {
    if (stopRequestedRef.current) return;
    const recorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
    recorderRef.current = recorder;
    vadStateRef.current = { start: performance.now(), hasSpeech: false, silenceSince: null };
    const seq = seqRef.current++;
    const parts: BlobPart[] = [];

    recorder.ondataavailable = (e) => { if (e.data.size > 0) parts.push(e.data); };
    recorder.onstop = async () => {
      if (vadStateRef.current.hasSpeech) {
        const blob = new Blob(parts, { type: "audio/webm" });
        setChunks((prev) => [...prev, {
          seq, text: "", status: "pending", t: performance.now() - sessionStartRef.current,
        }]);
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
    setStartedAt(new Date());

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
        type: string; seq: number; text: string; mood?: string; error?: string;
      };
      if (msg.type !== "live_transcript") return;
      setChunks((prev) => prev.map((c) =>
        c.seq === msg.seq
          ? { ...c, text: msg.text, mood: msg.mood, status: msg.error ? "error" : "done" }
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

  const done = chunks.filter((c) => c.status === "done");
  const pending = chunks.filter((c) => c.status === "pending").length;
  const flagged = [...done].reverse().find((c) => c.mood === "stressed" && c.text);
  const words = done.map((c) => c.text).join(" ").trim().split(/\s+/).filter(Boolean).length;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <header style={{
        display: "flex", alignItems: "flex-end", justifyContent: "space-between",
        gap: 20, flexWrap: "wrap",
      }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
          <span className="kicker">
            {startedAt
              ? `Live session · started ${startedAt.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}`
              : "Live session · not started"}
          </span>
          <h2 style={{ margin: 0, fontSize: 30 }}>Microphone channel</h2>
          <p style={{ margin: 0, maxWidth: "66ch", fontSize: 14, color: muted(68) }}>
            Transcription and tone readings run on the box as the audio arrives. Chunks close
            at natural pauses, not on a timer. Identification is not run per chunk — upload a
            recording when you need voices matched against enrolled profiles.
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
          <button className={recording ? "btn btn-secondary" : "btn btn-primary"}
                  onClick={recording ? stop : start}>
            {recording ? "Stop" : "Start listening"}
          </button>
        </div>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 24, alignItems: "start" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div style={{ display: "flex", flexDirection: "column", borderTop: "1px solid var(--color-divider)" }}>
            {chunks.map((c) => (
              <div key={c.seq} style={{
                display: "grid", gridTemplateColumns: "64px 1fr", gap: 14, padding: "12px 0",
                borderBottom: `1px solid ${muted(8)}`,
              }}>
                <span className="mono" style={{ fontSize: 12, color: muted(50) }}>{fmtElapsed(c.t)}</span>
                <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                    <span style={{ fontFamily: "var(--font-heading)", fontSize: 15, color: muted(70) }}>
                      Speaker (unnamed)
                    </span>
                    <span className="kicker-sm" style={{ letterSpacing: ".1em", color: muted(50) }}>
                      {c.status === "pending" ? "transcribing" : c.status === "error" ? "failed" : "session chunk"}
                    </span>
                    {c.mood && (
                      <span className="mono" style={{
                        display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: muted(55),
                      }}>
                        <span style={{ width: 26, height: 5, background: muted(12), display: "inline-block", position: "relative" }}>
                          <span style={{
                            position: "absolute", left: 0, top: 0, bottom: 0,
                            width: MOOD_PCT[c.mood] || "30%", background: MOOD_COLOR[c.mood] || muted(40),
                          }} />
                        </span>
                        {c.mood}
                      </span>
                    )}
                  </div>
                  <p style={{
                    margin: 0, fontSize: 15, lineHeight: 1.55,
                    color: c.status === "done" ? "var(--color-text)" : muted(60),
                  }}>
                    {c.status === "pending" ? "…" : c.status === "error" ? "chunk failed to transcribe"
                      : c.text || "(no speech detected)"}
                  </p>
                </div>
              </div>
            ))}

            {recording && (
              <div style={{
                display: "grid", gridTemplateColumns: "64px 1fr", gap: 14, padding: "12px 0",
                borderBottom: `1px solid ${muted(8)}`,
                background: "color-mix(in srgb, var(--color-accent) 6%, transparent)",
              }}>
                <span className="mono" style={{ fontSize: 12, color: "var(--color-accent)" }}>
                  {fmtElapsed(elapsedMs)}
                </span>
                <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span className="mono" style={{ fontSize: 12, color: muted(55) }}>
                    {pending > 0 ? `transcribing ${pending} chunk${pending === 1 ? "" : "s"}`
                      : speaking ? "hearing you · chunk closes at the next pause"
                      : "listening · chunk closes at the next pause"}
                  </span>
                  <span ref={liveBarsRef} style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 14 }}>
                    {Array.from({ length: 14 }, (_, i) => (
                      <span key={i} style={{ width: 3, height: 2, background: "var(--color-accent-400)" }} />
                    ))}
                  </span>
                </div>
              </div>
            )}

            {!recording && chunks.length === 0 && (
              <p style={{ padding: "24px 0", fontSize: 13, color: muted(55) }}>
                Nothing transcribed yet. Start listening and speak normally — text and a tone
                reading appear after each pause.
              </p>
            )}
          </div>

          {chunks.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 12, paddingTop: 4 }}>
              <div style={{
                display: "flex", alignItems: "baseline", justifyContent: "space-between",
                gap: 16, flexWrap: "wrap",
              }}>
                <h4 style={{ margin: 0 }}>What the pipeline has so far</h4>
                <a href="/ask" onClick={(e) => { e.preventDefault(); navigate("/ask"); }} style={{ fontSize: 12.5 }}>
                  Ask about the corpus in Ask →
                </a>
              </div>

              <section className="blueprint" style={{ padding: 14, display: "flex", flexDirection: "column", gap: 9 }}>
                <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
                  <span className="kicker-sm" style={{ color: muted(55) }}>Session so far</span>
                  <span className="mono" style={{ fontSize: 11, color: muted(45) }}>
                    {done.length} chunks · {words} words
                  </span>
                </div>
                <div className="mono" style={{ display: "flex", gap: 20, fontSize: 11.5, color: muted(60), flexWrap: "wrap" }}>
                  <span>calm {done.filter((c) => c.mood === "calm").length}</span>
                  <span>tired {done.filter((c) => c.mood === "tired").length}</span>
                  <span>stressed {done.filter((c) => c.mood === "stressed").length}</span>
                </div>
                <span style={{ fontSize: 12, color: muted(55) }}>
                  A live session is transcript and tone only. Nothing here invents an identity
                  for a voice it cannot place.
                </span>
              </section>

              {flagged && (
                <section className="blueprint" style={{ padding: 14, display: "flex", flexDirection: "column", gap: 9 }}>
                  <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
                    <span className="kicker-sm" style={{ color: muted(55) }}>Flagged moment</span>
                    <span className="mono" style={{ fontSize: 11, color: muted(45) }}>
                      {fmtElapsed(flagged.t)} · this session
                    </span>
                  </div>
                  <p style={{
                    margin: 0, fontSize: 15, lineHeight: 1.6,
                    borderLeft: "2px solid var(--color-divider)", paddingLeft: 12,
                  }}>
                    {flagged.text}
                  </p>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                    <span className="tag mono" style={{ border: `1px solid ${RED}`, color: RED }}>
                      voice · stressed
                    </span>
                    <span className="tag tag-neutral mono" style={{ border: "1px solid var(--color-neutral-300)" }}>
                      text · not scored live
                    </span>
                    <span style={{ fontSize: 12, color: muted(55) }}>
                      The tone read is acoustic only; text sentiment is scored when a clip is processed.
                    </span>
                  </div>
                </section>
              )}
            </div>
          )}
        </div>

        <aside style={{ display: "flex", flexDirection: "column", gap: 16, position: "sticky", top: 26 }}>
          <div className="blueprint" style={{ padding: 13, display: "flex", flexDirection: "column", gap: 9 }}>
            <span className="kicker-sm" style={{ color: muted(55) }}>Input</span>
            <div className="mono" style={{
              display: "flex", justifyContent: "space-between", fontSize: 11.5, color: muted(60),
            }}>
              <span>default microphone</span>
              <span>{recording ? (speaking ? "speech" : "silence") : "idle"}</span>
            </div>
            <div style={{ height: 8, background: muted(10), position: "relative" }}>
              <span ref={levelRef} style={{ display: "block", width: 0, height: "100%", background: "var(--color-accent)" }} />
              <div style={{
                position: "absolute", left: "88%", top: -3, bottom: -3, width: 1, background: RED,
              }} />
            </div>
            <span style={{ fontSize: 11.5, color: muted(55) }}>
              Chunks close at natural pauses, not on a timer, so lines do not split mid-sentence.
            </span>
          </div>

          <div className="blueprint" style={{ padding: 13, display: "flex", flexDirection: "column", gap: 10 }}>
            <span className="kicker-sm" style={{ color: muted(55) }}>Tone across the session</span>
            <div style={{ display: "flex", gap: 2, height: 34, alignItems: "flex-end" }}>
              {done.length === 0 && <span style={{ fontSize: 12, color: muted(45) }}>nothing scored yet</span>}
              {done.map((c) => (
                <div key={c.seq} title={c.mood || "unscored"} style={{
                  flex: 1,
                  height: c.mood === "stressed" ? 30 : c.mood === "tired" ? 22 : 14,
                  background: MOOD_COLOR[c.mood || ""] || muted(20),
                }} />
              ))}
            </div>
            <div style={{ display: "flex", gap: 12, fontSize: 11.5, color: muted(58) }}>
              <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <span style={{ width: 9, height: 9, background: GREEN }} />calm
              </span>
              <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <span style={{ width: 9, height: 9, background: AMBER }} />tired
              </span>
              <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <span style={{ width: 9, height: 9, background: RED }} />stressed
              </span>
            </div>
          </div>

          <div className="blueprint" style={{ padding: 13, display: "flex", flexDirection: "column", gap: 8 }}>
            <span className="kicker-sm" style={{ color: muted(55) }}>Where this session runs</span>
            {TOOLS.map(([name, scope]) => {
              const tone = scope === "local" ? tagTone("accent") : tagTone("warn");
              return (
                <div key={name} className="mono" style={{
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                  gap: 8, fontSize: 11.5,
                }}>
                  <span>{name}</span>
                  <span className="tag" style={{
                    border: `1px solid ${tone.border}`, color: tone.color, fontSize: 10,
                  }}>
                    {scope}
                  </span>
                </div>
              );
            })}
            <span style={{ fontSize: 11, lineHeight: 1.5, color: muted(50), marginTop: 2 }}>
              Everything in a live session stays local. Only Ask sends anything off-box, and
              only to compose an answer.
            </span>
          </div>

          <p style={{ margin: 0, fontSize: 11, lineHeight: 1.5, color: muted(50) }}>
            Tone labels are heuristic readings of the audio, not measurements of how anyone felt.
          </p>
          <p style={{ margin: 0, fontSize: 11, lineHeight: 1.5, color: AMBER_INK }}>
            Live chunks are transcribed only. Speaker identification needs a full clip.
          </p>
        </aside>
      </div>
    </div>
  );
}
