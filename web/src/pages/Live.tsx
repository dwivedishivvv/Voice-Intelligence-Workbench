import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/api/client";
import { Dialog } from "@/components/dialog";
import { AMBER, AMBER_INK, GREEN, MOOD_COLOR, RED, muted, tagTone } from "@/lib/ui";

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
// A chunk is one VAD-bounded upload; the worker now diarizes it, so it can come back as
// more than one turn — two people sharing a mic, or cross-talk on a radio channel.
type Turn = {
  text: string;
  // mood: acoustic-only (pitch/rate), what the voice sounds like. sentiment: fused —
  // the trained text model's read, with mood allowed only a small nudge (stages/sentiment.py
  // MOOD_SHIFT) that can never invent a verdict the words don't support. textSentiment is
  // the unfused text-only read, shown so a disagreement between the two is visible rather
  // than averaged away. sentiment is the one that decides what gets flagged.
  mood?: string;
  sentiment?: string | null; sentimentScore?: number | null;
  textSentiment?: string | null; textScore?: number | null;
  speaker?: string | null; speakerResult?: "confident" | "suggested" | null; speakerScore?: number | null;
  startS: number; endS: number;
  // Present whenever the turn was long enough to embed at all (see LIVE_EMBED_MIN_S on the
  // worker) — carried even on an unmatched turn so it can be named without re-recording.
  embedding?: number[] | null; reliability?: number | null;
  // Session-local voice index from the worker's online clustering (_pool_voice) — stable
  // for the life of the session, present whenever embedding is. Groups unmatched turns by
  // the same underlying voice in the Grouped tab even before anyone has named it.
  voiceCluster?: number | null;
};
type Chunk = { seq: number; status: ChunkStatus; t: number; turns: Turn[] };
type VadState = { start: number; hasSpeech: boolean; silenceSince: number | null };
type Source = "mic" | "f1";

const TOOLS: [string, "local" | "off-box"][] = [
  ["transcribe_chunk", "local"],
  ["diarize_turns", "local"],
  ["tone_readings", "local"],
  ["voice_activity", "local"],
  ["speaker_match", "local"],
  ["store_session", "local"],
  ["search_corpus", "local"],
  ["compose_answer", "off-box"],
];

function fmtElapsed(ms: number) {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, "0")}`;
}

// The fused (text-dominant) sentiment decides what's worth flagging, not raw acoustic
// mood alone — a stressed-sounding voice saying something ordinary isn't a flag, and a
// calm-sounding voice saying something bad still is. Falls back to acoustic mood only
// when the sentiment model isn't available (see worker/main.py::_process_live_turn).
function isFlagged(t: { mood?: string; sentiment?: string | null }) {
  if (t.sentiment) return t.sentiment === "negative";
  return !!t.mood && t.mood !== "calm";
}

export default function Live() {
  const navigate = useNavigate();
  const [recording, setRecording] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [feedTab, setFeedTab] = useState<"live" | "grouped">("live");
  const [elapsedMs, setElapsedMs] = useState(0);
  const [startedAt, setStartedAt] = useState<Date | null>(null);
  const [source, setSource] = useState<Source>("mic");
  const [f1Mode, setF1Mode] = useState<"url" | "file">("file");
  const [f1Url, setF1Url] = useState("");
  const [f1File, setF1File] = useState<File | null>(null);
  const [startError, setStartError] = useState<string | null>(null);
  // voiceCluster, not (seq, idx): tagging one turn from a session-local voice applies the
  // name to every turn sharing that cluster, in both the Live and Grouped tabs at once —
  // the whole point of clustering unmatched turns is that they're the same person.
  const [tagging, setTagging] = useState<{
    embedding: number[]; reliability: number | null; durationS: number;
    text?: string; voiceCluster: number | null;
  } | null>(null);
  const [tagName, setTagName] = useState("");
  const [tagBusy, setTagBusy] = useState(false);
  const sessionIdRef = useRef<string | null>(null);
  const sessionStartRef = useRef(0);
  const seqRef = useRef(0);
  const streamRef = useRef<MediaStream | null>(null);
  const f1AudioRef = useRef<HTMLAudioElement>(null);
  const f1ObjectUrlRef = useRef<string | null>(null);
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

  useEffect(() => () => {
    stop();
    if (f1ObjectUrlRef.current) URL.revokeObjectURL(f1ObjectUrlRef.current);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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
          seq, status: "pending", t: performance.now() - sessionStartRef.current, turns: [],
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

  // The F1 source is played out loud (so it "acts as radio") and captured back off the
  // <audio> element as a MediaStream — from there it is indistinguishable from a mic
  // stream to everything below (VAD, chunk boundaries, upload, transcript). A remote
  // livetiming URL only captures cleanly if that host sends CORS headers permitting this
  // origin; if it doesn't, the browser taints the element and captureStream() throws.
  // ponytail: no server-side proxy fallback for a tainted remote URL — downloading the
  // clip once and using "Load mp3" always works, so that's the answer for a CDN that
  // blocks capture rather than a second code path here.
  async function captureF1Stream(): Promise<MediaStream> {
    const audio = f1AudioRef.current!;
    if (f1Mode === "file") {
      if (!f1File) throw new Error("choose an mp3 file first");
      if (f1ObjectUrlRef.current) URL.revokeObjectURL(f1ObjectUrlRef.current);
      f1ObjectUrlRef.current = URL.createObjectURL(f1File);
      audio.src = f1ObjectUrlRef.current;
    } else {
      const url = f1Url.trim();
      if (!url) throw new Error("paste a radio clip URL first");
      audio.crossOrigin = "anonymous";
      audio.src = url;
    }
    audio.currentTime = 0;
    await audio.play();
    const cap = (audio as any).captureStream || (audio as any).mozCaptureStream;
    if (!cap) throw new Error("this browser cannot capture audio-element playback");
    return cap.call(audio) as MediaStream;
  }

  async function start() {
    const stream = source === "f1" ? await captureF1Stream()
      : await navigator.mediaDevices.getUserMedia({ audio: true });
    streamRef.current = stream;
    if (source === "f1" && f1AudioRef.current) {
      f1AudioRef.current.onended = () => stop();
    }
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
        type: string; seq: number; error?: string;
        turns?: {
          text: string; mood?: string; speaker?: string | null;
          speaker_result?: "confident" | "suggested" | null; speaker_score?: number | null;
          start_s: number; end_s: number; embedding?: number[] | null; reliability?: number | null;
          voice_cluster?: number | null;
          sentiment?: string | null; sentiment_score?: number | null;
          text_sentiment?: string | null; text_score?: number | null;
        }[];
      };
      if (msg.type !== "live_transcript") return;
      setChunks((prev) => prev.map((c) =>
        c.seq === msg.seq
          ? {
              ...c, status: msg.error ? "error" : "done",
              turns: (msg.turns || []).map((t) => ({
                text: t.text, mood: t.mood, speaker: t.speaker,
                speakerResult: t.speaker_result, speakerScore: t.speaker_score,
                startS: t.start_s, endS: t.end_s,
                embedding: t.embedding, reliability: t.reliability, voiceCluster: t.voice_cluster,
                sentiment: t.sentiment, sentimentScore: t.sentiment_score,
                textSentiment: t.text_sentiment, textScore: t.text_score,
              })),
            }
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
    if (f1AudioRef.current) {
      f1AudioRef.current.onended = null;
      f1AudioRef.current.pause();
    }
    setRecording(false);
    setSpeaking(false);
  }

  async function toggle() {
    if (recording) { stop(); return; }
    setStartError(null);
    try {
      await start();
    } catch (e) {
      setStartError((e as Error).message || String(e));
    }
  }

  async function confirmTag() {
    if (!tagging) return;
    const name = tagName.trim();
    if (!name) return;
    setTagBusy(true);
    try {
      await api.tagLiveSpeaker({
        display_name: name, embedding: tagging.embedding,
        duration_s: tagging.durationS, reliability: tagging.reliability ?? 0.3,
      });
      const cluster = tagging.voiceCluster;
      setChunks((prev) => prev.map((c) => ({
        ...c,
        turns: c.turns.map((t) => (cluster != null && t.voiceCluster === cluster
          ? { ...t, speaker: name, speakerResult: "confident", speakerScore: null }
          : t)),
      })));
      toast.success(`Tagged as ${name}`, { description: "Every turn from this voice this session now shows the name, and later chunks will match it automatically." });
      setTagging(null);
      setTagName("");
    } catch (e) {
      toast.error("Could not tag speaker", { description: String((e as Error).message) });
    } finally {
      setTagBusy(false);
    }
  }

  const f1Ready = f1Mode === "url" ? !!f1Url.trim() : !!f1File;
  const canStart = source === "mic" || f1Ready;

  const done = chunks.filter((c) => c.status === "done");
  const pending = chunks.filter((c) => c.status === "pending").length;
  // Flattened once, elapsed time carried down from the parent chunk: a chunk's own turns
  // don't know the session clock, only their offset inside the chunk's audio.
  const rawTurns = done.flatMap((c) => c.turns.map((t) => ({ ...t, chunkT: c.t })));

  // Identity firms up as a voice keeps talking: the worker pools each session-local
  // cluster's audio across chunks, so the same person can come back unmatched on their
  // first two-word turn and confidently matched three turns later. Pooling only ever runs
  // forward, so without this the early turns would keep showing "unnamed" for the rest of
  // the session even though the voice has since been identified. Carrying the newest
  // identity for a cluster back across its earlier turns is what makes it retroactive —
  // and it costs nothing, because the cluster id already says they are the same voice.
  const clusterIdentity = new Map<number, { speaker: string; result: Turn["speakerResult"]; score: number | null }>();
  for (const t of rawTurns) {
    if (t.voiceCluster != null && t.speaker) {
      clusterIdentity.set(t.voiceCluster, {
        speaker: t.speaker, result: t.speakerResult ?? null, score: t.speakerScore ?? null,
      });
    }
  }
  // `inherited` keeps the distinction visible: this turn was not itself matched, it is
  // wearing the name of the voice it clusters with.
  const resolveSpeaker = (t: Turn) => {
    if (t.speaker) return { speaker: t.speaker, result: t.speakerResult ?? null, score: t.speakerScore ?? null, inherited: false };
    const c = t.voiceCluster != null ? clusterIdentity.get(t.voiceCluster) : undefined;
    return c ? { ...c, inherited: true }
             : { speaker: null as string | null, result: null as Turn["speakerResult"], score: null as number | null, inherited: false };
  };

  const turns = rawTurns.map((t) => ({ ...t, ...resolveSpeaker(t) }));
  const flagged = [...turns].reverse().find((t) => t.text && isFlagged(t));
  const words = turns.map((t) => t.text).join(" ").trim().split(/\s+/).filter(Boolean).length;

  // Per-turn matches, not a session identity: the same person can land as "unmatched" on
  // one turn and a named match on the next, so this counts occurrences rather than
  // asserting one speaker per session the way a full clip's diarization would.
  const speakerCounts = new Map<string, { n: number; confident: number }>();
  for (const t of turns) {
    if (!t.speaker) continue;
    const row = speakerCounts.get(t.speaker) || { n: 0, confident: 0 };
    row.n += 1;
    if (t.speakerResult === "confident") row.confident += 1;
    speakerCounts.set(t.speaker, row);
  }
  const speakers = [...speakerCounts.entries()].sort((a, b) => b[1].n - a[1].n);
  const unmatched = turns.filter((t) => t.text && !t.speaker).length;

  // Grouped tab: bucket every turn with something to show by who said it — a matched name,
  // or (failing that) the session-local voice cluster the worker's online pooling already
  // assigned it to, so unmatched turns from the same voice sit together before anyone has
  // named it. `turns` is in chronological order, so each group's own list is too, and its
  // last entry carries the most-pooled (strongest) embedding available for tagging.
  type Group = typeof turns[number] & { chunkT: number };
  const groupMap = new Map<string, { key: string; name: string; matched: boolean; turns: Group[] }>();
  for (const t of turns) {
    if (!t.text) continue;
    const key = t.speaker ? `name:${t.speaker}`
      : t.voiceCluster != null ? `cluster:${t.voiceCluster}` : "unclustered";
    let g = groupMap.get(key);
    if (!g) {
      g = {
        key, matched: !!t.speaker,
        name: t.speaker || (t.voiceCluster != null
          ? `Unidentified voice ${t.voiceCluster + 1}` : "Too short to identify"),
        turns: [],
      };
      groupMap.set(key, g);
    }
    if (t.speaker) { g.name = t.speaker; g.matched = true; }
    g.turns.push(t);
  }
  const groups = [...groupMap.values()].sort((a, b) => b.turns.length - a.turns.length);

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
          <h2 style={{ margin: 0, fontSize: 30 }}>
            {source === "f1" ? "F1 radio channel" : "Microphone channel"}
          </h2>
          <p style={{ margin: 0, maxWidth: "66ch", fontSize: 14, color: muted(68) }}>
            {source === "f1"
              ? "The clip plays out loud and is captured back off the player — chunked, diarized, transcribed, tone-read and matched against enrolled profiles the same way a live mic is, as it plays."
              : "Transcription, tone and speaker matching all run on the box as the audio arrives. Chunks close at natural pauses, not on a timer, and each one is diarized on its own — so a chunk with more than one voice in it comes back as more than one turn. Still no memory across chunks, so treat a name here as a lead, not a verdict."}
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
                  disabled={!recording && !canStart}
                  onClick={toggle}>
            {recording ? "Stop" : source === "f1" ? "Play & transcribe" : "Start listening"}
          </button>
        </div>
      </header>

      <div className="blueprint" style={{ padding: 13, display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <div className="seg">
            <label className="seg-opt">
              <input type="radio" name="source" checked={source === "mic"} disabled={recording}
                     onChange={() => setSource("mic")} />
              Microphone
            </label>
            <label className="seg-opt">
              <input type="radio" name="source" checked={source === "f1"} disabled={recording}
                     onChange={() => setSource("f1")} />
              F1 team radio
            </label>
          </div>

          {source === "f1" && (
            <>
              <div className="seg" style={{ fontSize: 11.5 }}>
                <label className="seg-opt">
                  <input type="radio" name="f1mode" checked={f1Mode === "file"} disabled={recording}
                         onChange={() => setF1Mode("file")} />
                  mp3 file
                </label>
                <label className="seg-opt">
                  <input type="radio" name="f1mode" checked={f1Mode === "url"} disabled={recording}
                         onChange={() => setF1Mode("url")} />
                  livetiming URL
                </label>
              </div>
              {f1Mode === "file" ? (
                <label className="btn btn-secondary" style={{ cursor: recording ? "default" : "pointer", opacity: recording ? 0.6 : 1 }}>
                  {f1File ? f1File.name : "Choose mp3…"}
                  <input
                    type="file" accept="audio/*,.mp3" disabled={recording} style={{ display: "none" }}
                    onChange={(e) => setF1File(e.target.files?.[0] || null)}
                  />
                </label>
              ) : (
                <input
                  className="input mono" placeholder="https://livetiming.formula1.com/…/TeamRadio.mp3"
                  value={f1Url} disabled={recording} onChange={(e) => setF1Url(e.target.value)}
                  style={{ flex: 1, minWidth: 260, fontSize: 12.5 }}
                />
              )}
            </>
          )}
        </div>

        {source === "f1" && (
          <audio ref={f1AudioRef} controls style={{ width: "100%", height: 32 }} />
        )}

        {source === "f1" && f1Mode === "url" && (
          <span style={{ fontSize: 11, color: muted(50) }}>
            Some CDNs don't allow a page on another origin to capture their audio for
            transcription (only to play it) — if "Play & transcribe" fails, download the clip
            and use the mp3 file option instead.
          </span>
        )}

        {startError && (
          <span style={{ fontSize: 12, color: RED }}>{startError}</span>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 24, alignItems: "start" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div className="seg" style={{ alignSelf: "flex-start" }}>
            <label className="seg-opt">
              <input type="radio" name="feedTab" checked={feedTab === "live"} onChange={() => setFeedTab("live")} />
              Live
            </label>
            <label className="seg-opt">
              <input type="radio" name="feedTab" checked={feedTab === "grouped"} onChange={() => setFeedTab("grouped")} />
              Grouped
            </label>
          </div>

          {feedTab === "live" && (
          <>
          {/* Newest first. A live session runs as long as someone keeps talking, so an
              append-at-the-bottom list walks the newest line off the screen exactly when it
              matters most — and the live row belongs at the top with it, since that is
              where the next line will appear. */}
          <div style={{ display: "flex", flexDirection: "column", borderTop: "1px solid var(--color-divider)" }}>
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

            {/* One VAD-bounded chunk can hold more than one speaker turn now that live
                chunks are diarized (stages/diarize.py + stages/reconcile.py, scoped to one
                chunk) — each turn gets its own row inside the chunk's time-anchored block.
                A mood tag only earns a place on a turn when it's not the default "calm"
                read, same reasoning as before: pinning a tag on every line buries the
                handful actually worth a second look. */}
            {[...chunks].reverse().map((c) => (
              <div key={c.seq} style={{
                display: "grid", gridTemplateColumns: "64px 1fr", gap: 14, padding: "12px 0",
                borderBottom: `1px solid ${muted(8)}`,
              }}>
                <span className="mono" style={{ fontSize: 12, color: muted(50) }}>{fmtElapsed(c.t)}</span>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {c.status === "pending" && (
                    <p style={{ margin: 0, fontSize: 15, color: muted(60) }}>…</p>
                  )}
                  {c.status === "error" && (
                    <p style={{ margin: 0, fontSize: 15, color: muted(60) }}>chunk failed to transcribe</p>
                  )}
                  {c.status === "done" && c.turns.length === 0 && (
                    <p style={{ margin: 0, fontSize: 15, color: muted(60) }}>(no speech detected)</p>
                  )}
                  {c.turns.map((t, i) => {
                    const flag = isFlagged(t);
                    const flagColor = t.sentiment === "negative" ? RED
                      : (t.mood && MOOD_COLOR[t.mood]) || AMBER;
                    const disagrees = !!t.textSentiment && !!t.sentiment && t.textSentiment !== t.sentiment;
                    const who = resolveSpeaker(t);
                    return (
                      <div key={i} style={{
                        display: "flex", flexDirection: "column", gap: 5,
                        borderLeft: flag ? `2px solid ${flagColor}` : "2px solid transparent",
                        paddingLeft: flag ? 10 : 0,
                      }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                          <span style={{
                            fontFamily: "var(--font-heading)", fontSize: 15,
                            color: who.speaker ? "var(--color-text)" : muted(70),
                          }}>
                            {who.speaker || (source === "f1" ? "F1 radio" : "Speaker (unnamed)")}
                          </span>
                          {who.speaker && (
                            <span className="tag mono" style={{
                              fontSize: 10, padding: "1px 6px",
                              border: `1px solid ${who.result === "confident" && !who.inherited ? "var(--color-accent)" : "var(--color-neutral-300)"}`,
                              color: who.result === "confident" && !who.inherited ? "var(--color-accent-700)" : muted(55),
                            }}>
                              {who.inherited ? "same voice"
                                : who.result === "confident" ? "matched" : "suggested"}
                              {!who.inherited && who.score != null && ` ${who.score.toFixed(2)}`}
                            </span>
                          )}
                          {/* Naming an unmatched (or only-suggested) turn enrolls the
                              embedding the worker already computed for it — no re-recording,
                              and every later chunk's identify() call can then match this voice. */}
                          {t.embedding && !(who.result === "confident" && !who.inherited) && (
                            <a href="#" style={{ fontSize: 11.5 }}
                               onClick={(e) => {
                                 e.preventDefault();
                                 setTagging({
                                   embedding: t.embedding!, reliability: t.reliability ?? null,
                                   durationS: t.endS - t.startS, text: t.text,
                                   voiceCluster: t.voiceCluster ?? null,
                                 });
                                 setTagName(who.speaker || "");
                               }}>
                              {who.speaker ? "confirm name" : "tag speaker"}
                            </a>
                          )}
                          {c.turns.length > 1 && (
                            <span className="kicker-sm" style={{ letterSpacing: ".1em", color: muted(45) }}>
                              turn {i + 1}/{c.turns.length}
                            </span>
                          )}
                          {/* The fused sentiment (text-dominant — see stages/sentiment.py
                              fuse_one) decides whether this turn is flagged at all; both
                              readings are shown once it is, same "voice ·" / "text ·"
                              convention as the Ask page's evidence cards, so a disagreement
                              between what was said and how it sounded stays visible instead
                              of being averaged into one number. */}
                          {flag && t.mood && t.mood !== "calm" && (
                            <span className="tag mono" style={{
                              fontSize: 10, padding: "1px 6px",
                              border: `1px solid ${MOOD_COLOR[t.mood] || AMBER}`, color: MOOD_COLOR[t.mood] || AMBER,
                            }}>
                              voice · {t.mood}
                            </span>
                          )}
                          {flag && t.sentiment && (
                            <span className="tag tag-neutral mono" style={{
                              fontSize: 10, padding: "1px 6px", border: "1px solid var(--color-neutral-300)",
                            }}>
                              text · {t.sentiment}
                              {t.sentimentScore != null && ` ${t.sentimentScore > 0 ? "+" : ""}${t.sentimentScore.toFixed(2)}`}
                            </span>
                          )}
                          {flag && disagrees && (
                            <span style={{ fontSize: 11, color: AMBER_INK }}>
                              text and voice disagree
                            </span>
                          )}
                        </div>
                        {t.text && (
                          <p style={{ margin: 0, fontSize: 15, lineHeight: 1.55, color: "var(--color-text)" }}>
                            {t.text}
                          </p>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}

            {!recording && chunks.length === 0 && (
              <p style={{ padding: "24px 0", fontSize: 13, color: muted(55) }}>
                Nothing transcribed yet. Start listening and speak normally — text and a tone
                reading appear after each pause.
              </p>
            )}
          </div>
          </>
          )}

          {feedTab === "grouped" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 12, borderTop: "1px solid var(--color-divider)", paddingTop: 14 }}>
              {groups.length === 0 && (
                <p style={{ padding: "10px 0", fontSize: 13, color: muted(55) }}>
                  Nothing to group yet.
                </p>
              )}
              {groups.map((g) => {
                const latest = g.turns[g.turns.length - 1];
                const wordCount = g.turns.map((t) => t.text).join(" ").trim().split(/\s+/).filter(Boolean).length;
                return (
                  <section key={g.key} className="blueprint" style={{ padding: 14, display: "flex", flexDirection: "column", gap: 10 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                      <span style={{
                        fontFamily: "var(--font-heading)", fontSize: 16,
                        color: g.matched ? "var(--color-text)" : muted(70),
                      }}>
                        {g.name}
                      </span>
                      <span className="mono" style={{ fontSize: 11, color: muted(50) }}>
                        {g.turns.length} turn{g.turns.length === 1 ? "" : "s"} · {wordCount} words
                      </span>
                      {!g.matched && latest.embedding && (
                        <a href="#" style={{ fontSize: 11.5 }}
                           onClick={(e) => {
                             e.preventDefault();
                             setTagging({
                               embedding: latest.embedding!, reliability: latest.reliability ?? null,
                               durationS: latest.endS - latest.startS, text: latest.text,
                               voiceCluster: latest.voiceCluster ?? null,
                             });
                             setTagName("");
                           }}>
                          tag this voice
                        </a>
                      )}
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                      {g.turns.map((t, i) => (
                        <p key={i} style={{ margin: 0, fontSize: 14.5, lineHeight: 1.55, display: "flex", gap: 10 }}>
                          <span className="mono" style={{ fontSize: 11.5, color: muted(45), flex: "none", paddingTop: 2 }}>
                            {fmtElapsed(t.chunkT)}
                          </span>
                          <span>{t.text}</span>
                        </p>
                      ))}
                    </div>
                  </section>
                );
              })}
            </div>
          )}

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
                    {done.length} chunks · {turns.length} turns · {words} words
                  </span>
                </div>
                <div className="mono" style={{ display: "flex", gap: 20, fontSize: 11.5, color: muted(60), flexWrap: "wrap" }}>
                  <span>calm {turns.filter((t) => t.mood === "calm").length}</span>
                  <span>tired {turns.filter((t) => t.mood === "tired").length}</span>
                  <span>stressed {turns.filter((t) => t.mood === "stressed").length}</span>
                </div>
                <span style={{ fontSize: 12, color: muted(55) }}>
                  Each chunk is diarized on its own — turns can outnumber chunks when more
                  than one voice lands in the same pause-to-pause window.
                </span>
              </section>

              {flagged && (
                <section className="blueprint" style={{ padding: 14, display: "flex", flexDirection: "column", gap: 9 }}>
                  <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
                    <span className="kicker-sm" style={{ color: muted(55) }}>Flagged moment</span>
                    <span className="mono" style={{ fontSize: 11, color: muted(45) }}>
                      {fmtElapsed(flagged.chunkT)} · this session
                      {flagged.speaker ? ` · ${flagged.speaker}` : ""}
                    </span>
                  </div>
                  <p style={{
                    margin: 0, fontSize: 15, lineHeight: 1.6,
                    borderLeft: "2px solid var(--color-divider)", paddingLeft: 12,
                  }}>
                    {flagged.text}
                  </p>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                    {flagged.mood && flagged.mood !== "calm" && (
                      <span className="tag mono" style={{ border: `1px solid ${MOOD_COLOR[flagged.mood] || RED}`, color: MOOD_COLOR[flagged.mood] || RED }}>
                        voice · {flagged.mood}
                      </span>
                    )}
                    {flagged.sentiment && (
                      <span className="tag tag-neutral mono" style={{ border: "1px solid var(--color-neutral-300)" }}>
                        text · {flagged.sentiment}
                        {flagged.sentimentScore != null && ` ${flagged.sentimentScore > 0 ? "+" : ""}${flagged.sentimentScore.toFixed(2)}`}
                      </span>
                    )}
                    <span style={{ fontSize: 12, color: muted(55) }}>
                      Flagged on the fused read — the trained text model, with delivery only
                      allowed a small nudge it can't invent a verdict from on its own.
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
              <span>{source === "f1" ? "F1 radio playback" : "default microphone"}</span>
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
              {turns.length === 0 && <span style={{ fontSize: 12, color: muted(45) }}>nothing scored yet</span>}
              {turns.map((t, i) => (
                <div key={i} title={t.mood || "unscored"} style={{
                  flex: 1,
                  height: t.mood === "stressed" ? 30 : t.mood === "tired" ? 22 : 14,
                  background: MOOD_COLOR[t.mood || ""] || muted(20),
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

          <div className="blueprint" style={{ padding: 13, display: "flex", flexDirection: "column", gap: 9 }}>
            <span className="kicker-sm" style={{ color: muted(55) }}>Speakers heard</span>
            {speakers.length === 0 ? (
              <span style={{ fontSize: 12, color: muted(45) }}>nothing matched yet</span>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {speakers.map(([name, { n, confident }]) => (
                  <div key={name} className="mono" style={{
                    display: "flex", alignItems: "center", justifyContent: "space-between", fontSize: 11.5,
                  }}>
                    <span style={{ color: "var(--color-text)" }}>{name}</span>
                    <span style={{ color: muted(55) }}>
                      {n} chunk{n === 1 ? "" : "s"}{confident > 0 ? ` · ${confident} matched` : " · suggested only"}
                    </span>
                  </div>
                ))}
              </div>
            )}
            {unmatched > 0 && (
              <span className="mono" style={{ fontSize: 11, color: muted(50) }}>
                {unmatched} chunk{unmatched === 1 ? "" : "s"} with speech but no profile match
              </span>
            )}
            <span style={{ fontSize: 11, lineHeight: 1.5, color: muted(50) }}>
              Each turn is diarized and matched independently, with no memory across
              chunks — not a session-wide identity. Enroll a voice in Speakers to make it
              matchable here.
            </span>
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
            Each chunk is diarized and matched against enrolled profiles on its own, with no
            memory of earlier chunks — a lead, not the multi-clip identification a full
            upload gets.
          </p>
        </aside>
      </div>

      <Dialog
        open={tagging != null}
        onClose={() => { setTagging(null); setTagName(""); }}
        kicker="Live"
        title="Name this speaker"
        subject={tagging?.text ? `"${tagging.text.slice(0, 80)}${tagging.text.length > 80 ? "…" : ""}"` : ""}
        consequence="Enrolls into an existing profile of this name, or creates one. Every later chunk in any live session can then match this voice."
        confirm="Tag speaker"
        width="480px"
        busy={tagBusy}
        confirmDisabled={!tagName.trim()}
        onConfirm={confirmTag}
      >
        <div className="field">
          <label>Speaker name</label>
          <input
            className="input" value={tagName} autoFocus
            onChange={(e) => setTagName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && confirmTag()}
            placeholder="e.g. Lewis Hamilton"
          />
        </div>
        <p style={{ margin: 0, fontSize: 12.5, color: muted(60) }}>
          One short live turn is a thinner signal than a full clip's enrollment — the profile
          starts provisional either way, same as naming a voice anywhere else in the app.
        </p>
      </Dialog>
    </div>
  );
}
