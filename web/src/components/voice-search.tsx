import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/api/client";
import { AMBER, GREEN, RED, fmtTime, muted, num } from "@/lib/ui";

type Mode = "both" | "speech" | "speaker";
type VoiceMatch = {
  clip_id: string; local_label: string; filename: string; display_name: string | null;
  sim: number; band: "likely" | "possible" | "unlikely"; speech_s: number; first_s: number | null;
};
type Result = {
  heard: string; duration_s: number; speech_s: number; reliability: number;
  speech?: { utterance_id: string; clip_id: string; filename: string | null; start_s: number; text: string }[];
  speaker?: { who: { name: string; result: string; score: number } | null; result?: string;
              matches: VoiceMatch[]; note?: string };
};

// The worker caps a query at 30s too; stopping here means nothing recorded is thrown away.
const MAX_RECORD_S = 30;
const BAND_COLOR = { likely: GREEN, possible: AMBER, unlikely: muted(40) } as const;

/** Search by speaking. Records from the mic (or takes an audio file), sends it to the
 *  worker to transcribe and fingerprint, and shows both answers: where those words were
 *  said, and where that voice speaks. Similarity bands are identification's own thresholds,
 *  so "likely" here means the same thing it means on a clip page. */
export function VoiceSearch() {
  const navigate = useNavigate();
  const [mode, setMode] = useState<Mode>("both");
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<Result | null>(null);
  const recRef = useRef<MediaRecorder | null>(null);
  const timerRef = useRef<number | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => () => { recRef.current?.stop(); if (timerRef.current) clearInterval(timerRef.current); }, []);

  async function send(blob: Blob, name: string) {
    setBusy(true);
    setResult(null);
    try {
      setResult(await api.voiceSearch(blob, mode, name));
    } catch (e) {
      toast.error("Voice search failed", { description: String((e as Error).message) });
    } finally { setBusy(false); }
  }

  async function start() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      rec.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        if (timerRef.current) clearInterval(timerRef.current);
        setRecording(false);
        const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
        if (blob.size) send(blob, "query.webm");
      };
      recRef.current = rec;
      rec.start();
      setRecording(true);
      setElapsed(0);
      const t0 = Date.now();
      timerRef.current = window.setInterval(() => {
        const s = (Date.now() - t0) / 1000;
        setElapsed(s);
        if (s >= MAX_RECORD_S) rec.stop();
      }, 200);
    } catch (e) {
      toast.error("Microphone unavailable", { description: String((e as Error).message) });
    }
  }

  const stop = () => recRef.current?.state === "recording" && recRef.current.stop();

  return (
    <section className="blueprint" style={{ padding: "14px 15px", display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <button className={`btn ${recording ? "btn-secondary" : "btn-primary"}`} disabled={busy}
                onClick={recording ? stop : start} style={recording ? { color: RED } : undefined}>
          {recording ? `■ Stop · ${elapsed.toFixed(0)}s` : "● Speak a query"}
        </button>
        <button className="btn btn-secondary" disabled={busy || recording} onClick={() => fileRef.current?.click()}>
          Use an audio file
        </button>
        <input ref={fileRef} type="file" accept="audio/*" hidden
               onChange={(e) => { const f = e.target.files?.[0]; if (f) send(f, f.name); e.target.value = ""; }} />
        <div className="seg">
          {([["both", "Words + voice"], ["speech", "What was said"], ["speaker", "Who is speaking"]] as const).map(([v, l]) => (
            <label key={v} className="seg-opt">
              <input type="radio" name="vmode" checked={mode === v} onChange={() => setMode(v)} />{l}
            </label>
          ))}
        </div>
        <span style={{ fontSize: 12, color: muted(55) }}>
          {busy ? "Transcribing and fingerprinting on the worker…" : `Up to ${MAX_RECORD_S}s. Speech stays on this machine.`}
        </span>
      </div>

      {result && (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div className="mono" style={{ fontSize: 12, color: muted(62) }}>
            HEARD · {result.heard ? `“${result.heard}”` : "no words recognised"} · {num(result.speech_s, 1)}s of speech
          </div>

          {result.speaker && (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <span className="kicker-sm" style={{ color: muted(55) }}>Who is speaking</span>
              <span style={{ fontSize: 15 }}>
                {result.speaker.note
                  ? result.speaker.note
                  : result.speaker.who
                    ? <>Sounds like <b>{result.speaker.who.name}</b>{result.speaker.who.result === "suggested" ? "?" : ""}{" "}
                        <span className="mono" style={{ fontSize: 12, color: muted(55) }}>
                          ({result.speaker.who.result}, {num(result.speaker.who.score)})</span></>
                    : <span style={{ color: muted(62) }}>
                        No enrolled profile matches this voice{result.speaker.result === "abstained" ? " — too little clean speech to judge" : ""}.
                        Nearest voices in the corpus:
                      </span>}
              </span>
              {result.speaker.matches.map((m) => (
                <a key={`${m.clip_id}:${m.local_label}`} href={`/clips/${m.clip_id}?t=${m.first_s ?? 0}`}
                   onClick={(e) => { e.preventDefault(); navigate(`/clips/${m.clip_id}?t=${m.first_s ?? 0}`); }}
                   style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) 150px 70px", gap: 12,
                            alignItems: "center", padding: "6px 0", borderBottom: `1px solid ${muted(8)}`,
                            color: "inherit", textDecoration: "none" }}>
                  <span style={{ fontSize: 13, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {m.filename} · <span className="mono" style={{ fontSize: 11.5 }}>
                      {m.display_name || m.local_label.replace("SPEAKER_", "Voice ")}</span>
                  </span>
                  <span className="bar"><span style={{ width: `${Math.round(Math.max(0, m.sim) * 100)}%`,
                                                       background: BAND_COLOR[m.band] }} /></span>
                  <span className="mono" style={{ fontSize: 11, textAlign: "right", color: BAND_COLOR[m.band] }}>
                    {m.band} {num(m.sim)}
                  </span>
                </a>
              ))}
            </div>
          )}

          {result.speech && (
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              <span className="kicker-sm" style={{ color: muted(55) }}>Where those words were said</span>
              {result.speech.length === 0 && (
                <span style={{ fontSize: 13, color: muted(55) }}>No transcript line matched.</span>
              )}
              {result.speech.map((h) => (
                <a key={h.utterance_id} href={`/clips/${h.clip_id}?t=${h.start_s}`}
                   onClick={(e) => { e.preventDefault(); navigate(`/clips/${h.clip_id}?t=${h.start_s}`); }}
                   style={{ display: "grid", gridTemplateColumns: "minmax(0,200px) 60px 1fr", gap: 12,
                            padding: "6px 0", borderBottom: `1px solid ${muted(8)}`, color: "inherit", textDecoration: "none" }}>
                  <span className="mono" style={{ fontSize: 12, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {h.filename}</span>
                  <span className="mono" style={{ fontSize: 12, color: muted(55) }}>{fmtTime(h.start_s)}</span>
                  <span style={{ fontSize: 13.5 }}>{h.text}</span>
                </a>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
