import { useEffect, useMemo, useRef, useState } from "react";
import { fmtTime, muted, outcomeOf, OUTCOME, voiceColor, type Outcome } from "@/lib/ui";

type Turn = { start_s: number; end_s: number; local_label: string; is_overlap?: boolean };
type Speaker = {
  local_label: string; display_name?: string | null;
  match_result?: string | null; talk_share?: number | null;
};

const BARS = 96;

/** Bar heights are a deterministic decorative envelope, not a waveform read off the file —
 *  the API serves audio, not peak data. Colour and position ARE real: each bar is coloured
 *  by the speaker turn that covers its slice of the timeline, and hatched where the
 *  pipeline declined to judge, so the picture never claims more than the pipeline knows. */
const HEIGHTS = Array.from({ length: BARS }, (_, i) =>
  Math.round(18 + Math.abs(Math.sin(i * 1.7) * 46) + (i % 3) * 6));

export function AudioPlayer({
  src, duration: durationHint, turns = [], speakers = [], onTimeUpdate, seekTo, right,
}: {
  src: string;
  duration?: number;
  turns?: Turn[];
  speakers?: Speaker[];
  onTimeUpdate?: (t: number) => void;
  seekTo?: { time: number; nonce: number } | null;
  right?: React.ReactNode;
}) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const waveRef = useRef<HTMLDivElement>(null);
  const [playing, setPlaying] = useState(false);
  const [current, setCurrent] = useState(0);
  // durationHint (clip.duration_s) avoids a 00:00 flash before <audio> loads its metadata.
  const [loadedDuration, setLoadedDuration] = useState(0);
  const duration = loadedDuration || durationHint || 0;

  useEffect(() => {
    if (seekTo == null || !audioRef.current) return;
    audioRef.current.currentTime = seekTo.time;
    setCurrent(seekTo.time);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seekTo?.nonce]);

  const outcomeOfLabel = useMemo(() => {
    const m = new Map<string, Outcome>();
    for (const s of speakers) m.set(s.local_label, outcomeOf(s.match_result));
    return m;
  }, [speakers]);

  const bars = useMemo(() => {
    return HEIGHTS.map((h, i) => {
      const t = duration ? ((i + 0.5) / BARS) * duration : 0;
      const turn = turns.find((x) => t >= x.start_s && t < x.end_s);
      if (!turn) return { h, color: muted(12), hatch: false };
      const judged = outcomeOfLabel.get(turn.local_label) !== "abstained";
      return {
        h,
        color: judged ? voiceColor(turn.local_label) : muted(18),
        hatch: !judged,
      };
    });
  }, [turns, duration, outcomeOfLabel]);

  function seekFromClientX(clientX: number) {
    const el = waveRef.current, audio = audioRef.current;
    if (!el || !audio || !duration) return;
    const rect = el.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    audio.currentTime = frac * duration;
    setCurrent(frac * duration);
    onTimeUpdate?.(frac * duration);
  }

  const playedPct = duration ? (current / duration) * 100 : 0;

  return (
    <section className="blueprint" style={{ padding: "16px 18px" }}>
      <audio
        ref={audioRef}
        src={src}
        preload="metadata"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onLoadedMetadata={(e) => setLoadedDuration(e.currentTarget.duration)}
        onTimeUpdate={(e) => {
          setCurrent(e.currentTarget.currentTime);
          onTimeUpdate?.(e.currentTarget.currentTime);
        }}
        // Chrome will not load media data for a display:none element; this stays in flow.
        style={{ position: "absolute", width: 1, height: 1, opacity: 0, pointerEvents: "none" }}
      />

      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        gap: 16, marginBottom: 14,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <button
            className="btn btn-primary btn-icon"
            style={{ width: 38, height: 38 }}
            aria-label={playing ? "Pause" : "Play"}
            onClick={() => (playing ? audioRef.current?.pause() : audioRef.current?.play())}
          >
            {playing ? "▮▮" : "▶"}
          </button>
          <span className="mono" style={{ fontSize: 13 }}>
            {fmtTime(current)} <span style={{ color: muted(45) }}>/ {fmtTime(duration)}</span>
          </span>
        </div>
        {right}
      </div>

      <div
        ref={waveRef}
        onMouseDown={(e) => seekFromClientX(e.clientX)}
        style={{
          display: "flex", alignItems: "flex-end", gap: 2, height: 88,
          position: "relative", cursor: duration ? "pointer" : "default",
        }}
      >
        {bars.map((b, i) => (
          <div
            key={i}
            className={`wf-seg${b.hatch ? " hatch" : ""}`}
            style={{
              flex: 1, height: b.h,
              // the hatch is a background *image*, so a faint fill under it keeps an
              // unjudged stretch reading as bars rather than as a hole in the waveform
              background: b.hatch ? muted(10) : b.color,
              borderTop: b.hatch ? `1px solid ${muted(35)}` : undefined,
            }}
          />
        ))}
        <div style={{
          position: "absolute", left: `${playedPct}%`, top: -6, bottom: -6,
          width: 1, background: "var(--color-text)", pointerEvents: "none",
        }} />
      </div>

      {speakers.length > 0 && (
        <div style={{ display: "flex", gap: 18, marginTop: 14, flexWrap: "wrap", fontSize: 12 }}>
          {speakers.map((s) => {
            const outcome = outcomeOf(s.match_result);
            return (
              <span key={s.local_label} style={{ display: "flex", alignItems: "center", gap: 7 }}>
                <span style={{ width: 11, height: 11, background: voiceColor(s.local_label) }} />
                <span style={{ fontStyle: OUTCOME[outcome].italic ? "italic" : "normal" }}>
                  {s.display_name || s.local_label}{outcome === "suggested" ? "?" : ""}
                </span>
                <span className="mono" style={{ color: muted(45) }}>
                  {s.talk_share != null ? `${Math.round(s.talk_share * 100)}%` : ""}
                </span>
              </span>
            );
          })}
          {speakers.some((s) => outcomeOf(s.match_result) === "abstained") && (
            <span style={{ display: "flex", alignItems: "center", gap: 7, marginLeft: "auto" }}>
              <span className="hatch" style={{ width: 11, height: 11, border: `1px solid ${muted(35)}` }} />
              <span style={{ color: muted(60) }}>not judged</span>
            </span>
          )}
        </div>
      )}
    </section>
  );
}
