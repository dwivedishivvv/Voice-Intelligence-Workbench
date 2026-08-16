import { useEffect, useRef, useState } from "react";
import { muted } from "@/lib/ui";

const API_KEY = () => localStorage.getItem("api_key") || "change-me";

/** One excerpt at a time, across every card on the page. Two quotes talking over each
 *  other is the same problem the montage player avoids, and here it is worse: the whole
 *  point of playing a citation is hearing whether it says what the answer claims. */
const stoppers = new Set<() => void>();

const clock = (s: number) =>
  `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/** Plays the cited seconds of a clip, in place.
 *
 *  The audio route serves ranges (206), so seeking to the quote fetches only the bytes
 *  around it rather than the whole file. Checking a citation should cost a click, not a
 *  page navigation that loses the answer you were reading. */
export function ExcerptPlayer({ clipId, startS, endS }: {
  clipId: string;
  startS: number | null;
  endS?: number | null;
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [pos, setPos] = useState<number | null>(null);

  const from = startS ?? 0;
  // Utterance bounds when the corpus has them; otherwise a bounded window, so a citation
  // without an end does not quietly play the rest of the recording.
  const to = endS && endS > from ? endS : from + 20;
  const span = Math.max(0.1, to - from);

  const stop = () => {
    audioRef.current?.pause();
    setPlaying(false);
  };

  useEffect(() => {
    stoppers.add(stop);
    return () => {
      stoppers.delete(stop);
      audioRef.current?.pause();
      audioRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function toggle() {
    if (playing) return stop();
    for (const other of stoppers) if (other !== stop) other();

    let el = audioRef.current;
    if (!el) {
      el = new Audio(`/v1/clips/${clipId}/audio?variant=work&key=${API_KEY()}`);
      el.preload = "none";
      el.onended = () => setPlaying(false);
      el.ontimeupdate = () => {
        setPos(el!.currentTime);
        if (el!.currentTime >= to) stop();
      };
      audioRef.current = el;
    }
    // Re-seek on every press: after the excerpt ends the playhead sits past `to`, and
    // pressing play again should replay the quote rather than run on into the next one.
    if (el.currentTime < from || el.currentTime >= to - 0.05) el.currentTime = from;
    try {
      await el.play();
      setPlaying(true);
    } catch {
      setPlaying(false);
    }
  }

  const played = pos == null ? 0 : Math.min(1, Math.max(0, (pos - from) / span));

  return (
    <span style={{ display: "flex", alignItems: "center", gap: 10, minWidth: 0 }}>
      <button
        className="btn btn-secondary btn-icon"
        style={{ width: 26, height: 26, flex: "none", fontSize: 11 }}
        aria-label={playing ? "Pause excerpt" : "Play excerpt"}
        onClick={toggle}
      >
        {playing ? "▮▮" : "▶"}
      </button>
      <span
        onClick={toggle}
        title={`${clock(from)}–${clock(to)} of this clip`}
        style={{
          width: 130, height: 5, background: muted(12), flex: "none", cursor: "pointer",
        }}
      >
        <span style={{
          display: "block", height: "100%", width: `${played * 100}%`,
          background: "var(--color-accent)",
        }} />
      </span>
      <span className="mono" style={{ fontSize: 11.5, color: muted(55), whiteSpace: "nowrap" }}>
        {playing || pos != null
          ? `${clock(Math.max(from, pos ?? from))} / ${clock(to)}`
          : `plays ${clock(from)}`}
      </span>
    </span>
  );
}
