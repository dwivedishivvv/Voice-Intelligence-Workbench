import { useEffect, useRef, useState } from "react";
import { speakerColor } from "@/lib/speaker-color";
import { cn } from "@/lib/utils";
import { Play, Pause, Volume2, VolumeX } from "lucide-react";

type Turn = { start_s: number; end_s: number; local_label: string; is_overlap: boolean };
type Speaker = { local_label: string; display_name?: string | null };

const RATES = [0.75, 1, 1.25, 1.5, 2];

function fmtTime(s: number) {
  if (!Number.isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export function AudioPlayer({
  src, duration: durationHint, turns = [], speakers = [], onTimeUpdate, seekTo,
}: {
  src: string; duration?: number; turns?: Turn[]; speakers?: Speaker[];
  onTimeUpdate?: (t: number) => void;
  seekTo?: { time: number; nonce: number } | null;
}) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const [playing, setPlaying] = useState(false);
  const [current, setCurrent] = useState(0);
  const [rate, setRate] = useState(1);
  const [muted, setMuted] = useState(false);
  const [scrubbing, setScrubbing] = useState(false);
  // durationHint (e.g. clip.duration_s from the API) avoids a "0:00" flash before the
  // <audio> element itself finishes loading metadata — once it has, its real value wins.
  const [loadedDuration, setLoadedDuration] = useState(0);
  const duration = loadedDuration || durationHint || 0;

  useEffect(() => {
    if (seekTo == null || !audioRef.current) return;
    audioRef.current.currentTime = seekTo.time;
    setCurrent(seekTo.time);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seekTo?.nonce]);

  function seekFromClientX(clientX: number) {
    const el = trackRef.current;
    const audio = audioRef.current;
    if (!el || !audio || !duration) return;
    const rect = el.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
    const t = frac * duration;
    audio.currentTime = t;
    setCurrent(t);
  }

  const displayName = (label: string) => speakers.find((s) => s.local_label === label)?.display_name || label;

  return (
    <div className="relative rounded-xl border border-border bg-card p-4">
      <audio
        ref={audioRef}
        src={src}
        preload="metadata"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onTimeUpdate={(e) => {
          if (scrubbing) return;
          const t = e.currentTarget.currentTime;
          setCurrent(t);
          onTimeUpdate?.(t);
        }}
        onEnded={() => setPlaying(false)}
        onLoadedMetadata={(e) => setLoadedDuration(e.currentTarget.duration)}
        // Chrome throttles/never loads media data for display:none elements (Tailwind's
        // `hidden`) — this stays in normal flow but is visually and interactively invisible.
        style={{ position: "absolute", width: 1, height: 1, opacity: 0, pointerEvents: "none" }}
      />

      <div className="flex items-center gap-3">
        <button
          onClick={() => (playing ? audioRef.current?.pause() : audioRef.current?.play())}
          className="flex size-10 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground transition-transform hover:scale-105 active:scale-95"
        >
          {playing ? <Pause className="size-4.5 fill-current" /> : <Play className="ml-0.5 size-4.5 fill-current" />}
        </button>

        <div className="flex-1">
          <div
            ref={trackRef}
            className="group relative h-11 cursor-pointer select-none rounded-lg bg-muted"
            onMouseDown={(e) => { setScrubbing(true); seekFromClientX(e.clientX); }}
            onMouseMove={(e) => { if (e.buttons === 1 && scrubbing) seekFromClientX(e.clientX); }}
            onMouseUp={(e) => { seekFromClientX(e.clientX); setScrubbing(false); onTimeUpdate?.(audioRef.current?.currentTime || 0); }}
            onMouseLeave={() => setScrubbing(false)}
          >
            {/* speaker segments */}
            {duration > 0 && turns.map((t, i) => {
              const color = speakerColor(t.local_label);
              const left = (t.start_s / duration) * 100;
              const width = Math.max(0.3, ((t.end_s - t.start_s) / duration) * 100);
              return (
                <div
                  key={i}
                  title={`${displayName(t.local_label)} · ${fmtTime(t.start_s)}–${fmtTime(t.end_s)}`}
                  className={cn(
                    "absolute top-1.5 bottom-1.5 rounded-sm opacity-70 transition-opacity group-hover:opacity-85",
                    color.dot,
                    t.is_overlap && "ring-2 ring-warning"
                  )}
                  style={{ left: `${left}%`, width: `${width}%` }}
                />
              );
            })}

            {/* played progress overlay */}
            <div
              className="pointer-events-none absolute inset-y-0 left-0 rounded-l-lg bg-white/10"
              style={{ width: `${duration ? (current / duration) * 100 : 0}%` }}
            />

            {/* playhead */}
            <div
              className="pointer-events-none absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_6px_rgba(255,255,255,0.8)]"
              style={{ left: `${duration ? (current / duration) * 100 : 0}%` }}
            />
          </div>

          <div className="mt-1.5 flex items-center justify-between text-xs tabular-nums text-muted-foreground">
            <span>{fmtTime(current)}</span>
            <span>{fmtTime(duration)}</span>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1">
          <button
            onClick={() => setRate((r) => RATES[(RATES.indexOf(r) + 1) % RATES.length])}
            className="rounded-md px-2 py-1 text-xs font-medium text-muted-foreground tabular-nums hover:bg-accent hover:text-foreground"
            title="Playback speed"
          >
            {rate}x
          </button>
          <button
            onClick={() => setMuted((m) => !m)}
            className="flex size-8 items-center justify-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground"
          >
            {muted ? <VolumeX className="size-4" /> : <Volume2 className="size-4" />}
          </button>
        </div>
      </div>

      {speakers.length > 0 && (
        <div className="mt-3 flex flex-wrap items-center gap-3 border-t border-border pt-3">
          {speakers.map((s) => (
            <div key={s.local_label} className="flex items-center gap-1.5 text-xs text-muted-foreground">
              <span className={cn("size-2 rounded-full", speakerColor(s.local_label).dot)} />
              {s.display_name || s.local_label}
            </div>
          ))}
        </div>
      )}

      {/* keep native element's rate/mute in sync without re-render churn */}
      <RateSync audioRef={audioRef} rate={rate} muted={muted} />
    </div>
  );
}

function RateSync({ audioRef, rate, muted }: {
  audioRef: React.RefObject<HTMLAudioElement>; rate: number; muted: boolean;
}) {
  useEffect(() => { if (audioRef.current) audioRef.current.playbackRate = rate; }, [rate, audioRef]);
  useEffect(() => { if (audioRef.current) audioRef.current.muted = muted; }, [muted, audioRef]);
  return null;
}
