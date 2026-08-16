/** Shared vocabulary for the Industry design system: the colours the workbench assigns
 *  to identification outcomes, voices and tone readings, plus the formatters used in the
 *  monospace metric strips. Colours are CSS custom properties, so every one of them
 *  follows the light/dark theme automatically. */

export const RED = "var(--sig-red)";
export const AMBER = "var(--sig-amber)";
export const AMBER_INK = "var(--sig-amber-ink)";
export const GREEN = "var(--sig-green)";
export const VOICE_A = "var(--sig-voice-a)";
export const VOICE_B = "var(--sig-voice-b)";
export const TEXT = "var(--color-text)";
export const ACCENT = "var(--color-accent)";

/** Text at a fraction of the ink colour — the mockup's one muting idiom. */
export const muted = (pct: number) =>
  `color-mix(in srgb, var(--color-text) ${pct}%, transparent)`;

export const MUTED = muted(55);

/** Identification outcomes, as the pipeline records them on clip_speakers.match_result.
 *  Four distinct states, kept apart on purpose: a confident name, a proposal, a judged
 *  "nobody matches", and a refusal to judge at all. */
export type Outcome = "confident" | "suggested" | "unknown" | "abstained";

export const OUTCOME: Record<Outcome, {
  label: string; color: string; kicker: string; italic: boolean;
}> = {
  confident: { label: "Confident", color: VOICE_A, kicker: "var(--color-accent-700)", italic: false },
  suggested: { label: "Suggested", color: VOICE_B, kicker: "var(--color-accent-700)", italic: true },
  unknown: { label: "Unknown", color: "var(--color-neutral-600)", kicker: muted(60), italic: false },
  abstained: { label: "Abstained", color: AMBER, kicker: AMBER_INK, italic: false },
};

export const outcomeOf = (r: string | null | undefined): Outcome =>
  r === "confident" || r === "suggested" || r === "abstained" ? r : "unknown";

/** Stable per-voice colour. Voices are drawn from the two signal blues first — those two
 *  carry "named" and "suggested" everywhere else in the UI — then the neutral ramp. */
const VOICE_PALETTE = [VOICE_A, VOICE_B, "var(--color-accent-400)", "var(--color-neutral-500)",
                       "var(--color-accent-700)", "var(--color-neutral-600)"];

export function voiceColor(label: string) {
  let h = 0;
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) >>> 0;
  return VOICE_PALETTE[h % VOICE_PALETTE.length];
}

export type Mood = "calm" | "stressed" | "tired";
export const MOOD_COLOR: Record<string, string> = {
  calm: GREEN, stressed: RED, tired: AMBER,
};
/** How full the tone meter reads. Heuristic bands, not a measurement. */
export const MOOD_PCT: Record<string, string> = {
  calm: "30%", tired: "58%", stressed: "74%",
};

export const SENTIMENT_COLOR: Record<string, string> = {
  positive: GREEN, negative: RED, neutral: muted(45),
};

export const QUALITY_COLOR: Record<string, string> = {
  good: TEXT, fair: AMBER_INK, poor: RED,
};

/** Border/ink pair for a bordered tag. */
export const tagTone = (kind: "accent" | "neutral" | "warn" | "bad" | "good") => ({
  border:
    kind === "accent" ? "var(--color-accent-300)"
    : kind === "warn" ? AMBER
    : kind === "bad" ? RED
    : kind === "good" ? GREEN
    : "var(--color-neutral-300)",
  color:
    kind === "accent" ? "var(--color-accent-700)"
    : kind === "warn" ? AMBER_INK
    : kind === "bad" ? RED
    : kind === "good" ? GREEN
    : "var(--color-neutral-700)",
});

/** 01:24 — the clip-length form used across the tables. */
export function fmtDur(s: number | null | undefined) {
  if (!s && s !== 0) return "—";
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

/** 00:41.6 — the player/transcript form, tenths included so a seek is unambiguous. */
export function fmtTime(s: number | null | undefined) {
  if (s == null || !Number.isFinite(s)) return "00:00.0";
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${(s % 60).toFixed(1).padStart(4, "0")}`;
}

export function fmtSpeech(s: number | null | undefined) {
  if (!s) return "0s";
  const m = Math.floor(s / 60);
  return m ? `${m}m ${Math.round(s % 60)}s` : `${s.toFixed(1)}s`;
}

export function fmtAgo(iso: string | null | undefined) {
  if (!iso) return "—";
  const mins = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const h = Math.round(mins / 60);
  if (h < 24) return `${h} h ago`;
  return `${Math.round(h / 24)} d ago`;
}

export const pct = (v: number | null | undefined, of = 1) =>
  `${Math.round(((v || 0) / (of || 1)) * 100)}%`;

export const num = (v: number | null | undefined, d = 2) =>
  v == null ? "—" : v.toFixed(d);
