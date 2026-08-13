export type Mood = "calm" | "stressed" | "tired";

export const MOOD_STYLE: Record<Mood, string> = {
  calm: "bg-success/15 text-success border-success/30",
  stressed: "bg-destructive/15 text-destructive border-destructive/30",
  tired: "bg-warning/15 text-warning border-warning/30",
};

// dot/bar-only variant (first class of MOOD_STYLE) for compact indicators
export function moodDot(mood: Mood): string {
  return MOOD_STYLE[mood].split(" ")[0];
}
