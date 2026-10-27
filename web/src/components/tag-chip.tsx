import { muted, tagTone } from "@/lib/ui";

/** Which tone an auto tag reads in. Auto tags are `family:value`; the colour carries the
 *  same meaning it does everywhere else — amber for a caveat on the analysis itself, accent
 *  for identity, green for clean audio. */
function toneFor(tag: string): Parameters<typeof tagTone>[0] {
  if (tag === "audio:clean") return "good";
  if (tag === "audio:poor" || tag === "asr:low-confidence" || tag === "crosstalk") return "warn";
  if (tag.startsWith("speaker:")) return "accent";
  return "neutral";
}

const HINT: Record<string, string> = {
  "asr:low-confidence": "Transcription confidence is low — invented text is most likely here",
  "crosstalk": "A large share of the speech overlaps another voice",
};

/** One tag. Operator tags are solid; pipeline tags are dashed and say so on hover, so a
 *  machine reading is never mistaken for something a person wrote down. */
export function TagChip({ tag, auto, active, onClick }: {
  tag: string; auto: boolean; active?: boolean; onClick?: () => void;
}) {
  const i = auto ? tag.indexOf(":") : -1;
  const family = i > 0 ? tag.slice(0, i) : "";
  const value = i > 0 ? tag.slice(i + 1) : tag;
  const tone = tagTone(auto ? toneFor(tag) : "neutral");
  return (
    <span
      className={auto ? "tag mono" : "tag tag-neutral mono"}
      title={(HINT[tag] ? `${HINT[tag]} · ` : "") + (auto ? "added by the pipeline" : "added by an operator")}
      onClick={onClick}
      style={{
        fontSize: 10.5, padding: "2px 7px", gap: 3, whiteSpace: "nowrap",
        cursor: onClick ? "pointer" : undefined,
        border: auto ? `1px dashed ${tone.border}` : "1px solid transparent",
        color: auto ? tone.color : undefined,
        outline: active ? "2px solid var(--color-accent)" : undefined,
        outlineOffset: -1,
      }}
    >
      {family && <span style={{ color: muted(45) }}>{family}</span>}
      {value}
    </span>
  );
}
