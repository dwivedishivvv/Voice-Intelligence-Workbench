// Tailwind can only pick up class names it can see as complete static strings in source,
// so these must be spelled out in full here rather than built with template literals.
const PALETTE = [
  { dot: "bg-chart-1", chip: "bg-chart-1/20 text-chart-1", ring: "border-chart-1/40" },
  { dot: "bg-chart-2", chip: "bg-chart-2/20 text-chart-2", ring: "border-chart-2/40" },
  { dot: "bg-chart-3", chip: "bg-chart-3/20 text-chart-3", ring: "border-chart-3/40" },
  { dot: "bg-chart-4", chip: "bg-chart-4/20 text-chart-4", ring: "border-chart-4/40" },
  { dot: "bg-chart-5", chip: "bg-chart-5/20 text-chart-5", ring: "border-chart-5/40" },
];

export function speakerColor(label: string) {
  let h = 0;
  for (let i = 0; i < label.length; i++) h = (h * 31 + label.charCodeAt(i)) >>> 0;
  return PALETTE[h % PALETTE.length];
}
