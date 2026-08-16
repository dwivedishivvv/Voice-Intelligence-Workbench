import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { api } from "@/api/client";
import { AMBER_INK, GREEN, RED, muted, tagTone } from "@/lib/ui";

type ModelOption = {
  repo_id: string; dir_name: string; label: string; note: string;
  active: boolean; status: "not_downloaded" | "downloading" | "downloaded" | "error";
  error: string | null;
};
type CategoryInfo = { gated: boolean; options: ModelOption[] };
type ModelsData = Record<string, CategoryInfo>;

const CATEGORY_LABELS: Record<string, string> = {
  asr: "Speech recognition",
  diarization: "Speaker diarization",
  embedding: "Speaker embeddings",
  text_embedding: "Text embeddings (search)",
};

export function ModelsPanel() {
  const [data, setData] = useState<ModelsData | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval>>();

  async function load() {
    const d: ModelsData = await api.listModels();
    setData(d);
    const downloading = (x: ModelsData) =>
      Object.values(x).some((c) => c.options.some((o) => o.status === "downloading"));
    if (downloading(d) && !pollRef.current) {
      pollRef.current = setInterval(async () => {
        const fresh: ModelsData = await api.listModels();
        setData(fresh);
        if (!downloading(fresh)) {
          clearInterval(pollRef.current);
          pollRef.current = undefined;
        }
      }, 2000);
    }
  }
  useEffect(() => { load().catch(() => {}); return () => clearInterval(pollRef.current); }, []);

  async function pull(category: string, repoId: string) {
    await api.pullModel(category, repoId);
    toast.info("Download started", { description: repoId });
    load();
  }

  async function activate(category: string, repoId: string, label: string) {
    await api.activateModel(category, repoId);
    toast.success(`${label} set as pending`, { description: "Takes effect after the worker restarts." });
    load();
  }

  if (!data) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, maxWidth: 900 }}>
      {Object.entries(data).map(([category, info]) => (
        <ModelCategory key={category} category={category} info={info} onPull={pull} onActivate={activate} />
      ))}
    </div>
  );
}

function ModelCategory({ category, info, onPull, onActivate }: {
  category: string; info: CategoryInfo;
  onPull: (category: string, repoId: string) => void;
  onActivate: (category: string, repoId: string, label: string) => void;
}) {
  const active = info.options.find((o) => o.active);
  const [selected, setSelected] = useState(active?.repo_id || info.options[0]?.repo_id);
  const opt = info.options.find((o) => o.repo_id === selected) || info.options[0];
  const warn = tagTone("warn");

  return (
    <section className="blueprint" style={{ padding: 14, display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12 }}>
        <span className="kicker-sm" style={{ color: muted(55) }}>
          {CATEGORY_LABELS[category] || category}
        </span>
        {info.gated && (
          <span className="tag" style={{ border: `1px solid ${warn.border}`, color: warn.color, fontSize: 10 }}>
            gated
          </span>
        )}
      </div>
      <span className="mono" style={{ fontSize: 12, color: muted(60) }}>
        active · {active?.label || "unknown"}
      </span>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <select className="input mono" style={{ flex: 1 }} value={selected}
                onChange={(e) => setSelected(e.target.value)}>
          {info.options.map((o) => (
            <option key={o.repo_id} value={o.repo_id}>
              {o.label}{o.active ? " · active" : ""}
            </option>
          ))}
        </select>
        {opt && <ModelAction opt={opt} category={category} onPull={onPull} onActivate={onActivate} />}
      </div>
      {opt?.note && <span style={{ fontSize: 12.5, color: muted(62) }}>{opt.note}</span>}
      {opt?.status === "error" && (
        <span style={{ fontSize: 12.5, color: RED }}>{opt.error}</span>
      )}
    </section>
  );
}

function ModelAction({ opt, category, onPull, onActivate }: {
  opt: ModelOption; category: string;
  onPull: (category: string, repoId: string) => void;
  onActivate: (category: string, repoId: string, label: string) => void;
}) {
  if (opt.active) {
    return (
      <span className="tag mono" style={{ border: `1px solid ${GREEN}`, color: GREEN, flex: "none" }}>
        ✓ active
      </span>
    );
  }
  if (opt.status === "downloading") {
    return <button className="btn btn-secondary" disabled style={{ flex: "none" }}>Downloading…</button>;
  }
  if (opt.status === "downloaded") {
    return (
      <button className="btn btn-primary" style={{ flex: "none" }}
              onClick={() => onActivate(category, opt.repo_id, opt.label)}>
        Activate
      </button>
    );
  }
  return (
    <button className="btn btn-secondary" style={{ flex: "none", color: opt.status === "error" ? AMBER_INK : undefined }}
            onClick={() => onPull(category, opt.repo_id)}>
      {opt.status === "error" ? "Retry" : "Download"}
    </button>
  );
}
