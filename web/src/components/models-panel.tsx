import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { api } from "@/api/client";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Download, Loader2, CheckCircle2, AlertTriangle, Lock, RotateCw } from "lucide-react";

type ModelOption = {
  repo_id: string; dir_name: string; label: string; note: string;
  active: boolean; status: "not_downloaded" | "downloading" | "downloaded" | "error"; error: string | null;
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
    const anyDownloading = Object.values(d).some((c) => c.options.some((o) => o.status === "downloading"));
    if (anyDownloading && !pollRef.current) {
      pollRef.current = setInterval(async () => {
        const fresh: ModelsData = await api.listModels();
        setData(fresh);
        if (!Object.values(fresh).some((c) => c.options.some((o) => o.status === "downloading"))) {
          clearInterval(pollRef.current);
          pollRef.current = undefined;
        }
      }, 2000);
    }
  }
  useEffect(() => { load(); return () => clearInterval(pollRef.current); }, []);

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
    <div className="space-y-6">
      {Object.entries(data).map(([category, info]) => (
        <ModelCategoryCard
          key={category}
          category={category}
          info={info}
          onPull={pull}
          onActivate={activate}
        />
      ))}
    </div>
  );
}

function ModelCategoryCard({ category, info, onPull, onActivate }: {
  category: string; info: CategoryInfo;
  onPull: (category: string, repoId: string) => void;
  onActivate: (category: string, repoId: string, label: string) => void;
}) {
  const active = info.options.find((o) => o.active);
  const [selected, setSelected] = useState(active?.repo_id || info.options[0]?.repo_id);
  const opt = info.options.find((o) => o.repo_id === selected) || info.options[0];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          {CATEGORY_LABELS[category] || category}
          {info.gated && (
            <Badge variant="outline" className="gap-1 border-warning/30 bg-warning/10 text-[10px] text-warning">
              <Lock className="size-2.5" /> gated
            </Badge>
          )}
        </CardTitle>
        <CardDescription>Currently active: <span className="font-mono">{active?.label || "unknown"}</span></CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-3">
          <Select value={selected} onValueChange={setSelected}>
            <SelectTrigger className="flex-1">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {info.options.map((o) => (
                <SelectItem key={o.repo_id} value={o.repo_id}>
                  <div className="flex items-center gap-2">
                    {o.label}
                    {o.active && <Badge variant="secondary" className="text-[10px]">active</Badge>}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {opt && <ModelAction opt={opt} category={category} onPull={onPull} onActivate={onActivate} />}
        </div>
        {opt && <p className="mt-2 text-xs text-muted-foreground">{opt.note}</p>}
        {opt?.status === "error" && (
          <p className="mt-2 flex items-center gap-1.5 text-xs text-destructive">
            <AlertTriangle className="size-3" /> {opt.error}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function ModelAction({ opt, category, onPull, onActivate }: {
  opt: ModelOption; category: string;
  onPull: (category: string, repoId: string) => void;
  onActivate: (category: string, repoId: string, label: string) => void;
}) {
  if (opt.active) {
    return <Badge variant="outline" className="gap-1.5 border-success/30 bg-success/10 text-success"><CheckCircle2 className="size-3" /> Active</Badge>;
  }
  if (opt.status === "downloading") {
    return <Button disabled size="sm" className="gap-1.5"><Loader2 className="size-3.5 animate-spin" /> Downloading…</Button>;
  }
  if (opt.status === "downloaded") {
    return (
      <Button size="sm" className="gap-1.5" onClick={() => onActivate(category, opt.repo_id, opt.label)}>
        <CheckCircle2 className="size-3.5" /> Activate
      </Button>
    );
  }
  if (opt.status === "error") {
    return (
      <Button size="sm" variant="outline" className="gap-1.5" onClick={() => onPull(category, opt.repo_id)}>
        <RotateCw className="size-3.5" /> Retry
      </Button>
    );
  }
  return (
    <Button size="sm" variant="outline" className="gap-1.5" onClick={() => onPull(category, opt.repo_id)}>
      <Download className="size-3.5" /> Download
    </Button>
  );
}
