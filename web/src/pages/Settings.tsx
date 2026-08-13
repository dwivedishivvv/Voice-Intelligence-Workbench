import { useEffect, useState, useCallback, useRef } from "react";
import { toast } from "sonner";
import { api } from "@/api/client";
import { PageHeader } from "@/components/page-header";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ModelsPanel } from "@/components/models-panel";
import { Lock, RotateCcw, SlidersHorizontal, Cpu } from "lucide-react";

type Field = {
  key: string; value: boolean | number | string; default: boolean | number | string;
  overridden: boolean; updated_at: string | null; type: "bool" | "int" | "float" | "str";
};
type Category = { name: string; fields: Field[] };
type RestartField = { key: string; value: any; pending: any };
type SettingsView = { categories: Category[]; restart_required: RestartField[] };

function humanize(key: string) {
  return key.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase())
    .replace(/\bId\b/g, "ID").replace(/\bSnr\b/g, "SNR").replace(/\bHz\b/g, "Hz")
    .replace(/\bMb\b/g, "MB").replace(/\bMs\b/g, "ms").replace(/\bS\b/g, "(s)")
    .replace(/\bLufs\b/g, "LUFS");
}

export default function Settings() {
  const [data, setData] = useState<SettingsView | null>(null);

  const load = useCallback(async () => setData(await api.getSettings()), []);
  useEffect(() => { load(); }, [load]);

  async function save(key: string, value: boolean | number | string) {
    try {
      await api.patchSettings({ [key]: value });
      toast.success(`${humanize(key)} updated`, { description: "Takes effect on the next job." });
      load();
    } catch (e) {
      toast.error(`Couldn't update ${humanize(key)}`, { description: String(e) });
    }
  }

  async function reset(key: string) {
    await api.resetSetting(key);
    toast.success(`${humanize(key)} reset to default`);
    load();
  }

  if (!data) {
    return (
      <div>
        <PageHeader title="Settings" description="Tune the processing pipeline without a restart." />
        <div className="space-y-3 px-8 py-6">
          {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-14 w-full rounded-lg" />)}
        </div>
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title="Settings"
        description="Tune the processing pipeline live — changes apply to the next job, no restart required."
        actions={
          <Badge variant="outline" className="gap-1.5">
            <SlidersHorizontal className="size-3" />
            {data.categories.reduce((n, c) => n + c.fields.filter((f) => f.overridden).length, 0)} overridden
          </Badge>
        }
      />

      <div className="px-8 py-6">
        <Tabs defaultValue={data.categories[0]?.name} className="gap-6">
          <TabsList className="h-auto flex-wrap justify-start bg-transparent p-0">
            {data.categories.map((c) => (
              <TabsTrigger key={c.name} value={c.name} className="data-[state=active]:bg-primary/10 data-[state=active]:text-primary">
                {c.name}
              </TabsTrigger>
            ))}
            <TabsTrigger value="models" className="data-[state=active]:bg-primary/10 data-[state=active]:text-primary">
              <Cpu className="mr-1 size-3" /> Models
            </TabsTrigger>
            <TabsTrigger value="system" className="data-[state=active]:bg-primary/10 data-[state=active]:text-primary">
              <Lock className="mr-1 size-3" /> System
            </TabsTrigger>
          </TabsList>

          {data.categories.map((cat) => (
            <TabsContent key={cat.name} value={cat.name}>
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">{cat.name}</CardTitle>
                </CardHeader>
                <CardContent className="divide-y divide-border">
                  {cat.fields.map((f) => (
                    <SettingRow key={f.key} field={f} onSave={save} onReset={reset} />
                  ))}
                </CardContent>
              </Card>
            </TabsContent>
          ))}

          <TabsContent value="models">
            <ModelsPanel />
          </TabsContent>

          <TabsContent value="system">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base"><Lock className="size-4" /> System configuration</CardTitle>
                <CardDescription>
                  Baked into the model pool at worker startup — edit <code className="rounded bg-muted px-1 py-0.5">.env</code> or the Models tab, then restart the worker.
                </CardDescription>
              </CardHeader>
              <CardContent className="divide-y divide-border">
                {data.restart_required.map((f) => (
                  <div key={f.key} className="flex items-center justify-between py-3">
                    <span className="text-sm text-muted-foreground">{humanize(f.key)}</span>
                    <div className="flex items-center gap-2">
                      <Badge variant="secondary" className="font-mono text-xs">{String(f.value)}</Badge>
                      {f.pending != null && f.pending !== f.value && (
                        <Badge variant="outline" className="gap-1 border-primary/30 bg-primary/10 font-mono text-xs text-primary">
                          → {String(f.pending)} pending
                        </Badge>
                      )}
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

function SettingRow({ field, onSave, onReset }: {
  field: Field; onSave: (key: string, value: boolean | number | string) => void; onReset: (key: string) => void;
}) {
  const [local, setLocal] = useState(String(field.value));
  const timer = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => setLocal(String(field.value)), [field.value]);

  function debouncedSave(raw: string) {
    setLocal(raw);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      const value = field.type === "int" ? parseInt(raw, 10) : field.type === "float" ? parseFloat(raw) : raw;
      if (typeof value === "number" && Number.isNaN(value)) return;
      onSave(field.key, value);
    }, 600);
  }

  return (
    <div className="flex items-center justify-between gap-6 py-3.5">
      <div className="min-w-0 flex-1 space-y-0.5">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">{humanize(field.key)}</span>
          {field.overridden && <Badge variant="outline" className="border-primary/30 bg-primary/10 text-[10px] text-primary">modified</Badge>}
        </div>
        <p className="font-mono text-xs text-muted-foreground">default: {String(field.default)}</p>
      </div>

      <div className="flex shrink-0 items-center gap-2">
        {field.type === "bool" ? (
          <Switch checked={local === "true"} onCheckedChange={(v) => onSave(field.key, v)} />
        ) : (
          <Input
            type={field.type === "int" || field.type === "float" ? "number" : "text"}
            step={field.type === "float" ? "0.01" : undefined}
            value={local}
            onChange={(e) => debouncedSave(e.target.value)}
            className="h-8 w-28 text-right tabular-nums"
          />
        )}
        {field.overridden && (
          <Button variant="ghost" size="icon" className="size-8 text-muted-foreground" onClick={() => onReset(field.key)} title="Reset to default">
            <RotateCcw className="size-3.5" />
          </Button>
        )}
      </div>
    </div>
  );
}
