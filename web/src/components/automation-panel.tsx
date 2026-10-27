import { useEffect, useState } from "react";
import { toast } from "sonner";
import { api } from "@/api/client";
import { fmtAgo, muted } from "@/lib/ui";

type Status = {
  enabled: boolean; consolidate_clusters: boolean; retention_days: number; untagged: number;
  recent: { ts: string; action: string; object_id: string; after: string | null }[];
};

const ACTION_TEXT: Record<string, string> = {
  "maintenance.stalled": "marked stalled clips FAILED",
  "maintenance.requeue": "re-queued clips whose job had vanished",
  "maintenance.retention": "deleted clips past retention",
  "maintenance.tag_backfill": "tagged untagged clips",
  "maintenance.consolidate": "merged duplicate voice clusters",
};

function countOf(after: string | null) {
  try {
    const a = after ? JSON.parse(after) : null;
    return a?.count ?? a?.merged ?? null;
  } catch { return null; }
}

/** What the worker's background upkeep has done, read back from the audit log — the same
 *  record a person's actions leave — plus the two buttons that run it on demand. */
export function AutomationPanel() {
  const [s, setS] = useState<Status | null>(null);
  const load = () => api.maintenanceStatus().then(setS).catch(() => {});
  useEffect(() => { load(); }, []);

  async function act(fn: () => Promise<unknown>, done: string) {
    try {
      await fn();
      toast.success(done, { description: "Queued on the worker." });
      setTimeout(load, 5000);
    } catch (e) {
      toast.error("Could not queue it", { description: String(e) });
    }
  }

  return (
    <section className="blueprint" style={{
      padding: "13px 15px", display: "flex", flexDirection: "column", gap: 10, maxWidth: 900,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
        <span style={{ fontSize: 13, color: muted(72) }}>
          Maintenance is <b>{s ? (s.enabled ? "on" : "off") : "…"}</b>
          {s && <> · {s.untagged} clip{s.untagged === 1 ? "" : "s"} waiting for tags
            · retention {s.retention_days ? `${s.retention_days} days` : "off"}
            · cluster merging {s.consolidate_clusters ? "on" : "off"}</>}
        </span>
        <span style={{ display: "flex", gap: 8 }}>
          <button className="btn btn-secondary" style={{ fontSize: 12, padding: "5px 10px" }}
                  onClick={() => act(api.runMaintenance, "Maintenance pass started")}>Run now</button>
          <button className="btn btn-secondary" style={{ fontSize: 12, padding: "5px 10px" }}
                  onClick={() => act(api.retagAll, "Re-tagging every clip")}>Re-tag all clips</button>
        </span>
      </div>
      {s && s.recent.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {s.recent.slice(0, 8).map((r, i) => (
            <div key={i} className="mono" style={{ fontSize: 11.5, color: muted(62), display: "flex", gap: 10 }}>
              <span style={{ width: 90, flex: "none" }}>{fmtAgo(r.ts)}</span>
              <span>{ACTION_TEXT[r.action] || r.action}{countOf(r.after) != null && ` · ${countOf(r.after)}`}</span>
            </div>
          ))}
        </div>
      )}
      {s && s.recent.length === 0 && (
        <span style={{ fontSize: 12, color: muted(50) }}>Nothing has needed doing yet.</span>
      )}
    </section>
  );
}
