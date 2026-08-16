import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { api } from "@/api/client";
import { Dialog } from "@/components/dialog";
import { ModelsPanel } from "@/components/models-panel";
import { AskAIPanel } from "@/components/ask-ai-panel";
import { AMBER_INK, muted, num, tagTone } from "@/lib/ui";

type Field = {
  key: string; value: boolean | number | string; default: boolean | number | string;
  overridden: boolean; updated_at: string | null; type: "bool" | "int" | "float" | "str";
};
type Category = { name: string; fields: Field[] };
type RestartField = { key: string; value: any; pending: any; editable: boolean; options: string[] | null };
type SettingsView = { categories: Category[]; restart_required: RestartField[] };

/** Help text for the thresholds whose behaviour is not obvious from the name. Deliberately
 *  partial: a field with nothing true to say gets no paragraph rather than a restatement
 *  of its own key. */
const HELP: Record<string, string> = {
  id_threshold: "Cosine similarity needed before a name is written to a turn.",
  id_suggest_delta: "Window below the threshold that still surfaces as a suggestion awaiting confirmation.",
  id_min_margin: "Required gap between the best and runner-up profile. Below it the system abstains rather than guessing.",
  id_threshold_penalty: "The threshold is raised by this much on low-reliability embeddings.",
  embed_min_s: "Shortest run of speech that yields an embedding. Below it, identification is never attempted.",
  cluster_threshold: "Cross-clip clustering cutoff — how close two voices must be to be treated as one person.",
  verify_threshold: "Similarity required to confirm an existing claim, stricter than a first guess.",
  auto_enroll: "Auto-enrol a cluster as a profile when confident. Off on purpose — one wrong auto-enrolment poisons every later match.",
  auto_enroll_min_sim: "Similarity required to auto-enrol. Inert while auto_enroll is off.",
  asr_language: "auto runs language detection; anything else pins it. A typo here silently transcribes in the wrong language.",
  reliability_fair: "The floor under which identification is not attempted at all.",
  retention_days: "How long processed audio survives before the retention sweep removes it.",
};

export default function Settings() {
  const [data, setData] = useState<SettingsView | null>(null);
  const [tab, setTab] = useState<string>("");
  const [calibrating, setCalibrating] = useState(false);
  const [calib, setCalib] = useState<any>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    const d: SettingsView = await api.getSettings();
    setData(d);
    setTab((t) => t || d.categories[0]?.name || "");
  }, []);
  useEffect(() => { load().catch((e) => toast.error(String(e))); }, [load]);

  async function save(key: string, value: boolean | number | string) {
    try {
      await api.patchSettings({ [key]: value });
      toast.success(`${key} updated`, { description: "Takes effect on the next job." });
      load();
    } catch (e) {
      toast.error(`Couldn't update ${key}`, { description: String(e) });
    }
  }

  // Same PATCH, different promise: these are read once when the worker builds its model
  // pool, so the change is pending until it restarts. Saying "next job" here would be a lie
  // that looks like a bug when the next job runs on the old device.
  async function restartSave(key: string, value: string) {
    try {
      await api.patchSettings({ [key]: value });
      toast.success(`${key} set to ${value}`, { description: "Takes effect after the worker restarts." });
      load();
    } catch (e) {
      toast.error(`Couldn't update ${key}`, { description: String(e) });
    }
  }

  async function reset(key: string) {
    await api.resetSetting(key);
    toast.success(`${key} reset to default`);
    load();
  }

  async function resetAll() {
    const overridden = (data?.categories || []).flatMap((c) => c.fields.filter((f) => f.overridden));
    for (const f of overridden) await api.resetSetting(f.key);
    toast.success(`${overridden.length} override${overridden.length === 1 ? "" : "s"} reset`);
    load();
  }

  async function openCalibrate() {
    setCalibrating(true);
    setCalib(null);
    try {
      setCalib(await api.calibrate());
    } catch (e) {
      setCalib({ error: String(e) });
    }
  }

  async function applyCalibration() {
    if (!calib?.suggested_id_threshold) return;
    setBusy(true);
    try {
      await api.patchSettings({
        id_threshold: Number(calib.suggested_id_threshold.toFixed(2)),
        verify_threshold: Number(calib.suggested_verify_threshold.toFixed(2)),
      });
      toast.success("Calibration applied", { description: "Two overrides written; reset them individually any time." });
      setCalibrating(false);
      load();
    } catch (e) {
      toast.error("Could not apply calibration", { description: String(e) });
    } finally { setBusy(false); }
  }

  if (!data) return <p style={{ color: muted(55) }}>Loading settings…</p>;

  const overrideCount = data.categories.reduce(
    (n, c) => n + c.fields.filter((f) => f.overridden).length, 0);
  const pendingCount = data.restart_required.filter((f) => f.pending != null && f.pending !== String(f.value)).length;
  const category = data.categories.find((c) => c.name === tab);
  const current = (key: string) => {
    const f = data.categories.flatMap((c) => c.fields).find((x) => x.key === key);
    return f ? String(f.value) : "—";
  };

  const tabs = [
    ...data.categories.map((c) => ({
      name: c.name,
      badge: c.fields.filter((f) => f.overridden).length || "",
    })),
    // Ask AI is not one of the pipeline categories: it is the only page that decides
    // whether anything leaves the box, so it gets its own tab rather than a row among the
    // thresholds, and its own endpoint behind it.
    { name: "Ask AI", badge: "" as string | number },
    { name: "Models", badge: "" as string | number },
    { name: "System", badge: pendingCount ? `${pendingCount} pending` : "" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <header style={{
        display: "flex", alignItems: "flex-end", justifyContent: "space-between",
        gap: 20, flexWrap: "wrap",
      }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
          <h2 style={{ margin: 0, fontSize: 30 }}>Settings</h2>
          <p style={{ margin: 0, fontSize: 14, color: muted(68) }}>
            <b>{overrideCount} setting{overrideCount === 1 ? "" : "s"} overridden.</b> Changes
            apply to the next job — no restart. Anything that needs a restart says so and is
            stored as pending.
          </p>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <button className="btn btn-secondary" disabled={!overrideCount} onClick={resetAll}>
            Reset all overrides
          </button>
          <button className="btn btn-primary" onClick={openCalibrate}>Calibrate from my profiles</button>
        </div>
      </header>

      <div style={{
        display: "flex", gap: 2, flexWrap: "wrap",
        borderBottom: "1px solid var(--color-divider)",
      }}>
        {tabs.map((t) => (
          <span
            key={t.name}
            onClick={() => setTab(t.name)}
            style={{
              fontFamily: "var(--font-heading)", fontSize: 14, padding: "8px 12px",
              border: "1px solid transparent",
              borderBottom: `2px solid ${t.name === tab ? "var(--color-accent)" : "transparent"}`,
              color: t.name === tab ? "var(--color-text)" : muted(55),
              cursor: "pointer", whiteSpace: "nowrap",
            }}
          >
            {t.name}{" "}
            <span className="mono" style={{ fontSize: 10, color: muted(40) }}>{t.badge}</span>
          </span>
        ))}
      </div>

      {tab === "Speaker identification" && (
        <div className="blueprint" style={{
          padding: "13px 15px", display: "flex", gap: 14, alignItems: "flex-start", maxWidth: 900,
        }}>
          <span className="mono" style={{ fontSize: 11, color: "var(--color-accent-700)", paddingTop: 2 }}>
            NOTE
          </span>
          <span style={{ fontSize: 13, lineHeight: 1.55, color: muted(72) }}>
            These defaults are not universal constants. With 3+ profiles at 2+ enrollments
            each, calibration re-derives <span className="mono">id_threshold</span> and{" "}
            <span className="mono">verify_threshold</span> from your own voices. Prefer it over
            hand-tuning.
          </span>
        </div>
      )}

      {category && (
        <section style={{
          display: "flex", flexDirection: "column", maxWidth: 900,
          borderTop: "1px solid var(--color-divider)",
        }}>
          {category.fields.map((f) => (
            <SettingRow key={f.key} field={f} onSave={save} onReset={reset} />
          ))}
        </section>
      )}

      {tab === "Ask AI" && <AskAIPanel />}

      {tab === "Models" && <ModelsPanel />}

      {tab === "System" && (
        <section style={{ display: "flex", flexDirection: "column", gap: 14, maxWidth: 900 }}>
          <p style={{ margin: 0, fontSize: 13.5, color: muted(68) }}>
            These are read once when the worker builds its model pool, so an edit is stored as
            pending and takes effect at the next worker start. They are listed here{" "}
            <i>with the reason</i>, not hidden.
          </p>
          <div style={{ display: "flex", flexDirection: "column", borderTop: "1px solid var(--color-divider)" }}>
            {data.restart_required.map((f) => {
              const pending = f.pending != null && String(f.pending) !== String(f.value);
              return (
                <div key={f.key} style={{
                  display: "grid", gridTemplateColumns: "1fr 250px", gap: 24, padding: "14px 0",
                  borderBottom: `1px solid ${muted(8)}`, alignItems: "center",
                }}>
                  <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                      <span className="mono" style={{ fontSize: 13 }}>{f.key}</span>
                      <span className="tag" style={{
                        border: `1px solid ${tagTone("warn").border}`, color: AMBER_INK, fontSize: 10,
                      }}>
                        RESTART
                      </span>
                    </div>
                    <span className="mono" style={{ fontSize: 12, color: muted(60) }}>
                      {String(f.value)}
                      {pending && <> → <span style={{ color: AMBER_INK }}>{String(f.pending)} pending</span></>}
                    </span>
                  </div>
                  <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", alignItems: "center" }}>
                    {f.editable && f.options && (
                      <select className="input mono" value={String(f.pending ?? f.value)}
                              onChange={(e) => restartSave(f.key, e.target.value)}>
                        {f.options.map((o) => <option key={o} value={o}>{o}</option>)}
                      </select>
                    )}
                    {f.editable && pending && (
                      <button className="btn btn-secondary" style={{ fontSize: 12, padding: "5px 10px" }}
                              onClick={() => reset(f.key)}>
                        Clear pending
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}

      <Dialog
        open={calibrating}
        onClose={() => setCalibrating(false)}
        kicker="Settings"
        title="Calibrate thresholds from your profiles"
        subject={calib?.n_profiles ? `${calib.n_profiles} profiles · ${calib.n_genuine} same-voice pairs · ${calib.n_impostor} cross-voice pairs` : "measuring…"}
        consequence={calib?.suggested_id_threshold
          ? "Writes two overrides you can reset individually afterwards."
          : ""}
        confirm={calib?.suggested_id_threshold ? "Apply calibration" : undefined}
        width="780px"
        busy={busy}
        onConfirm={applyCalibration}
      >
        {!calib && <p style={{ margin: 0, fontSize: 14, color: muted(65) }}>Comparing every enrollment against every other…</p>}
        {calib?.error && (
          <p style={{ margin: 0, fontSize: 14, color: AMBER_INK }}>
            {calib.error} — calibration needs at least 3 profiles with 2 enrollments each.
          </p>
        )}
        {calib?.suggested_id_threshold && (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <p style={{ margin: 0, fontSize: 14, lineHeight: 1.55, color: muted(72) }}>
              Calibration compares every enrollment against every other and picks the cutoffs
              that separate your voices best. Nothing is written until you apply it.
            </p>
            <div style={{ display: "flex", flexDirection: "column", borderTop: "1px solid var(--color-divider)" }}>
              {([
                ["id_threshold", current("id_threshold"), calib.suggested_id_threshold.toFixed(2),
                 "The bar a match must clear before a name is written."],
                ["verify_threshold", current("verify_threshold"), calib.suggested_verify_threshold.toFixed(2),
                 "Confirmations get stricter than first guesses."],
              ] as const).map(([key, from, to, effect]) => (
                <div key={key} style={{
                  display: "grid", gridTemplateColumns: "1fr 130px 1fr", gap: 12, padding: "11px 2px",
                  borderBottom: `1px solid ${muted(8)}`, alignItems: "center",
                }}>
                  <span className="mono" style={{ fontSize: 13 }}>{key}</span>
                  <span className="mono" style={{ fontSize: 12.5, textAlign: "center", color: muted(60) }}>
                    {from} → <span style={{ color: "var(--color-accent-700)" }}>{to}</span>
                  </span>
                  <span style={{ fontSize: 12.5, color: muted(62) }}>{effect}</span>
                </div>
              ))}
            </div>
            <div className="blueprint" style={{ padding: 12, display: "flex", flexDirection: "column", gap: 6 }}>
              <span className="kicker-sm" style={{ color: muted(55) }}>Measured on your own data</span>
              <div className="mono" style={{ display: "flex", gap: 20, fontSize: 12, flexWrap: "wrap" }}>
                <span>equal error rate {(calib.eer * 100).toFixed(1)}%</span>
                <span>same-voice mean {num(calib.genuine_mean)}</span>
                <span>cross-voice mean {num(calib.impostor_mean)}</span>
                <span>separation {num(calib.separation)}</span>
              </div>
            </div>
            <span style={{ fontSize: 12.5, color: muted(60) }}>
              Existing manual corrections are kept. Clips already processed are not re-run.
            </span>
          </div>
        )}
      </Dialog>
    </div>
  );
}

function SettingRow({ field, onSave, onReset }: {
  field: Field;
  onSave: (key: string, value: boolean | number | string) => void;
  onReset: (key: string) => void;
}) {
  const [local, setLocal] = useState(String(field.value));
  const timer = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => setLocal(String(field.value)), [field.value]);

  function debouncedSave(raw: string) {
    setLocal(raw);
    clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      const value = field.type === "int" ? parseInt(raw, 10)
        : field.type === "float" ? parseFloat(raw) : raw;
      if (typeof value === "number" && Number.isNaN(value)) return;
      onSave(field.key, value);
    }, 600);
  }

  const tone = tagTone("accent");
  return (
    <div style={{
      display: "grid", gridTemplateColumns: "1fr 250px", gap: 24, padding: "14px 0",
      borderBottom: `1px solid ${muted(8)}`,
    }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <span className="mono" style={{ fontSize: 13 }}>{field.key}</span>
          <span className="tag" style={{ border: `1px solid ${tone.border}`, color: tone.color, fontSize: 10 }}>
            LIVE
          </span>
          <span className="mono" style={{
            fontSize: 11, color: field.overridden ? AMBER_INK : muted(55),
          }}>
            {field.overridden ? "OVERRIDDEN" : "DEFAULT"}
          </span>
        </div>
        {HELP[field.key] && (
          <span style={{ fontSize: 13, lineHeight: 1.5, color: muted(68) }}>{HELP[field.key]}</span>
        )}
        <span className="mono" style={{ fontSize: 11, color: muted(45) }}>
          {field.type} · default {String(field.default)}
          {field.updated_at && ` · changed ${new Date(field.updated_at).toLocaleString()}`}
        </span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: "stretch" }}>
        {field.type === "bool" ? (
          <div className="seg" style={{ alignSelf: "flex-end" }}>
            <label className="seg-opt">
              <input type="radio" name={field.key} checked={local === "true"}
                     onChange={() => onSave(field.key, true)} />
              On
            </label>
            <label className="seg-opt">
              <input type="radio" name={field.key} checked={local !== "true"}
                     onChange={() => onSave(field.key, false)} />
              Off
            </label>
          </div>
        ) : (
          <input
            className="input mono"
            value={local}
            onChange={(e) => debouncedSave(e.target.value)}
            style={{
              textAlign: "right",
              borderColor: field.overridden ? "var(--color-accent)" : "var(--color-divider)",
            }}
          />
        )}
        {field.overridden && (
          <a href="#" style={{ fontSize: 12, textAlign: "right" }}
             onClick={(e) => { e.preventDefault(); onReset(field.key); }}>
            Reset to {String(field.default)}
          </a>
        )}
      </div>
    </div>
  );
}
