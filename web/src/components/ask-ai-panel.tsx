import { useEffect, useState } from "react";
import { toast } from "sonner";
import { api } from "@/api/client";
import { AMBER_INK, GREEN, muted, tagTone } from "@/lib/ui";

type Provider = {
  name: string;
  models: string[];
  default_model: string;
  key_set: boolean;
  key_hint: string;
  base_url: string | null;
};
type LLMConfig = {
  enabled: boolean;
  provider: string;
  model: string;
  providers: Provider[];
};

const BLURB: Record<string, string> = {
  anthropic: "Hosted. Strongest tool use and the most reliable at citing what it read.",
  groq: "Hosted, and fast — hundreds of tokens a second, so an answer arrives while you are still reading the question.",
  nvidia: "Hosted, OpenAI-shaped. Also the path to a local vLLM or Ollama server by pointing its base URL at one.",
};

/** Ask's configuration, and the only place a provider key can be set.
 *
 *  The key is write-only by design: the API reports whether one is installed and its last
 *  four characters, never the key, so it cannot arrive in the browser, its devtools, or
 *  anything that logs a response. The field is therefore always empty on load — typing in
 *  it replaces the stored key, leaving it alone keeps it. */
export function AskAIPanel() {
  const [cfg, setCfg] = useState<LLMConfig | null>(null);
  const [keyDraft, setKeyDraft] = useState("");
  const [busy, setBusy] = useState(false);

  async function load() {
    setCfg(await api.getLLM());
    setKeyDraft("");
  }
  useEffect(() => { load().catch((e) => toast.error(String(e))); }, []);

  async function save(body: Parameters<typeof api.patchLLM>[0], note: string) {
    setBusy(true);
    try {
      setCfg(await api.patchLLM(body));
      setKeyDraft("");
      toast.success(note);
    } catch (e) {
      toast.error("Could not save that", { description: String(e) });
    } finally { setBusy(false); }
  }

  if (!cfg) return <p style={{ color: muted(55) }}>Loading the agent's configuration…</p>;

  const active = cfg.providers.find((p) => p.name === cfg.provider);
  const models = active?.models ?? [];
  const effectiveModel = cfg.model || active?.default_model || "";
  const ready = cfg.enabled && !!active?.key_set;

  return (
    <section style={{ display: "flex", flexDirection: "column", gap: 18, maxWidth: 900 }}>
      <div className="blueprint hatch" style={{
        padding: "13px 15px", display: "flex", gap: 14, alignItems: "flex-start",
      }}>
        <span className="mono" style={{ fontSize: 11, color: AMBER_INK, paddingTop: 2 }}>
          OFF-BOX
        </span>
        <span style={{ fontSize: 13, lineHeight: 1.55, color: muted(72) }}>
          Ask is the only feature that sends anything off this machine. Retrieval stays
          local; the passages it retrieves are sent to the provider below to be turned into
          an answer. Everything else — transcription, diarization, identification, tone —
          runs here regardless of what is set on this page.
        </span>
      </div>

      <div style={{
        display: "grid", gridTemplateColumns: "1fr 250px", gap: 24, padding: "14px 0",
        borderTop: "1px solid var(--color-divider)", borderBottom: `1px solid ${muted(8)}`,
      }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span className="mono" style={{ fontSize: 13 }}>llm_enabled</span>
            <span className="tag mono" style={{
              border: `1px solid ${ready ? GREEN : tagTone("neutral").border}`,
              color: ready ? GREEN : tagTone("neutral").color, fontSize: 10,
            }}>
              {ready ? "READY" : cfg.enabled ? "NO KEY" : "OFF"}
            </span>
          </div>
          <span style={{ fontSize: 13, lineHeight: 1.5, color: muted(68) }}>
            With this off, Ask answers 503 and nothing leaves the box. Takes effect on the
            next question — no restart.
          </span>
        </div>
        <div className="seg" style={{ alignSelf: "flex-end" }}>
          <label className="seg-opt">
            <input type="radio" name="llm_enabled" checked={cfg.enabled} disabled={busy}
                   onChange={() => save({ enabled: true }, "Ask enabled")} />
            On
          </label>
          <label className="seg-opt">
            <input type="radio" name="llm_enabled" checked={!cfg.enabled} disabled={busy}
                   onChange={() => save({ enabled: false }, "Ask disabled — nothing leaves the box")} />
            Off
          </label>
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
          <h4 style={{ margin: 0 }}>Provider</h4>
          <span style={{ fontSize: 12, color: muted(55) }}>
            One at a time. Each keeps its own key, so switching back does not mean typing it again.
          </span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 16 }}>
          {cfg.providers.map((p) => {
            const on = p.name === cfg.provider;
            return (
              <article
                key={p.name}
                onClick={() => !on && save({ provider: p.name }, `Provider set to ${p.name}`)}
                className="blueprint"
                style={{
                  padding: 14, display: "flex", flexDirection: "column", gap: 9,
                  cursor: on ? "default" : "pointer",
                  borderLeft: `3px solid ${on ? "var(--color-accent)" : "transparent"}`,
                  background: on ? "color-mix(in srgb, var(--color-accent) 7%, transparent)" : "transparent",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <span style={{ fontFamily: "var(--font-heading)", fontSize: 18 }}>{p.name}</span>
                  <span className="tag mono" style={{
                    border: `1px solid ${p.key_set ? GREEN : tagTone("warn").border}`,
                    color: p.key_set ? GREEN : AMBER_INK, fontSize: 10,
                  }}>
                    {p.key_set ? `key ${p.key_hint}` : "no key"}
                  </span>
                </div>
                <span style={{ fontSize: 12.5, lineHeight: 1.5, color: muted(62) }}>
                  {BLURB[p.name]}
                </span>
                {p.base_url && (
                  <span className="mono" style={{ fontSize: 11, color: muted(45) }}>
                    {p.base_url.replace(/^https?:\/\//, "")}
                  </span>
                )}
              </article>
            );
          })}
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
          <h4 style={{ margin: 0 }}>Model</h4>
          <span style={{ fontSize: 12, color: muted(55) }}>
            For {cfg.provider}. Speed and tool-following vary far more than the names suggest.
          </span>
        </div>
        <div style={{ display: "flex", flexDirection: "column" }}>
          {models.map((m) => {
            const on = m === effectiveModel;
            return (
              <label key={m} style={{
                display: "grid", gridTemplateColumns: "18px 1fr auto", gap: 12,
                alignItems: "center", padding: "11px 4px", cursor: "pointer",
                borderBottom: `1px solid ${muted(8)}`,
                background: on ? "color-mix(in srgb, var(--color-accent) 7%, transparent)" : "transparent",
              }}>
                <input type="radio" name="llm_model" checked={on} disabled={busy}
                       onChange={() => save({ model: m }, `Model set to ${m}`)} />
                <span className="mono" style={{ fontSize: 13 }}>{m}</span>
                {m === active?.default_model && (
                  <span className="mono" style={{ fontSize: 11, color: muted(45) }}>default</span>
                )}
              </label>
            );
          })}
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <input
            className="input mono"
            style={{ flex: 1 }}
            placeholder="…or any model id this endpoint serves"
            defaultValue=""
            onKeyDown={(e) => {
              if (e.key !== "Enter") return;
              const v = (e.target as HTMLInputElement).value.trim();
              if (v) save({ model: v }, `Model set to ${v}`);
            }}
          />
          <span className="mono" style={{ fontSize: 11, color: muted(48) }}>enter to set</span>
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
          <h4 style={{ margin: 0 }}>API key</h4>
          <span style={{ fontSize: 12, color: muted(55) }}>
            For {cfg.provider}. Stored on this machine and never sent back to the browser.
          </span>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
          <div className="field" style={{ flex: 1 }}>
            <label>
              {active?.key_set
                ? `A key is installed (${active.key_hint}). Typing here replaces it.`
                : "No key installed. Ask cannot answer until one is set."}
            </label>
            <input
              className="input mono"
              type="password"
              autoComplete="off"
              value={keyDraft}
              placeholder={`${cfg.provider} api key`}
              onChange={(e) => setKeyDraft(e.target.value)}
            />
          </div>
          <button
            className="btn btn-primary"
            disabled={busy || !keyDraft.trim()}
            onClick={() => save(
              { api_key_provider: cfg.provider, api_key: keyDraft.trim() },
              `Key saved for ${cfg.provider}`)}
          >
            Save key
          </button>
          {active?.key_set && (
            <button
              className="btn btn-secondary"
              disabled={busy}
              onClick={() => save(
                { api_key_provider: cfg.provider, api_key: "" },
                `Key removed for ${cfg.provider}`)}
            >
              Remove
            </button>
          )}
        </div>
        <span style={{ fontSize: 12, lineHeight: 1.5, color: muted(55) }}>
          The key is held in the settings table in plaintext, like every other override, and
          anyone holding this workbench's API key can change it. Keep it in <span className="mono">.env</span>{" "}
          instead if that is not an acceptable trade for your deployment — a key set there
          still works, and this page will show it as installed.
        </span>
      </div>
    </section>
  );
}
