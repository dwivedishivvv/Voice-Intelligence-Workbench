import { useEffect } from "react";
import { muted, RED, ACCENT } from "@/lib/ui";

/** The workbench dialog: a kicker that says which surface you are on, the question as a
 *  title, the subject in monospace, and a footer that states the consequence of the
 *  confirm button in words before you press it. Every destructive or identity-changing
 *  action in the app goes through this shell so the consequence is never missing. */
export function Dialog({
  open, onClose, kicker, title, subject, consequence, confirm, cancel = "Cancel",
  danger, width = "560px", onConfirm, busy, confirmDisabled, children,
}: {
  open: boolean;
  onClose: () => void;
  kicker: string;
  title: string;
  subject?: string;
  consequence?: string;
  confirm?: string;
  cancel?: string;
  danger?: boolean;
  width?: string;
  onConfirm?: () => void;
  busy?: boolean;
  confirmDisabled?: boolean;
  children?: React.ReactNode;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, zIndex: 60, background: "var(--sig-scrim)",
        display: "flex", alignItems: "center", justifyContent: "center", padding: 24,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width, maxWidth: "calc(100vw - 48px)", maxHeight: "calc(100vh - 48px)",
          display: "flex", flexDirection: "column", background: "var(--color-bg)",
          border: "1px solid var(--color-text)", boxShadow: "var(--shadow-lg)",
        }}
      >
        <header style={{
          flex: "none", padding: "16px 20px 14px", borderBottom: "1px solid var(--color-divider)",
          display: "flex", flexDirection: "column", gap: 4,
        }}>
          <span className="kicker">{kicker}</span>
          <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16 }}>
            <h3 style={{ margin: 0, fontSize: 22, lineHeight: 1.15 }}>{title}</h3>
            <button
              onClick={onClose}
              aria-label="Close"
              style={{
                background: "transparent", border: "none", font: "inherit", fontSize: 18,
                lineHeight: 1, cursor: "pointer", color: muted(45), padding: "2px 0",
              }}
            >×</button>
          </div>
          {subject && (
            <span className="mono" style={{ fontSize: 11.5, color: muted(55) }}>{subject}</span>
          )}
        </header>

        <div style={{
          flex: 1, minHeight: 0, overflow: "auto", padding: "18px 20px",
          display: "flex", flexDirection: "column", gap: 16,
        }}>
          {children}
        </div>

        <footer style={{
          flex: "none", padding: "14px 20px", borderTop: "1px solid var(--color-divider)",
          display: "flex", alignItems: "center", justifyContent: "space-between",
          gap: 16, flexWrap: "wrap",
        }}>
          <span style={{ fontSize: 12, lineHeight: 1.45, color: muted(58), flex: 1, minWidth: 200 }}>
            {consequence}
          </span>
          <div style={{ display: "flex", gap: 8, flex: "none" }}>
            <button className="btn btn-secondary" onClick={onClose}>{cancel}</button>
            {confirm && (
              <button
                className="btn btn-primary"
                onClick={onConfirm}
                disabled={busy || confirmDisabled}
                style={{ background: danger ? RED : ACCENT, borderColor: danger ? RED : ACCENT }}
              >
                {busy ? "Working…" : confirm}
              </button>
            )}
          </div>
        </footer>
      </div>
    </div>
  );
}
