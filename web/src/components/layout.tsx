import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { Toaster } from "sonner";
import { api } from "@/api/client";
import { useTheme } from "@/lib/theme";
import { muted } from "@/lib/ui";

/** A screen can take over the sidebar for the length of its stay — Ask replaces the nav
 *  with its thread history, the way the mockup does. Nothing else does, and the nav comes
 *  straight back when the screen unmounts. */
const SidebarCtx = createContext<(n: ReactNode | null) => void>(() => {});

export function useSidebarTakeover(node: ReactNode | null, deps: unknown[]) {
  const set = useContext(SidebarCtx);
  useEffect(() => {
    set(node);
    return () => set(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}

const NAV = [
  { to: "/library", label: "Library", mark: "▤", end: true, count: "clips" },
  { to: "/review", label: "Review queue", mark: "◈", end: false, count: "review" },
  { to: "/ask", label: "Ask", mark: "?", end: false, count: null },
  { to: "/upload", label: "Upload", mark: "↑", end: false, count: "inflight" },
  { to: "/live", label: "Live", mark: "◉", end: false, count: null },
  { to: "/races", label: "Races", mark: "◇", end: false, count: "races" },
  { to: "/speakers", label: "Speakers", mark: "◍", end: false, count: "speakers" },
  { to: "/settings", label: "Settings", mark: "≡", end: false, count: "overrides" },
] as const;

type Counts = Partial<Record<"clips" | "review" | "inflight" | "races" | "speakers" | "overrides", number>>;

const TERMINAL = ["COMPLETE", "FAILED", "REJECTED", "DEAD"];

export default function Layout({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useTheme();
  const [sidebar, setSidebar] = useState<ReactNode | null>(null);
  const [counts, setCounts] = useState<Counts>({});
  const [online, setOnline] = useState(true);
  const { pathname } = useLocation();

  // Counts are what makes the nav a status board rather than a list of links, so they are
  // read from the API — never hard-coded — and refreshed as you move between screens.
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [stats, review, races, speakers, settings] = await Promise.all([
          api.stats(), api.listClips({ needs_review: "true", limit: "1" }),
          api.listRaces(), api.listSpeakers(), api.getSettings(),
        ]);
        if (!alive) return;
        const inflight = Object.entries(stats.by_status || {})
          .filter(([s]) => !TERMINAL.includes(s))
          .reduce((n, [, c]) => n + (c as number), 0);
        setCounts({
          clips: stats.total_clips, review: review.total, inflight,
          races: races.items.length, speakers: speakers.items.length,
          overrides: settings.categories.reduce(
            (n: number, c: any) => n + c.fields.filter((f: any) => f.overridden).length, 0),
        });
        setOnline(true);
      } catch {
        if (alive) setOnline(false);
      }
    })();
    return () => { alive = false; };
  }, [pathname]);

  return (
    <SidebarCtx.Provider value={setSidebar}>
      <div style={{
        display: "flex", minHeight: "100vh", alignItems: "stretch", fontSize: 14,
        background: "var(--color-bg)", color: "var(--color-text)",
      }}>
        <aside style={{
          width: 240, flex: "none", borderRight: "1px solid var(--color-divider)",
          display: "flex", flexDirection: "column", padding: "20px 0",
          position: "sticky", top: 0, height: "100vh",
        }}>
          {/* The wordmark goes back out to the landing page. */}
          <NavLink to="/" style={{ padding: "0 18px 18px", display: "flex", flexDirection: "column", gap: 2, color: "inherit" }}>
            <span style={{ fontFamily: "var(--font-heading)", fontSize: 19, lineHeight: 1.05, letterSpacing: "-.01em" }}>
              SPEAKER INTELLIGENCE
            </span>
            <span style={{ fontFamily: "var(--font-heading)", fontSize: 19, lineHeight: 1.05, color: "var(--color-accent)" }}>
              WORKBENCH
            </span>
            <span className="kicker" style={{ marginTop: 6 }}>on-premises · v2.4</span>
          </NavLink>

          {sidebar}

          {!sidebar && (
            <nav style={{ display: "flex", flexDirection: "column", gap: 1, padding: "8px 8px 0" }}>
              {NAV.map((item) => (
                <NavLink key={item.to} to={item.to} end={item.end} style={{ color: "inherit" }}>
                  {({ isActive }) => (
                    <span
                      className="row-hover"
                      style={{
                        display: "flex", alignItems: "center", gap: 10, width: "100%",
                        textAlign: "left", font: "inherit", cursor: "pointer", padding: "8px 10px",
                        border: "1px solid transparent",
                        borderLeftColor: isActive ? "var(--color-accent)" : "transparent",
                        background: isActive
                          ? "color-mix(in srgb, var(--color-accent) 12%, transparent)"
                          : "transparent",
                      }}
                    >
                      <span style={{ width: 14, textAlign: "center", fontSize: 11, color: "var(--color-accent)" }}>
                        {item.mark}
                      </span>
                      <span style={{ flex: 1 }}>{item.label}</span>
                      <span className="mono" style={{ fontSize: 11, color: muted(45) }}>
                        {item.count ? (counts[item.count] ?? "—") : "—"}
                      </span>
                    </span>
                  )}
                </NavLink>
              ))}
            </nav>
          )}

          <div style={{
            marginTop: "auto", padding: "14px 18px 0", borderTop: "1px solid var(--color-divider)",
            display: "flex", flexDirection: "column", gap: 11,
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: muted(60) }}>
              <span style={{
                width: 7, height: 7, borderRadius: "50%", flex: "none",
                background: online ? "var(--sig-green)" : "var(--sig-red)",
              }} />
              {online ? "All systems operational" : "API unreachable"}
            </div>
            <div className="seg" style={{ alignSelf: "stretch" }}>
              <label className="seg-opt" style={{ flex: 1, justifyContent: "center" }}>
                <input type="radio" name="theme" checked={theme === "light"} onChange={() => setTheme("light")} />
                Light
              </label>
              <label className="seg-opt" style={{ flex: 1, justifyContent: "center" }}>
                <input type="radio" name="theme" checked={theme === "dark"} onChange={() => setTheme("dark")} />
                Dark
              </label>
            </div>
          </div>
        </aside>

        {/* The column caps at 1240 — the width the design was drawn for. Past that the
            leftover space is split either side instead of all piling up on the right. */}
        <main style={{
          flex: 1, minWidth: 0, padding: "26px 34px 60px", maxWidth: 1240, marginInline: "auto",
        }}>
          {children}
        </main>

        <Toaster
          theme={theme}
          position="bottom-right"
          style={{
            "--normal-bg": "var(--color-bg)",
            "--normal-text": "var(--color-text)",
            "--normal-border": "var(--color-text)",
            "--border-radius": "0",
          } as React.CSSProperties}
        />
      </div>
    </SidebarCtx.Provider>
  );
}
