import { NavLink } from "react-router-dom";
import { AudioLines, Upload, Users, SlidersHorizontal, Mic, Flag, Trophy, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  { to: "/", label: "Library", icon: AudioLines, end: true },
  { to: "/ask", label: "Ask", icon: Sparkles, end: false },
  { to: "/upload", label: "Upload", icon: Upload, end: false },
  { to: "/live", label: "Live", icon: Mic, end: false },
  { to: "/f1", label: "Race Radio", icon: Flag, end: false },
  { to: "/races", label: "Races", icon: Trophy, end: false },
  { to: "/speakers", label: "Speakers", icon: Users, end: false },
  { to: "/settings", label: "Settings", icon: SlidersHorizontal, end: false },
];

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen bg-background">
      <aside className="fixed inset-y-0 left-0 z-40 flex w-60 flex-col border-r border-sidebar-border bg-sidebar">
        <div className="flex h-16 items-center gap-2.5 px-5">
          <div className="flex size-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-chart-5 shadow-lg shadow-primary/20">
            <AudioLines className="size-4.5 text-primary-foreground" strokeWidth={2.5} />
          </div>
          <div className="flex flex-col leading-none">
            <span className="text-sm font-semibold tracking-tight text-sidebar-foreground">Speaker Workbench</span>
            <span className="text-[11px] text-muted-foreground">Intelligence Suite</span>
          </div>
        </div>

        <nav className="flex-1 space-y-0.5 px-3 py-2">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                cn(
                  "group flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:bg-white/5 hover:text-foreground"
                )
              }
            >
              {({ isActive }) => (
                <>
                  <Icon className={cn("size-4", isActive ? "text-primary" : "text-muted-foreground group-hover:text-foreground")} strokeWidth={2} />
                  {label}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="border-t border-sidebar-border px-5 py-4">
          <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
            <span className="relative flex size-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-success opacity-75" />
              <span className="relative inline-flex size-1.5 rounded-full bg-success" />
            </span>
            All systems operational
          </div>
        </div>
      </aside>

      <div className="flex flex-1 flex-col pl-60">
        <main className="flex-1 animate-fade-in">{children}</main>
      </div>
    </div>
  );
}
