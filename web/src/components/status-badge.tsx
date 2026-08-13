import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { CheckCircle2, XCircle, AlertTriangle, Loader2, Clock } from "lucide-react";

const TERMINAL: Record<string, { icon: typeof CheckCircle2; className: string }> = {
  COMPLETE: { icon: CheckCircle2, className: "border-success/30 bg-success/10 text-success" },
  FAILED: { icon: XCircle, className: "border-destructive/30 bg-destructive/10 text-destructive" },
  DEAD: { icon: XCircle, className: "border-destructive/30 bg-destructive/10 text-destructive" },
  REJECTED: { icon: AlertTriangle, className: "border-warning/30 bg-warning/10 text-warning" },
  QUEUED: { icon: Clock, className: "border-border bg-muted text-muted-foreground" },
};

export function StatusBadge({ status }: { status: string }) {
  const terminal = TERMINAL[status];
  if (terminal) {
    const Icon = terminal.icon;
    return (
      <Badge variant="outline" className={cn("gap-1.5 font-medium", terminal.className)}>
        <Icon className="size-3" strokeWidth={2.5} />
        {status}
      </Badge>
    );
  }
  return (
    <Badge variant="outline" className="gap-1.5 border-primary/30 bg-primary/10 font-medium text-primary">
      <Loader2 className="size-3 animate-spin" strokeWidth={2.5} />
      {status}
    </Badge>
  );
}
