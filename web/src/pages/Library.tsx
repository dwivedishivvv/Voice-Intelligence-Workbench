import { useEffect, useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/api/client";
import { PageHeader } from "@/components/page-header";
import { StatusBadge } from "@/components/status-badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter,
} from "@/components/ui/dialog";
import {
  DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import {
  AudioLines, Search, Upload as UploadIcon, AlertTriangle, Users2, MoreVertical, Pencil, Trash2,
} from "lucide-react";

type Clip = {
  id: string; filename: string; duration_s: number; status: string;
  language: string; n_speakers: number; needs_review: boolean; created_at: string;
};

type EditState = { id: string; filename: string; notes: string; tags: string; needs_review: boolean };

function fmtDuration(s: number) {
  if (!s) return "—";
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

export default function Library() {
  const [clips, setClips] = useState<Clip[] | null>(null);
  const [q, setQ] = useState("");
  const [needsReview, setNeedsReview] = useState(false);
  const [edit, setEdit] = useState<EditState | null>(null);
  const [saving, setSaving] = useState(false);
  const navigate = useNavigate();

  async function load() {
    const params: Record<string, string> = {};
    if (q) params.q = q;
    if (needsReview) params.needs_review = "true";
    const res = await api.listClips(params);
    setClips(res.items);
  }

  useEffect(() => {
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q, needsReview]);

  async function openEdit(clip: Clip) {
    const full = await api.getClip(clip.id);
    setEdit({
      id: clip.id, filename: clip.filename,
      notes: full.notes || "", tags: (full.tags || []).join(", "),
      needs_review: full.needs_review,
    });
  }

  async function saveEdit() {
    if (!edit) return;
    setSaving(true);
    try {
      await api.patchClip(edit.id, {
        notes: edit.notes,
        tags: edit.tags.split(",").map((t) => t.trim()).filter(Boolean),
        needs_review: edit.needs_review,
      });
      toast.success("Clip updated");
      setEdit(null);
      load();
    } catch (e) {
      toast.error("Update failed", { description: String(e) });
    } finally {
      setSaving(false);
    }
  }

  async function removeClip(clip: Clip) {
    if (!confirm(`Delete "${clip.filename}"? This permanently removes the audio and all results.`)) return;
    try {
      await api.deleteClip(clip.id);
      toast.success("Clip deleted");
      load();
    } catch (e) {
      toast.error("Delete failed", { description: String(e) });
    }
  }

  return (
    <div>
      <PageHeader
        title="Clip Library"
        description="Every processed recording, searchable and reviewable in one place."
        actions={
          <Button asChild>
            <Link to="/upload">
              <UploadIcon className="size-4" />
              Upload clip
            </Link>
          </Button>
        }
      />

      <div className="flex items-center gap-3 px-8 pt-6">
        <div className="relative w-72">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search filename…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            className="pl-9"
          />
        </div>
        <Button
          variant={needsReview ? "default" : "outline"}
          size="sm"
          onClick={() => setNeedsReview((v) => !v)}
          className="gap-1.5"
        >
          <AlertTriangle className="size-3.5" />
          Needs review
        </Button>
      </div>

      <div className="px-8 py-6">
        {clips === null ? (
          <div className="space-y-2">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-12 w-full rounded-lg" />
            ))}
          </div>
        ) : clips.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-border py-20 text-center animate-slide-up">
            <div className="flex size-12 items-center justify-center rounded-full bg-muted">
              <AudioLines className="size-6 text-muted-foreground" />
            </div>
            <div className="space-y-1">
              <p className="font-medium">No clips yet</p>
              <p className="text-sm text-muted-foreground">Upload your first recording to get started.</p>
            </div>
            <Button asChild size="sm" className="mt-2">
              <Link to="/upload">Upload a clip</Link>
            </Button>
          </div>
        ) : (
          <div className="overflow-hidden rounded-xl border border-border">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead>File</TableHead>
                  <TableHead>Duration</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Language</TableHead>
                  <TableHead>Speakers</TableHead>
                  <TableHead className="text-right">Uploaded</TableHead>
                  <TableHead className="w-10" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {clips.map((c) => (
                  <TableRow
                    key={c.id}
                    className="cursor-pointer"
                    onClick={() => navigate(`/clips/${c.id}`)}
                  >
                    <TableCell className="font-medium">
                      <div className="flex items-center gap-2">
                        {c.filename}
                        {c.needs_review && (
                          <AlertTriangle className="size-3.5 shrink-0 text-warning" />
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="tabular-nums text-muted-foreground">{fmtDuration(c.duration_s)}</TableCell>
                    <TableCell><StatusBadge status={c.status} /></TableCell>
                    <TableCell>
                      {c.language ? (
                        <Badge variant="secondary" className="uppercase">{c.language}</Badge>
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </TableCell>
                    <TableCell>
                      {c.n_speakers != null ? (
                        <span className="inline-flex items-center gap-1.5 text-muted-foreground">
                          <Users2 className="size-3.5" />
                          {c.n_speakers}
                        </span>
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right text-muted-foreground">
                      {new Date(c.created_at).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}
                    </TableCell>
                    <TableCell onClick={(e) => e.stopPropagation()}>
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon-sm" className="text-muted-foreground">
                            <MoreVertical className="size-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={() => openEdit(c)}>
                            <Pencil className="size-3.5" /> Edit
                          </DropdownMenuItem>
                          <DropdownMenuItem variant="destructive" onClick={() => removeClip(c)}>
                            <Trash2 className="size-3.5" /> Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </div>

      <Dialog open={edit != null} onOpenChange={(open) => !open && setEdit(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit clip</DialogTitle>
          </DialogHeader>
          {edit && (
            <div className="space-y-4">
              <p className="truncate text-sm text-muted-foreground">{edit.filename}</p>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">Tags (comma-separated)</label>
                <Input
                  value={edit.tags}
                  onChange={(e) => setEdit({ ...edit, tags: e.target.value })}
                  placeholder="meeting, follow-up"
                />
              </div>
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">Notes</label>
                <Textarea
                  value={edit.notes}
                  onChange={(e) => setEdit({ ...edit, notes: e.target.value })}
                  rows={4}
                />
              </div>
              <Button
                type="button"
                variant={edit.needs_review ? "default" : "outline"}
                size="sm"
                className="gap-1.5"
                onClick={() => setEdit({ ...edit, needs_review: !edit.needs_review })}
              >
                <AlertTriangle className="size-3.5" />
                {edit.needs_review ? "Flagged for review" : "Not flagged"}
              </Button>
            </div>
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setEdit(null)}>Cancel</Button>
            <Button onClick={saveEdit} disabled={saving}>{saving ? "Saving…" : "Save"}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
