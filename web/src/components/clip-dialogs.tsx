import { useEffect, useState } from "react";
import { toast } from "sonner";
import { api } from "@/api/client";
import { Dialog } from "@/components/dialog";
import { AMBER_INK, fmtAgo, fmtDur, muted } from "@/lib/ui";

export type ClipRef = {
  id: string; filename: string; duration_s?: number | null;
  n_speakers?: number | null; created_at?: string | null;
};

/** Edit a clip's metadata. Only the three things the API can actually store are offered —
 *  a field that looks editable but is dropped on save is worse than no field. */
export function EditClipDialog({ clip, onClose, onSaved }: {
  clip: ClipRef | null; onClose: () => void; onSaved?: () => void;
}) {
  const [form, setForm] = useState({ tags: "", notes: "", needs_review: false });
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!clip) return;
    api.getClip(clip.id).then((full) => setForm({
      tags: (full.tags || []).join(", "),
      notes: full.notes || "",
      needs_review: full.needs_review,
    })).catch(() => {});
  }, [clip?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  async function save() {
    if (!clip) return;
    setBusy(true);
    try {
      await api.patchClip(clip.id, {
        tags: form.tags.split(",").map((t) => t.trim()).filter(Boolean),
        notes: form.notes,
        needs_review: form.needs_review,
      });
      toast.success("Clip updated");
      onClose();
      onSaved?.();
    } catch (e) {
      toast.error("Update failed", { description: String(e) });
    } finally { setBusy(false); }
  }

  return (
    <Dialog
      open={clip != null}
      onClose={onClose}
      kicker="Library"
      title="Edit clip details"
      subject={clip ? `${clip.filename}${clip.created_at ? ` · uploaded ${fmtAgo(clip.created_at)}` : ""}` : ""}
      consequence="Metadata only. Transcript, voices and tone readings are untouched."
      confirm="Save changes"
      width="540px"
      busy={busy}
      onConfirm={save}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className="field">
          <label>Tags</label>
          <input className="input" value={form.tags} placeholder="pitwall, stint-3, radio"
                 onChange={(e) => setForm({ ...form, tags: e.target.value })} />
        </div>
        <div className="field">
          <label>
            Note <span style={{ fontWeight: 400, color: muted(50) }}>shown on the clip, not indexed for retrieval</span>
          </label>
          <textarea className="input" rows={3} style={{ resize: "vertical" }} value={form.notes}
                    onChange={(e) => setForm({ ...form, notes: e.target.value })} />
        </div>
        <div className="field">
          <label>Review flag</label>
          <div className="seg">
            <label className="seg-opt">
              <input type="radio" name="rev" checked={form.needs_review}
                     onChange={() => setForm({ ...form, needs_review: true })} />
              In the queue
            </label>
            <label className="seg-opt">
              <input type="radio" name="rev" checked={!form.needs_review}
                     onChange={() => setForm({ ...form, needs_review: false })} />
              Cleared
            </label>
          </div>
        </div>
        <span className="mono" style={{ fontSize: 11, color: muted(48) }}>
          Saved to the next job's index. Retrieval picks up the new tags within a minute.
        </span>
      </div>
    </Dialog>
  );
}

/** Delete a clip, having first counted what goes with it. The inventory is read off the
 *  stored result, so the dialog states the real cost rather than a generic warning. */
export function DeleteClipDialog({ clip, onClose, onDeleted }: {
  clip: ClipRef | null; onClose: () => void; onDeleted?: () => void;
}) {
  const [detail, setDetail] = useState<any>(null);
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setTyped("");
    setDetail(null);
    if (clip) api.getResult(clip.id).then(setDetail).catch(() => {});
  }, [clip?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  async function remove() {
    if (!clip) return;
    setBusy(true);
    try {
      await api.deleteClip(clip.id);
      toast.success("Clip deleted");
      onClose();
      onDeleted?.();
    } catch (e) {
      toast.error("Delete failed", { description: String(e) });
    } finally { setBusy(false); }
  }

  const items = detail ? [
    { label: "Audio file and its processed copy", detail: fmtDur(detail.clip?.duration_s), warn: false },
    { label: `Transcript, ${detail.utterances?.length ?? 0} turns`, detail: `${detail.words?.length ?? 0} words`, warn: false },
    { label: `Voice embeddings, ${detail.speakers?.length ?? 0}`, detail: "this clip only", warn: false },
    ...(detail.speakers || []).filter((s: any) => s.profile_id).map((s: any) => ({
      label: `Identification on ${s.display_name || s.local_label}`,
      detail: "its enrollment from this clip goes too",
      warn: true,
    })),
  ] : [];

  return (
    <Dialog
      open={clip != null}
      onClose={onClose}
      kicker="Library · destructive"
      title="Delete this clip?"
      subject={clip ? `${clip.filename} · ${fmtDur(clip.duration_s)} · ${clip.n_speakers ?? 0} voices` : ""}
      consequence="Not reversible. The audio is removed from disk on the next retention sweep."
      confirm="Delete clip"
      cancel="Keep clip"
      danger
      width="820px"
      busy={busy}
      confirmDisabled={typed !== clip?.filename}
      onConfirm={remove}
    >
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 26, alignItems: "start" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
          <p style={{ margin: 0, fontSize: 14, lineHeight: 1.55 }}>
            Deleting removes the audio and everything derived from it.
          </p>
          <div style={{ display: "flex", flexDirection: "column", borderTop: "1px solid var(--color-divider)" }}>
            {items.map((d, i) => (
              <div key={i} style={{
                display: "grid", gridTemplateColumns: "1fr auto", gap: 12, padding: "9px 2px",
                borderBottom: `1px solid ${muted(8)}`, alignItems: "baseline",
              }}>
                <span style={{ fontSize: 13.5, color: d.warn ? AMBER_INK : "var(--color-text)" }}>{d.label}</span>
                <span className="mono" style={{ fontSize: 11.5, color: muted(55) }}>{d.detail}</span>
              </div>
            ))}
            {!detail && (
              <span style={{ padding: "9px 2px", fontSize: 13, color: muted(55) }}>
                Counting what this removes…
              </span>
            )}
          </div>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 14, minWidth: 0 }}>
          <div className="blueprint hatch" style={{ padding: 12, display: "flex", flexDirection: "column", gap: 6 }}>
            <span className="kicker-sm" style={{ color: AMBER_INK }}>Knock-on effect</span>
            <span style={{ fontSize: 13, lineHeight: 1.5, color: muted(75) }}>
              Any profile enrolled from this clip loses that enrollment and has its cohesion
              recomputed from what is left. Answers in Ask that cite this clip will show the
              citation as deleted rather than dropping it silently.
            </span>
          </div>
          <div className="field">
            <label>Type the filename to confirm</label>
            <input className="input" placeholder={clip?.filename} value={typed}
                   onChange={(e) => setTyped(e.target.value)} />
          </div>
        </div>
      </div>
    </Dialog>
  );
}
