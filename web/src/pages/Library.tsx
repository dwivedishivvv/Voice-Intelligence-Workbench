import { useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/api/client";
import { EditClipDialog, DeleteClipDialog, type ClipRef } from "@/components/clip-dialogs";
import { AMBER, RED, fmtAgo, fmtDur, muted, tagTone } from "@/lib/ui";

type Clip = {
  id: string; filename: string; duration_s: number; status: string;
  language: string; n_speakers: number; needs_review: boolean;
  created_at: string; error_code: string | null;
};

const STATUS_TONE = (status: string) =>
  status === "COMPLETE" ? tagTone("accent")
  : status === "REJECTED" ? tagTone("warn")
  : status === "FAILED" || status === "DEAD" ? tagTone("bad")
  : tagTone("neutral");

/** Why a clip is sitting in the queue, in the pipeline's own terms. A rejection states its
 *  code; a flag means identification hesitated or the audio graded poor — postprocess sets
 *  needs_review for exactly those two reasons. */
function reasonFor(c: Clip) {
  if (c.error_code) return { text: c.error_code.replaceAll("_", " "), color: RED };
  if (c.needs_review) return { text: "Identification hesitated, or audio graded poor", color: AMBER };
  return { text: "—", color: "transparent" };
}

export default function Library() {
  const { pathname } = useLocation();
  const inQueue = pathname === "/review";
  const [clips, setClips] = useState<Clip[] | null>(null);
  const [totals, setTotals] = useState({ all: 0, review: 0 });
  const [stats, setStats] = useState<any>(null);
  const [q, setQ] = useState("");
  const [edit, setEdit] = useState<ClipRef | null>(null);
  const [del, setDel] = useState<ClipRef | null>(null);
  const navigate = useNavigate();

  async function load() {
    const params: Record<string, string> = { limit: "100" };
    if (q) params.q = q;
    if (inQueue) params.needs_review = "true";
    const [res, all, review] = await Promise.all([
      api.listClips(params),
      api.listClips({ limit: "1" }),
      api.listClips({ limit: "1", needs_review: "true" }),
    ]);
    setClips(res.items);
    setTotals({ all: all.total, review: review.total });
  }

  useEffect(() => {
    load().catch((e) => toast.error(String(e)));
    api.stats().then(setStats).catch(() => {});
    const t = setInterval(() => load().catch(() => {}), 5000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q, inQueue]);

  const results = stats?.identification_results || {};
  const tiles = inQueue
    ? [
        { value: results.abstained ?? 0, label: "abstained — declined to judge" },
        { value: results.suggested ?? 0, label: "suggested, awaiting confirmation" },
        { value: results.unknown ?? 0, label: "judged, no profile matched" },
        { value: totals.review, label: "clips flagged for review" },
      ]
    : [
        { value: totals.all, label: "recordings processed" },
        { value: stats?.by_status?.COMPLETE ?? 0, label: "completed cleanly" },
        { value: totals.review, label: "flagged for review" },
        { value: results.confident ?? 0, label: "voices named with confidence" },
      ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
      <header style={{
        display: "flex", alignItems: "flex-end", justifyContent: "space-between",
        gap: 20, flexWrap: "wrap",
      }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <span className="kicker">{inQueue ? "Curator" : "Library"}</span>
          <h2 style={{ margin: 0, fontSize: 30 }}>{inQueue ? "Review queue" : "All recordings"}</h2>
          <p style={{ margin: 0, maxWidth: "60ch", fontSize: 14, color: muted(68) }}>
            {inQueue
              ? "Clips where the pipeline declined, hesitated, or graded the audio poor. A correction here re-enrolls the voice and improves every clip processed after it."
              : `Every processed recording. ${totals.review} are flagged for review — the pipeline declined, hesitated, or graded the audio poor on those.`}
          </p>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <input className="input" placeholder="Filter by filename" style={{ width: 190 }}
                 value={q} onChange={(e) => setQ(e.target.value)} />
          <div className="seg">
            <label className="seg-opt">
              <input type="radio" name="lib" checked={inQueue} onChange={() => navigate("/review")} />
              Needs review · {totals.review}
            </label>
            <label className="seg-opt">
              <input type="radio" name="lib" checked={!inQueue} onChange={() => navigate("/")} />
              All clips · {totals.all}
            </label>
          </div>
        </div>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
        {tiles.map((s) => (
          <div key={s.label} className="blueprint"
               style={{ padding: "13px 14px", display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontFamily: "var(--font-heading)", fontSize: 30, lineHeight: 1 }}>{s.value}</span>
            <span style={{ fontSize: 12, color: muted(62) }}>{s.label}</span>
          </div>
        ))}
      </div>

      <table className="table">
        <thead>
          <tr>
            <th>Clip</th><th>Why it is here</th><th>Status</th><th>Language</th>
            <th>Voices</th><th>Duration</th><th>Uploaded</th><th />
          </tr>
        </thead>
        <tbody>
          {(clips || []).map((c) => {
            const reason = reasonFor(c);
            const tone = STATUS_TONE(c.status);
            return (
              <tr key={c.id}>
                <td className="mono" style={{ fontSize: 12.5 }}>
                  <a href={`/clips/${c.id}`}
                     onClick={(e) => { e.preventDefault(); navigate(`/clips/${c.id}`); }}>
                    {c.filename}
                  </a>
                </td>
                <td>
                  <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <span style={{ width: 9, height: 9, flex: "none", background: reason.color }} />
                    <span style={{ fontSize: 13 }}>{reason.text}</span>
                  </span>
                </td>
                <td>
                  <span className="tag mono" style={{ border: `1px solid ${tone.border}`, color: tone.color }}>
                    {c.status}
                  </span>
                </td>
                <td className="mono" style={{ fontSize: 12.5 }}>
                  {c.language ? c.language.toUpperCase() : "—"}
                </td>
                <td className="mono" style={{ fontSize: 12.5 }}>{c.n_speakers ?? "—"}</td>
                <td className="mono" style={{ fontSize: 12.5 }}>{fmtDur(c.duration_s)}</td>
                <td style={{ fontSize: 12.5, color: muted(55) }}>{fmtAgo(c.created_at)}</td>
                <td style={{ textAlign: "right" }}>
                  <span style={{ display: "flex", gap: 6, justifyContent: "flex-end" }}>
                    <button className="btn btn-secondary" style={{ fontSize: 12, padding: "4px 9px" }}
                            onClick={() => navigate(`/clips/${c.id}`)}>Review</button>
                    <button className="btn btn-ghost" style={{ fontSize: 12, padding: "4px 7px" }}
                            onClick={() => setEdit(c)}>Edit</button>
                    <button className="btn btn-ghost" style={{ fontSize: 12, padding: "4px 7px", color: RED }}
                            onClick={() => setDel(c)}>Delete</button>
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      {clips !== null && clips.length === 0 && (
        <p style={{ margin: 0, padding: "40px 0", textAlign: "center", fontSize: 13, color: muted(55) }}>
          {inQueue
            ? "Nothing is waiting on a curator right now."
            : "No recordings yet — upload some audio to get started."}
        </p>
      )}

      <EditClipDialog clip={edit} onClose={() => setEdit(null)} onSaved={load} />
      <DeleteClipDialog clip={del} onClose={() => setDel(null)} onDeleted={load} />
    </div>
  );
}
