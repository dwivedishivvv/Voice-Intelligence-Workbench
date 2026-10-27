import { useEffect, useState, type FormEvent } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/api/client";
import { EditClipDialog, DeleteClipDialog, type ClipRef } from "@/components/clip-dialogs";
import { TagChip } from "@/components/tag-chip";
import { VoiceSearch } from "@/components/voice-search";
import { AMBER, RED, fmtAgo, fmtDur, fmtTime, muted, tagTone } from "@/lib/ui";

type Clip = {
  id: string; filename: string; duration_s: number; status: string;
  language: string; n_speakers: number; needs_review: boolean;
  created_at: string; error_code: string | null;
  tags: string[] | null; auto_tags: string[] | null;
};
type TagCount = { tag: string; source: "manual" | "auto"; n: number };
type Hit = {
  utterance_id: string; clip_id: string; filename: string | null;
  start_s: number; text: string; score: number;
};

const ROW_TAGS = 4;   // chips shown per row before collapsing into "+n"

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
  const [tagCounts, setTagCounts] = useState<TagCount[]>([]);
  const [selTags, setSelTags] = useState<string[]>([]);
  // Transcript search: `said` is the box, `hits` the last result (null = not searching).
  const [said, setSaid] = useState("");
  const [mode, setMode] = useState("hybrid");
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [voiceOpen, setVoiceOpen] = useState(false);
  // Row selection for bulk actions. Cleared whenever the visible list changes shape.
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [bulkTag, setBulkTag] = useState("");
  const [bulkBusy, setBulkBusy] = useState(false);
  const [confirmBulkDelete, setConfirmBulkDelete] = useState(false);
  const navigate = useNavigate();

  async function load() {
    const params: Record<string, string> = { limit: "100" };
    if (q) params.q = q;
    if (inQueue) params.needs_review = "true";
    const [res, all, review] = await Promise.all([
      api.listClips(params, selTags),
      api.listClips({ limit: "1" }),
      api.listClips({ limit: "1", needs_review: "true" }),
    ]);
    setClips(res.items);
    setTotals({ all: all.total, review: review.total });
    api.tagCounts().then((r) => setTagCounts(r.items)).catch(() => {});
  }

  useEffect(() => {
    load().catch((e) => toast.error(String(e)));
    api.stats().then(setStats).catch(() => {});
    const t = setInterval(() => load().catch(() => {}), 5000);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [q, inQueue, selTags]);

  useEffect(() => { setSel(new Set()); }, [q, inQueue, selTags]);

  const toggleRow = (id: string) => setSel((cur) => {
    const next = new Set(cur);
    next.has(id) ? next.delete(id) : next.add(id);
    return next;
  });
  const allSelected = !!clips?.length && clips.every((c) => sel.has(c.id));

  async function bulk(action: string, tag?: string) {
    if (!sel.size) return;
    setBulkBusy(true);
    try {
      const r = await api.bulkClips([...sel], action, tag);
      toast.success(`${r.n} clip${r.n === 1 ? "" : "s"} updated`);
      if (action === "delete") setSel(new Set());
      if (action === "add_tag" || action === "remove_tag") setBulkTag("");
      await load();
    } catch (e) {
      toast.error("Bulk action failed", { description: String(e) });
    } finally {
      setBulkBusy(false);
      setConfirmBulkDelete(false);
    }
  }

  const toggleTag = (t: string) =>
    setSelTags((cur) => (cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t]));

  async function runSearch(e?: FormEvent) {
    e?.preventDefault();
    if (!said.trim()) { setHits(null); return; }
    setSearching(true);
    try {
      setHits((await api.search(said.trim(), { mode, tags: selTags })).items);
    } catch (err) {
      toast.error("Search failed", { description: String(err) });
    } finally { setSearching(false); }
  }

  // re-run an open search when the tag filter changes, so the two never disagree
  useEffect(() => { if (hits) runSearch(); }, [selTags, mode]); // eslint-disable-line

  const manualTags = tagCounts.filter((t) => t.source === "manual");
  const autoTags = tagCounts.filter((t) => t.source === "auto");

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
              <input type="radio" name="lib" checked={!inQueue} onChange={() => navigate("/library")} />
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

      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <form onSubmit={runSearch} style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <input className="input" style={{ flex: "1 1 320px" }}
                 placeholder="Search what was said — keywords, or a description of the moment"
                 value={said} onChange={(e) => setSaid(e.target.value)} />
          <select className="input" style={{ width: 150 }} value={mode}
                  onChange={(e) => setMode(e.target.value)} title="How the search ranks lines">
            <option value="hybrid">Words + meaning</option>
            <option value="fts">Exact words</option>
            <option value="semantic">Meaning only</option>
          </select>
          <button className="btn btn-primary" type="submit" disabled={searching}>
            {searching ? "Searching…" : "Search"}
          </button>
          {hits && (
            <button className="btn btn-ghost" type="button"
                    onClick={() => { setHits(null); setSaid(""); }}>Clear</button>
          )}
          <button className="btn btn-secondary" type="button" onClick={() => setVoiceOpen((v) => !v)}>
            {voiceOpen ? "Close voice search" : "Search by voice"}
          </button>
          <select className="input" style={{ width: 210 }} value=""
                  onChange={(e) => e.target.value && toggleTag(e.target.value)}>
            <option value="">Filter by tag…</option>
            {manualTags.length > 0 && (
              <optgroup label="Your tags">
                {manualTags.map((t) => (
                  <option key={`m-${t.tag}`} value={t.tag}>{t.tag} ({t.n})</option>
                ))}
              </optgroup>
            )}
            {autoTags.length > 0 && (
              <optgroup label="Pipeline tags">
                {autoTags.map((t) => (
                  <option key={`a-${t.tag}`} value={t.tag}>{t.tag} ({t.n})</option>
                ))}
              </optgroup>
            )}
          </select>
        </form>
        {selTags.length > 0 && (
          <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap", fontSize: 12.5 }}>
            <span style={{ color: muted(55) }}>Only clips tagged</span>
            {selTags.map((t) => (
              <TagChip key={t} tag={t} active auto={autoTags.some((a) => a.tag === t)}
                       onClick={() => toggleTag(t)} />
            ))}
            <button className="btn btn-ghost" style={{ fontSize: 12, padding: "2px 7px" }}
                    onClick={() => setSelTags([])}>clear</button>
          </div>
        )}
      </div>

      {voiceOpen && <VoiceSearch />}

      {hits && (
        <section style={{ display: "flex", flexDirection: "column", borderTop: "1px solid var(--color-divider)" }}>
          <div className="mono" style={{ fontSize: 11, color: muted(55), padding: "8px 0" }}>
            {hits.length} LINE{hits.length === 1 ? "" : "S"} · RANKED BY {
              mode === "fts" ? "WORD MATCH" : mode === "semantic" ? "MEANING" : "WORDS + MEANING (RANK FUSION)"}
          </div>
          {hits.map((h) => (
            <a key={h.utterance_id} href={`/clips/${h.clip_id}?t=${h.start_s}`}
               onClick={(e) => { e.preventDefault(); navigate(`/clips/${h.clip_id}?t=${h.start_s}`); }}
               style={{
                 display: "grid", gridTemplateColumns: "minmax(0, 220px) 64px 1fr", gap: 14,
                 padding: "10px 0", borderBottom: `1px solid ${muted(8)}`, alignItems: "baseline",
                 color: "inherit", textDecoration: "none",
               }}>
              <span className="mono" style={{ fontSize: 12, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {h.filename || h.clip_id.slice(0, 8)}
              </span>
              <span className="mono" style={{ fontSize: 12, color: muted(55) }}>{fmtTime(h.start_s)}</span>
              <span style={{ fontSize: 14 }}>{h.text}</span>
            </a>
          ))}
          {hits.length === 0 && (
            <p style={{ margin: 0, padding: "24px 0", fontSize: 13, color: muted(55) }}>
              No line matched. Semantic search only covers fully processed clips; try "Exact words" for
              a name the transcript may have spelled differently.
            </p>
          )}
        </section>
      )}

      {!hits && sel.size > 0 && (
        <div style={{
          position: "sticky", top: 0, zIndex: 5, display: "flex", gap: 8, alignItems: "center",
          flexWrap: "wrap", padding: "10px 12px", background: "var(--color-surface)",
          border: "1px solid var(--color-text)",
        }}>
          <span className="mono" style={{ fontSize: 12, marginRight: 6 }}>
            {sel.size} SELECTED
          </span>
          <button className="btn btn-secondary" style={{ fontSize: 12, padding: "4px 9px" }}
                  disabled={bulkBusy} onClick={() => bulk("reprocess")}>Reprocess</button>
          <button className="btn btn-secondary" style={{ fontSize: 12, padding: "4px 9px" }}
                  disabled={bulkBusy} onClick={() => bulk("mark_reviewed")}>Clear review flag</button>
          <button className="btn btn-secondary" style={{ fontSize: 12, padding: "4px 9px" }}
                  disabled={bulkBusy} onClick={() => bulk("flag_review")}>Flag for review</button>
          <span style={{ display: "flex", gap: 4, alignItems: "center" }}>
            <input className="input" placeholder="tag" value={bulkTag} style={{ width: 120, padding: "4px 8px" }}
                   onChange={(e) => setBulkTag(e.target.value)}
                   onKeyDown={(e) => { if (e.key === "Enter" && bulkTag.trim()) bulk("add_tag", bulkTag.trim()); }} />
            <button className="btn btn-ghost" style={{ fontSize: 12, padding: "4px 7px" }}
                    disabled={bulkBusy || !bulkTag.trim()} onClick={() => bulk("add_tag", bulkTag.trim())}>+ Add</button>
            <button className="btn btn-ghost" style={{ fontSize: 12, padding: "4px 7px" }}
                    disabled={bulkBusy || !bulkTag.trim()} onClick={() => bulk("remove_tag", bulkTag.trim())}>− Remove</button>
          </span>
          <span style={{ flex: 1 }} />
          {confirmBulkDelete ? (
            <>
              <span style={{ fontSize: 12.5, color: RED }}>
                Delete {sel.size} clip{sel.size === 1 ? "" : "s"}, their audio and transcripts?
              </span>
              <button className="btn btn-primary" style={{ fontSize: 12, padding: "4px 9px", background: RED, borderColor: RED }}
                      disabled={bulkBusy} onClick={() => bulk("delete")}>Delete for good</button>
              <button className="btn btn-ghost" style={{ fontSize: 12, padding: "4px 7px" }}
                      onClick={() => setConfirmBulkDelete(false)}>Cancel</button>
            </>
          ) : (
            <button className="btn btn-ghost" style={{ fontSize: 12, padding: "4px 7px", color: RED }}
                    disabled={bulkBusy} onClick={() => setConfirmBulkDelete(true)}>Delete…</button>
          )}
          <button className="btn btn-ghost" style={{ fontSize: 12, padding: "4px 7px" }}
                  onClick={() => setSel(new Set())}>Clear</button>
        </div>
      )}

      {!hits && (
      <table className="table">
        <thead>
          <tr>
            <th style={{ width: 28 }}>
              <input type="checkbox" aria-label="Select all" checked={allSelected}
                     onChange={() => setSel(allSelected ? new Set() : new Set((clips || []).map((c) => c.id)))} />
            </th>
            <th>Clip</th><th>Why it is here</th><th>Tags</th><th>Status</th><th>Language</th>
            <th>Voices</th><th>Duration</th><th>Uploaded</th><th />
          </tr>
        </thead>
        <tbody>
          {(clips || []).map((c) => {
            const reason = reasonFor(c);
            const tone = STATUS_TONE(c.status);
            const rowTags = [
              ...(c.tags || []).map((t) => ({ t, auto: false })),
              ...(c.auto_tags || []).map((t) => ({ t, auto: true })),
            ];
            return (
              <tr key={c.id} style={sel.has(c.id)
                ? { background: "color-mix(in srgb, var(--color-accent) 7%, transparent)" } : undefined}>
                <td>
                  <input type="checkbox" aria-label={`Select ${c.filename}`} checked={sel.has(c.id)}
                         onChange={() => toggleRow(c.id)} />
                </td>
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
                  <span style={{ display: "flex", gap: 4, flexWrap: "wrap", maxWidth: 300 }}
                        title={rowTags.map((x) => x.t).join(", ")}>
                    {rowTags.slice(0, ROW_TAGS).map(({ t, auto }) => (
                      <TagChip key={`${auto}-${t}`} tag={t} auto={auto}
                               active={selTags.includes(t)} onClick={() => toggleTag(t)} />
                    ))}
                    {rowTags.length > ROW_TAGS && (
                      <span className="mono" style={{ fontSize: 10.5, color: muted(50) }}>
                        +{rowTags.length - ROW_TAGS}
                      </span>
                    )}
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
      )}

      {!hits && clips !== null && clips.length === 0 && (
        <p style={{ margin: 0, padding: "40px 0", textAlign: "center", fontSize: 13, color: muted(55) }}>
          {selTags.length
            ? "No clip carries every selected tag."
            : inQueue
            ? "Nothing is waiting on a curator right now."
            : "No recordings yet — upload some audio to get started."}
        </p>
      )}

      <EditClipDialog clip={edit} onClose={() => setEdit(null)} onSaved={load} />
      <DeleteClipDialog clip={del} onClose={() => setDel(null)} onDeleted={load} />
    </div>
  );
}
