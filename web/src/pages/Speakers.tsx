import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/api/client";
import { Dialog } from "@/components/dialog";
import { audioPeaks } from "@/lib/peaks";
import { AMBER, GREEN, RED, fmtAgo, fmtDur, fmtSpeech, fmtTime, muted, num, tagTone } from "@/lib/ui";

type Profile = {
  id: string; display_name: string; status: string; n_enrollments: number;
  total_speech_s: number | null; intra_cohesion: number | null; mean_reliability: number | null;
  notes: string | null; tags: string[] | null; created_at: string; updated_at: string | null;
  n_clips: number; last_seen: string | null; heard_s: number | null;
};
type ProfileClip = {
  id: string; filename: string; duration_s: number | null; created_at: string;
  talk_share: number | null; match_result: string | null;
};
type Enrollment = {
  id: string; clip_id: string | null; duration_s: number; reliability: number;
  quality: string | null; source: string; is_outlier: boolean; cos_to_centroid: number | null;
  created_at: string;
};
type Line = { id: string; clip_id: string; start_s: number; text: string; filename: string };
type Relations = {
  co_speakers: { id: string; display_name: string; n_clips: number }[];
  topics: { topic: string; n_clips: number }[];
  entities: { name: string; kind: string; n: number }[];
  unidentified_co_voices: number;
};
type Cluster = {
  id: string; label: string; n_members: number; n_clips: number;
  total_speech_s: number | null; intra_cohesion: number | null;
};
type Candidate = {
  clip_id: string; local_label: string; filename: string; sim: number; band: "likely" | "possible";
  speech_s: number; reliability: number | null; match_result: string; current_name: string | null;
  first_s: number | null;
};
type Suggestion = {
  a_id: string; a_name: string; a_n: number; b_id: string; b_name: string; b_n: number;
  sim: number; band: "likely" | "possible";
};

const STATUSES = ["provisional", "confirmed", "locked"] as const;
// What each status actually changes, per worker/stages/identify.py: only a confirmed
// profile takes new samples from confident matches on its own.
const STATUS_HINT: Record<string, string> = {
  provisional: "Samples come only from your own assignments",
  confirmed: "Confident matches are added as samples automatically",
  locked: "Samples frozen — nothing is added automatically",
};
const MONTAGE_BARS = 40;
type Tab = "recordings" | "lines" | "connections" | "samples";
type Sort = "name" | "recent" | "speech" | "clips";

/** The envelope of the montage itself — but only once we hold the audio. Stitching a
 *  montage is expensive, so nothing is fetched until you press play; until then (and if
 *  the decode fails) the strip rests flat rather than drawing a shape nobody measured. */
function MontageBars({ peaks, active }: { peaks?: number[]; active: boolean }) {
  return (
    <>
      {Array.from({ length: MONTAGE_BARS }, (_, i) => (
        <div key={i} style={{
          flex: 1, minWidth: 2,
          height: peaks ? Math.max(2, Math.round(peaks[i] * 24)) : 2,
          background: active ? "var(--color-accent)" : "var(--color-accent-400)",
        }} />
      ))}
    </>
  );
}

/** Clusters are stored with a placeholder label until someone names them, so fall back to
 *  the id — "Voice 9a2f" is at least a handle you can tell apart from the next one. */
const clusterLabel = (c: Cluster) =>
  c.label && c.label.toLowerCase() !== "unknown" ? c.label : `Voice ${c.id.slice(0, 4)}`;

const cohesionColor = (c: number | null) =>
  c == null ? muted(30) : c >= 0.7 ? GREEN : c >= 0.5 ? AMBER : RED;

const statusTone = (s: string) =>
  s === "confirmed" ? tagTone("accent") : s === "locked" ? tagTone("warn") : tagTone("neutral");

const initials = (name: string) =>
  name.split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0]!.toUpperCase()).join("") || "?";

/** Monogram in the accent family, so a list of people scans by face rather than by text. */
function Monogram({ name, size = 34 }: { name: string; size?: number }) {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  const shades = ["var(--color-accent-600)", "var(--color-accent-2-600)", "var(--color-accent-800)",
                  "var(--color-accent-2-800)", "var(--color-neutral-700)"];
  return (
    <span aria-hidden style={{
      width: size, height: size, flex: "none", display: "grid", placeItems: "center",
      background: shades[h % shades.length], color: "#fff",
      fontFamily: "var(--font-heading)", fontSize: size * 0.42, letterSpacing: ".02em",
    }}>{initials(name)}</span>
  );
}

const errMessage = (e: unknown) => (e instanceof Error ? e.message : String(e));

export default function Speakers() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const [profiles, setProfiles] = useState<Profile[] | null>(null);
  const [clusters, setClusters] = useState<Cluster[] | null>(null);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [busy, setBusy] = useState(false);
  const [q, setQ] = useState("");
  const [sort, setSort] = useState<Sort>("recent");
  const [checked, setChecked] = useState<Set<string>>(new Set());

  const selectedId = params.get("id");
  const selected = profiles?.find((p) => p.id === selectedId) || null;
  const select = (id: string | null) => {
    const next = new URLSearchParams(params);
    id ? next.set("id", id) : next.delete("id");
    setParams(next, { replace: true });
  };

  // detail panel
  const [tab, setTab] = useState<Tab>("recordings");
  const [clips, setClips] = useState<ProfileClip[] | null>(null);
  const [lines, setLines] = useState<Line[] | null>(null);
  const [relations, setRelations] = useState<Relations | null>(null);
  const [samples, setSamples] = useState<Enrollment[] | null>(null);
  const [nameDraft, setNameDraft] = useState("");
  const [notesDraft, setNotesDraft] = useState("");
  const [tagDraft, setTagDraft] = useState("");

  // dialogs
  const [creating, setCreating] = useState(false);
  const [createName, setCreateName] = useState("");
  const [createNotes, setCreateNotes] = useState("");
  const [merging, setMerging] = useState<{ keep: string; absorb: string[] } | null>(null);
  const [deleting, setDeleting] = useState<Profile | null>(null);
  const [reassignTo, setReassignTo] = useState("");
  const [promote, setPromote] = useState<Cluster | null>(null);
  const [promoteName, setPromoteName] = useState("");
  const [promoteInto, setPromoteInto] = useState("");
  const [dismissing, setDismissing] = useState<Cluster | null>(null);

  // "find this voice elsewhere": candidates for one profile, and which are ticked
  const [finding, setFinding] = useState<Profile | null>(null);
  const [cands, setCands] = useState<Candidate[] | null>(null);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const candKey = (c: Candidate) => `${c.clip_id}:${c.local_label}`;

  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [info, setInfo] = useState<Record<string, { segments: number; seconds: number; clips: number }>>({});
  const [peaks, setPeaks] = useState<Record<string, number[]>>({});
  const audioRef = useRef<HTMLAudioElement | null>(null);

  async function load() {
    const [s, c] = await Promise.all([api.listSpeakers(), api.listClusters()]);
    setProfiles(s.items);
    setClusters(c.items);
    api.mergeSuggestions().then((r) => setSuggestions(r.items)).catch(() => setSuggestions([]));
  }
  useEffect(() => { load().catch((e) => toast.error(errMessage(e))); }, []);

  // Reset and refetch the panel whenever a different person is opened.
  useEffect(() => {
    setClips(null); setLines(null); setRelations(null); setSamples(null);
    if (!selected) return;
    setNameDraft(selected.display_name);
    setNotesDraft(selected.notes || "");
    setTagDraft("");
    const id = selected.id;
    api.speakerClips(id).then((r) => setClips(r.items)).catch(() => setClips([]));
    api.speakerUtterances(id, 60).then((r) => setLines(r.items)).catch(() => setLines([]));
    api.speakerRelations(id).then(setRelations).catch(() => setRelations(null));
    api.getSpeaker(id).then((r) => setSamples(r.enrollments)).catch(() => setSamples([]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected?.id]);

  const shown = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const list = (profiles || []).filter((p) => !needle
      || p.display_name.toLowerCase().includes(needle)
      || (p.tags || []).some((t) => t.toLowerCase().includes(needle)));
    const by: Record<Sort, (a: Profile, b: Profile) => number> = {
      name: (a, b) => a.display_name.localeCompare(b.display_name),
      recent: (a, b) => (b.last_seen || b.created_at).localeCompare(a.last_seen || a.created_at),
      speech: (a, b) => (b.total_speech_s || 0) - (a.total_speech_s || 0),
      clips: (a, b) => b.n_clips - a.n_clips,
    };
    return [...list].sort(by[sort]);
  }, [profiles, q, sort]);

  // ── writes ─────────────────────────────────────────────────────────────────

  async function patch(body: Parameters<typeof api.patchSpeaker>[1], okMsg: string) {
    if (!selected) return;
    try {
      await api.patchSpeaker(selected.id, body);
      toast.success(okMsg);
      await load();
    } catch (e: any) {
      // A rename onto someone else's name is almost always the same person twice.
      if (e?.status === 409 && e.detail?.existing_id) {
        toast.error(errMessage(e), {
          action: { label: "Merge them", onClick: () => setMerging({ keep: e.detail.existing_id, absorb: [selected.id] }) },
        });
        setNameDraft(selected.display_name);
      } else {
        toast.error("Could not save", { description: errMessage(e) });
      }
    }
  }

  async function addTag() {
    const t = tagDraft.trim();
    if (!selected || !t || (selected.tags || []).includes(t)) return setTagDraft("");
    await patch({ tags: [...(selected.tags || []), t] }, `Tagged “${t}”`);
    setTagDraft("");
  }

  async function doCreate() {
    if (!createName.trim()) return;
    setBusy(true);
    try {
      const r = await api.createSpeaker({ display_name: createName.trim(), notes: createNotes.trim() || undefined });
      toast.success(`Created ${createName.trim()}`, { description: "Assign them a voice from any clip, or name an unclaimed voice below." });
      setCreating(false); setCreateName(""); setCreateNotes("");
      await load();
      select(r.id);
    } catch (e: any) {
      if (e?.status === 409 && e.detail?.existing_id) {
        toast.error(errMessage(e));
        setCreating(false);
        select(e.detail.existing_id);
      } else toast.error("Could not create speaker", { description: errMessage(e) });
    } finally { setBusy(false); }
  }

  async function doMerge() {
    if (!merging || !merging.keep || !merging.absorb.length) return;
    setBusy(true);
    try {
      await api.mergeSpeakers(merging.keep, merging.absorb);
      const keepName = profiles?.find((p) => p.id === merging.keep)?.display_name;
      toast.success(`Merged ${merging.absorb.length} profile${merging.absorb.length === 1 ? "" : "s"} into ${keepName}`,
                    { description: "Cohesion recomputed; other clips are re-checked against the combined voice." });
      setMerging(null);
      setChecked(new Set());
      await load();
      select(merging.keep);
    } catch (e) {
      toast.error("Merge failed", { description: errMessage(e) });
    } finally { setBusy(false); }
  }

  async function doDelete() {
    if (!deleting) return;
    setBusy(true);
    try {
      await api.deleteSpeaker(deleting.id, reassignTo || undefined);
      toast.success(reassignTo ? `${deleting.display_name} folded into another profile` : `${deleting.display_name} deleted`,
                    { description: reassignTo ? undefined : "Their clip voices are unattributed again; enrolments are gone." });
      if (selectedId === deleting.id) select(reassignTo || null);
      setDeleting(null); setReassignTo("");
      await load();
    } catch (e) {
      toast.error("Delete failed", { description: errMessage(e) });
    } finally { setBusy(false); }
  }

  async function removeSample(eid: string) {
    if (!selected) return;
    try {
      await api.deleteEnrollment(selected.id, eid);
      setSamples((s) => s?.filter((x) => x.id !== eid) ?? null);
      toast.success("Sample removed", { description: "Voiceprint recomputed from what remains." });
      load();
    } catch (e) { toast.error("Could not remove sample", { description: errMessage(e) }); }
  }

  // ── unclaimed voices ──────────────────────────────────────────────────────

  // one montage at a time — two voices overlapping is exactly what makes a cluster
  // unrecognisable, which is the problem this feature exists to solve
  function stopMontage() {
    audioRef.current?.pause();
    audioRef.current = null;
    setPlayingId(null);
  }
  useEffect(() => stopMontage, []);

  async function playMontage(id: string) {
    if (playingId === id) return stopMontage();
    stopMontage();
    setLoadingId(id);
    try {
      // fetched rather than pointed at directly so the first (slow, stitching) request has
      // a spinner and a 404 "nothing usable here" can be explained instead of failing mutely
      const res = await fetch(api.clusterMontageUrl(id));
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail || `montage unavailable (${res.status})`);
      }
      setInfo((m) => ({ ...m, [id]: {
        segments: Number(res.headers.get("X-Montage-Segments") || 0),
        seconds: Number(res.headers.get("X-Montage-Seconds") || 0),
        clips: Number(res.headers.get("X-Montage-Clips") || 0),
      } }));
      const blob = await res.blob();
      audioPeaks(blob, MONTAGE_BARS)
        .then((p) => setPeaks((m) => ({ ...m, [id]: p })))
        .catch(() => {}); // an undecodable montage still plays; the strip just stays flat
      const url = URL.createObjectURL(blob);
      const el = new Audio(url);
      el.onended = () => { URL.revokeObjectURL(url); setPlayingId(null); };
      el.onerror = () => { toast.error("Could not play that montage"); setPlayingId(null); };
      audioRef.current = el;
      await el.play();
      setPlayingId(id);
    } catch (e) {
      toast.error(errMessage(e));
    } finally {
      setLoadingId(null);
    }
  }

  async function doPromote() {
    if (!promote || (!promoteName.trim() && !promoteInto)) return;
    setBusy(true);
    try {
      const r = await api.promoteCluster(promote.id, promoteInto
        ? { profile_id: promoteInto } : { display_name: promoteName.trim() });
      const who = promoteInto ? profiles?.find((p) => p.id === promoteInto)?.display_name : promoteName.trim();
      toast.success(`Voice named ${who}`, { description: "Every past appearance is re-labelled." });
      setPromote(null); setPromoteName(""); setPromoteInto("");
      await load();
      select(r.profile_id);
    } catch (e) {
      toast.error("Could not name that voice", { description: errMessage(e) });
    } finally { setBusy(false); }
  }

  async function doDismiss() {
    if (!dismissing) return;
    setBusy(true);
    try {
      await api.dismissCluster(dismissing.id);
      toast.success(`${clusterLabel(dismissing)} dismissed`);
      setDismissing(null);
      load();
    } catch (e) {
      toast.error("Could not dismiss", { description: errMessage(e) });
    } finally { setBusy(false); }
  }

  async function findElsewhere(p: Profile) {
    setFinding(p);
    setCands(null);
    try {
      const r = await api.speakerCandidates(p.id);
      setCands(r.items);
      // pre-tick only the strong ones: a "possible" should be listened to, not waved through
      setPicked(new Set(r.items.filter((c: Candidate) => c.band === "likely").map(candKey)));
    } catch (e) {
      toast.error("Could not search for this voice", { description: errMessage(e) });
      setFinding(null);
    }
  }

  async function doClaim() {
    if (!finding || !cands) return;
    const items = cands.filter((c) => picked.has(candKey(c)))
      .map((c) => ({ clip_id: c.clip_id, local_label: c.local_label }));
    if (!items.length) return;
    setBusy(true);
    try {
      const r = await api.claimVoices(finding.id, items);
      toast.success(`${r.assigned} voice${r.assigned === 1 ? "" : "s"} named ${finding.display_name}`, {
        description: `${r.enrolled} added as samples. Other clips are re-checked against the updated profile.`,
      });
      setFinding(null);
      load();
    } catch (e) {
      toast.error("Could not confirm those voices", { description: errMessage(e) });
    } finally { setBusy(false); }
  }

  const byId = (id: string) => profiles?.find((p) => p.id === id);
  const mergeKeep = merging ? byId(merging.keep) : undefined;
  const mergeAbsorb = merging ? merging.absorb.map(byId).filter(Boolean) as Profile[] : [];
  const toggleCheck = (id: string) => setChecked((cur) => {
    const n = new Set(cur); n.has(id) ? n.delete(id) : n.add(id); return n;
  });

  // ── render ─────────────────────────────────────────────────────────────────

  const smallBtn = { fontSize: 12, padding: "5px 10px" } as const;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      <header style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 20, flexWrap: "wrap" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <span className="kicker">Directory</span>
          <h2 style={{ margin: 0, fontSize: 30 }}>Speakers</h2>
          <p style={{ margin: 0, maxWidth: "62ch", fontSize: 14, color: muted(68) }}>
            People the system can name, and recurring voices that could become people. Every
            rename, merge and delete re-labels the clips behind it and is kept in the audit log.
          </p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn btn-secondary" style={smallBtn} onClick={() => navigate("/graph")}>
            View in graph
          </button>
          <button className="btn btn-primary" style={smallBtn} onClick={() => setCreating(true)}>
            + New speaker
          </button>
        </div>
      </header>

      {/* Stat strip */}
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 1,
        background: "var(--color-divider)", border: "1px solid var(--color-divider)",
      }}>
        {[
          [profiles?.length ?? "—", "enrolled people"],
          [profiles?.filter((p) => p.status === "confirmed").length ?? "—", "confirmed by a person"],
          [clusters?.length ?? "—", "voices waiting for a name"],
          [suggestions.length, "likely duplicates"],
        ].map(([v, l]) => (
          <div key={String(l)} style={{ background: "var(--color-bg)", padding: "14px 16px" }}>
            <div style={{ fontFamily: "var(--font-heading)", fontSize: 28, lineHeight: 1 }}>{v}</div>
            <div style={{ fontSize: 12, color: muted(58), marginTop: 4 }}>{l}</div>
          </div>
        ))}
      </div>

      {suggestions.length > 0 && (
        <section className="blueprint" style={{ padding: "12px 14px", display: "flex", flexDirection: "column", gap: 8,
                                                 borderLeft: `3px solid ${AMBER}` }}>
          <span className="kicker-sm" style={{ color: muted(60) }}>Probably the same person</span>
          {suggestions.slice(0, 5).map((s) => (
            <div key={s.a_id + s.b_id} style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", fontSize: 13.5 }}>
              <strong style={{ fontWeight: 500 }}>{s.a_name}</strong>
              <span style={{ color: muted(45) }}>↔</span>
              <strong style={{ fontWeight: 500 }}>{s.b_name}</strong>
              <span className="mono" style={{ fontSize: 11, color: s.band === "likely" ? GREEN : AMBER }}>
                voiceprints {num(s.sim)} · {s.band}
              </span>
              <span style={{ flex: 1 }} />
              <button className="btn btn-secondary" style={{ fontSize: 11.5, padding: "3px 9px" }}
                      onClick={() => setMerging(s.a_n >= s.b_n
                        ? { keep: s.a_id, absorb: [s.b_id] } : { keep: s.b_id, absorb: [s.a_id] })}>
                Review merge
              </button>
            </div>
          ))}
        </section>
      )}

      <section style={{
        display: "grid", gap: 20, alignItems: "start",
        gridTemplateColumns: selected ? "minmax(280px, 380px) minmax(0, 1fr)" : "1fr",
      }}>
        {/* ── directory ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 10, minWidth: 0 }}>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <input className="input" placeholder="Search names and tags" value={q}
                   onChange={(e) => setQ(e.target.value)} style={{ flex: 1, minWidth: 0 }} />
            <select className="input" value={sort} onChange={(e) => setSort(e.target.value as Sort)}
                    style={{ width: "auto" }} aria-label="Sort speakers">
              <option value="recent">Recently heard</option>
              <option value="name">Name</option>
              <option value="speech">Most speech</option>
              <option value="clips">Most recordings</option>
            </select>
          </div>

          {checked.size > 0 && (
            <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 10px",
                          background: "var(--color-surface)", border: "1px solid var(--color-text)" }}>
              <span className="mono" style={{ fontSize: 12 }}>{checked.size} SELECTED</span>
              <span style={{ flex: 1 }} />
              <button className="btn btn-secondary" style={{ fontSize: 12, padding: "3px 9px" }}
                      disabled={checked.size < 2}
                      onClick={() => {
                        // keep the one with the most evidence; the rest fold into it
                        const ids = [...checked].sort((a, b) =>
                          (byId(b)?.n_enrollments || 0) - (byId(a)?.n_enrollments || 0));
                        setMerging({ keep: ids[0], absorb: ids.slice(1) });
                      }}>
                Merge {checked.size}
              </button>
              <button className="btn btn-ghost" style={{ fontSize: 12, padding: "3px 7px" }}
                      onClick={() => setChecked(new Set())}>Clear</button>
            </div>
          )}

          <div style={{
            display: selected ? "flex" : "grid", flexDirection: "column",
            gridTemplateColumns: "repeat(auto-fill, minmax(290px, 1fr))", gap: selected ? 0 : 14,
            borderTop: selected ? "1px solid var(--color-divider)" : undefined,
          }}>
            {shown.map((p) => {
              const tone = statusTone(p.status);
              const active = p.id === selectedId;
              const low = p.intra_cohesion != null && p.intra_cohesion < 0.55;
              return (
                <article key={p.id} className={selected ? "row-hover" : "blueprint row-hover"}
                         onClick={() => select(active ? null : p.id)}
                         style={{
                           padding: selected ? "10px 8px" : 14, cursor: "pointer",
                           display: "flex", flexDirection: "column", gap: selected ? 4 : 10,
                           borderBottom: selected ? `1px solid ${muted(8)}` : undefined,
                           borderLeft: active ? "3px solid var(--color-accent)" : selected ? "3px solid transparent" : undefined,
                           background: active ? "color-mix(in srgb, var(--color-accent) 8%, transparent)" : undefined,
                         }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <input type="checkbox" checked={checked.has(p.id)} aria-label={`Select ${p.display_name}`}
                           onClick={(e) => e.stopPropagation()} onChange={() => toggleCheck(p.id)} />
                    <Monogram name={p.display_name} size={selected ? 28 : 34} />
                    <div style={{ display: "flex", flexDirection: "column", minWidth: 0, flex: 1 }}>
                      <span style={{ fontFamily: "var(--font-heading)", fontSize: selected ? 16 : 19, lineHeight: 1.1,
                                     whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        {p.display_name}
                      </span>
                      <span className="mono" style={{ fontSize: 11, color: muted(55) }}>
                        {p.n_clips} rec · {fmtSpeech(p.total_speech_s)} · {p.last_seen ? fmtAgo(p.last_seen) : "not heard yet"}
                      </span>
                    </div>
                    <span className="tag mono" style={{ border: `1px solid ${tone.border}`, color: tone.color, fontSize: 10 }}>
                      {p.status.toUpperCase()}
                    </span>
                  </div>
                  {!selected && (
                    <>
                      {(p.tags || []).length > 0 && (
                        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                          {(p.tags || []).map((t) => (
                            <span key={t} className="tag tag-neutral" style={{ border: "1px solid var(--color-neutral-300)", fontSize: 11 }}>{t}</span>
                          ))}
                        </div>
                      )}
                      <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                        <div className="mono" style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
                          <span>VOICEPRINT COHESION · {p.n_enrollments} samples</span><span>{num(p.intra_cohesion)}</span>
                        </div>
                        <div className="bar">
                          <span style={{ width: `${Math.round((p.intra_cohesion || 0) * 100)}%`, background: cohesionColor(p.intra_cohesion) }} />
                        </div>
                      </div>
                      <p style={{ margin: 0, fontSize: 12.5, lineHeight: 1.5, color: low ? AMBER : muted(62) }}>
                        {low
                          ? "Some samples disagree with the rest — this profile may hold a second voice. Prune samples or split before trusting it."
                          : p.notes ? p.notes.slice(0, 140) + (p.notes.length > 140 ? "…" : "")
                          : p.n_enrollments < 3 ? "Few samples. Matches stay suggestions until the profile firms up."
                          : "Samples agree. Identified automatically wherever the audio is reliable."}
                      </p>
                    </>
                  )}
                </article>
              );
            })}
          </div>
          {profiles?.length === 0 && (
            <p style={{ margin: 0, fontSize: 13, color: muted(55) }}>
              No speakers yet — create one, or name one of the recurring voices below.
            </p>
          )}
          {profiles && profiles.length > 0 && shown.length === 0 && (
            <p style={{ margin: 0, fontSize: 13, color: muted(55) }}>Nobody matches “{q}”.</p>
          )}
        </div>

        {/* ── detail ── */}
        {selected && (
          <div className="blueprint" style={{ padding: 18, display: "flex", flexDirection: "column", gap: 16, minWidth: 0,
                                               position: "sticky", top: 16 }}>
            <div style={{ display: "flex", gap: 14, alignItems: "flex-start" }}>
              <Monogram name={selected.display_name} size={52} />
              <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 6 }}>
                <input className="input" value={nameDraft} aria-label="Speaker name"
                       onChange={(e) => setNameDraft(e.target.value)}
                       onKeyDown={(e) => { if (e.key === "Enter") (e.target as HTMLInputElement).blur(); if (e.key === "Escape") setNameDraft(selected.display_name); }}
                       onBlur={() => {
                         const n = nameDraft.trim();
                         if (n && n !== selected.display_name) patch({ display_name: n }, `Renamed to ${n}`);
                         else setNameDraft(selected.display_name);
                       }}
                       style={{ fontFamily: "var(--font-heading)", fontSize: 24, padding: "4px 8px", border: "1px solid transparent", background: "transparent" }} />
                <div className="seg" style={{ alignSelf: "flex-start" }}>
                  {STATUSES.map((s) => (
                    <label key={s} className="seg-opt" title={STATUS_HINT[s]}>
                      <input type="radio" name="status" checked={selected.status === s}
                             onChange={() => patch({ status: s }, `Marked ${s}`)} />
                      {s}
                    </label>
                  ))}
                </div>
              </div>
              <button className="btn btn-ghost btn-icon" aria-label="Close" onClick={() => select(null)}
                      style={{ width: 28, height: 28 }}>✕</button>
            </div>

            <div className="mono" style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 1,
                                           background: "var(--color-divider)", border: "1px solid var(--color-divider)", fontSize: 11 }}>
              {[
                ["recordings", selected.n_clips],
                ["speech", fmtSpeech(selected.total_speech_s)],
                ["samples", selected.n_enrollments],
                ["cohesion", num(selected.intra_cohesion)],
              ].map(([k, v]) => (
                <div key={String(k)} style={{ background: "var(--color-bg)", padding: "8px 10px" }}>
                  <div style={{ fontSize: 16, fontFamily: "var(--font-heading)" }}>{v}</div>
                  <div style={{ color: muted(55) }}>{k}</div>
                </div>
              ))}
            </div>

            <div className="field">
              <label>Tags</label>
              <div style={{ display: "flex", gap: 5, flexWrap: "wrap", alignItems: "center" }}>
                {(selected.tags || []).map((t) => (
                  <span key={t} className="tag" style={{ border: "1px solid var(--color-accent-300)", color: "var(--color-accent-700)", display: "inline-flex", gap: 6 }}>
                    {t}
                    <button aria-label={`Remove tag ${t}`} onClick={() => patch({ tags: (selected.tags || []).filter((x) => x !== t) }, `Removed “${t}”`)}
                            style={{ border: 0, background: "none", padding: 0, cursor: "pointer", color: "inherit" }}>✕</button>
                  </span>
                ))}
                <input className="input" placeholder="+ tag" value={tagDraft} style={{ width: 110, padding: "3px 8px", fontSize: 12.5 }}
                       onChange={(e) => setTagDraft(e.target.value)}
                       onKeyDown={(e) => { if (e.key === "Enter") addTag(); }} onBlur={() => tagDraft.trim() && addTag()} />
              </div>
            </div>

            <div className="field">
              <label>Notes</label>
              <textarea className="input" rows={2} style={{ resize: "vertical" }} value={notesDraft}
                        placeholder="Role, team, how to recognise them…"
                        onChange={(e) => setNotesDraft(e.target.value)}
                        onBlur={() => notesDraft !== (selected.notes || "") && patch({ notes: notesDraft }, "Notes saved")} />
            </div>

            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              <button className="btn btn-secondary" style={smallBtn} onClick={() => findElsewhere(selected)}>
                Find this voice elsewhere
              </button>
              <button className="btn btn-secondary" style={smallBtn}
                      onClick={() => setMerging({ keep: "", absorb: [selected.id] })}>
                Merge into…
              </button>
              <button className="btn btn-secondary" style={smallBtn}
                      onClick={() => navigate(`/graph?focus=${selected.id}`)}>
                Graph
              </button>
              <span style={{ flex: 1 }} />
              <button className="btn btn-ghost" style={{ ...smallBtn, color: RED }}
                      onClick={() => { setDeleting(selected); setReassignTo(""); }}>
                Delete
              </button>
            </div>

            <div className="seg" role="tablist">
              {(["recordings", "lines", "connections", "samples"] as Tab[]).map((t) => (
                <label key={t} className="seg-opt">
                  <input type="radio" name="sptab" checked={tab === t} onChange={() => setTab(t)} />
                  {t}
                </label>
              ))}
            </div>

            {tab === "recordings" && (
              <div style={{ display: "flex", flexDirection: "column", borderTop: "1px solid var(--color-divider)" }}>
                {!clips && <p style={{ fontSize: 13, color: muted(55) }}>Loading…</p>}
                {clips?.length === 0 && <p style={{ fontSize: 13, color: muted(55) }}>Not attributed in any recording yet.</p>}
                {clips?.map((c) => (
                  <div key={c.id} className="row-hover" onClick={() => navigate(`/clips/${c.id}`)}
                       style={{ display: "flex", alignItems: "center", gap: 12, padding: "9px 4px",
                                borderBottom: `1px solid ${muted(8)}`, cursor: "pointer" }}>
                    <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 2 }}>
                      <span style={{ fontSize: 13.5, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{c.filename}</span>
                      <span className="mono" style={{ fontSize: 11, color: muted(50) }}>
                        {fmtAgo(c.created_at)} · {fmtDur(c.duration_s)}
                      </span>
                    </div>
                    {c.talk_share != null && (
                      <div style={{ width: 90, display: "flex", flexDirection: "column", gap: 3 }} title="Share of the recording's speech">
                        <span className="mono" style={{ fontSize: 10.5, color: muted(55), textAlign: "right" }}>
                          {Math.round(c.talk_share * 100)}% talk
                        </span>
                        <div className="bar" style={{ height: 4 }}><span style={{ width: `${Math.round(c.talk_share * 100)}%`, background: "var(--color-accent)" }} /></div>
                      </div>
                    )}
                    <span className="tag mono" style={{ fontSize: 10, flex: "none",
                      border: `1px solid ${c.match_result === "confident" ? "var(--color-accent)" : "var(--color-neutral-300)"}`,
                      color: c.match_result === "confident" ? "var(--color-accent-700)" : muted(55) }}>
                      {c.match_result || "unknown"}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {tab === "lines" && (
              <div style={{ display: "flex", flexDirection: "column", borderTop: "1px solid var(--color-divider)", maxHeight: 460, overflowY: "auto" }}>
                {!lines && <p style={{ fontSize: 13, color: muted(55) }}>Loading…</p>}
                {lines?.length === 0 && <p style={{ fontSize: 13, color: muted(55) }}>No transcribed lines attributed to them yet.</p>}
                {lines?.map((l) => (
                  <a key={l.id} href={`/clips/${l.clip_id}?t=${l.start_s}`}
                     onClick={(e) => { e.preventDefault(); navigate(`/clips/${l.clip_id}?t=${l.start_s}`); }}
                     className="row-hover"
                     style={{ display: "grid", gridTemplateColumns: "58px 1fr", gap: 10, padding: "9px 4px",
                              borderBottom: `1px solid ${muted(8)}`, color: "inherit", textDecoration: "none" }}>
                    <span className="mono" style={{ fontSize: 11.5, color: "var(--color-accent-700)" }}>{fmtTime(l.start_s)}</span>
                    <span style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
                      <span style={{ fontSize: 13.5, lineHeight: 1.5 }}>{l.text}</span>
                      <span className="mono" style={{ fontSize: 10.5, color: muted(48), whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{l.filename}</span>
                    </span>
                  </a>
                ))}
              </div>
            )}

            {tab === "connections" && (
              <div style={{ display: "grid", gap: 16, gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))" }}>
                {!relations && <p style={{ fontSize: 13, color: muted(55) }}>Loading…</p>}
                {relations && (
                  <>
                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                      <span className="kicker-sm" style={{ color: muted(55) }}>Shares recordings with</span>
                      {relations.co_speakers.length === 0 && (
                        <span style={{ fontSize: 12.5, color: muted(55) }}>
                          Nobody enrolled{relations.unidentified_co_voices ? ` — ${relations.unidentified_co_voices} unnamed voice${relations.unidentified_co_voices === 1 ? "" : "s"} though` : ""}.
                        </span>
                      )}
                      {relations.co_speakers.map((c) => (
                        <button key={c.id} className="btn btn-ghost" onClick={() => select(c.id)}
                                style={{ justifyContent: "space-between", display: "flex", fontSize: 13, padding: "4px 6px" }}>
                          <span>{c.display_name}</span><span className="mono" style={{ fontSize: 11, color: muted(50) }}>{c.n_clips}×</span>
                        </button>
                      ))}
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                      <span className="kicker-sm" style={{ color: muted(55) }}>Talks about</span>
                      {relations.topics.length === 0 && <span style={{ fontSize: 12.5, color: muted(55) }}>No topics tagged yet.</span>}
                      <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                        {relations.topics.map((t) => (
                          <span key={t.topic} className="tag" style={{ border: `1px dashed ${AMBER}`, fontSize: 11.5 }}>
                            {t.topic}{t.n_clips > 1 ? ` · ${t.n_clips}` : ""}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                      <span className="kicker-sm" style={{ color: muted(55) }}>Names when they speak</span>
                      {relations.entities.length === 0 && (
                        <span style={{ fontSize: 12.5, color: muted(55) }}>Nothing yet — sync the graph after new recordings.</span>
                      )}
                      <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                        {relations.entities.map((e) => (
                          <span key={e.name} className="tag" style={{ border: `1px solid ${GREEN}`, fontSize: 11.5 }}
                                title={e.kind}>
                            {e.name}{e.n > 1 ? ` · ${e.n}` : ""}
                          </span>
                        ))}
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}

            {tab === "samples" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <span style={{ fontSize: 12.5, color: muted(60) }}>
                  The voiceprint is the average of these. Remove an outlier and it is recomputed at once.
                </span>
                <div style={{ display: "flex", flexDirection: "column", borderTop: "1px solid var(--color-divider)" }}>
                  {!samples && <p style={{ fontSize: 13, color: muted(55) }}>Loading…</p>}
                  {samples?.length === 0 && <p style={{ fontSize: 13, color: muted(55) }}>No samples — this profile cannot match anyone yet.</p>}
                  {samples?.map((s) => (
                    <div key={s.id} style={{ display: "grid", gridTemplateColumns: "1fr 110px auto", gap: 12, alignItems: "center",
                                             padding: "8px 2px", borderBottom: `1px solid ${muted(8)}` }}>
                      <span className="mono" style={{ fontSize: 11.5 }}>
                        {s.source} · {s.duration_s.toFixed(1)}s · rel {num(s.reliability)}{s.quality ? ` · ${s.quality}` : ""}
                        {s.is_outlier && <span style={{ color: RED }}> · outlier</span>}
                        {s.clip_id && <> · <a href={`/clips/${s.clip_id}`} onClick={(e) => { e.preventDefault(); navigate(`/clips/${s.clip_id}`); }}>clip</a></>}
                      </span>
                      <span title="Similarity to the voiceprint" style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                        <span className="mono" style={{ fontSize: 10.5, textAlign: "right", color: muted(55) }}>{num(s.cos_to_centroid)}</span>
                        <span className="bar" style={{ height: 4 }}><span style={{ width: `${Math.round(Math.max(0, s.cos_to_centroid || 0) * 100)}%`, background: cohesionColor(s.cos_to_centroid) }} /></span>
                      </span>
                      <button className="btn btn-ghost" style={{ fontSize: 11.5, padding: "2px 7px", color: RED }}
                              disabled={(samples?.length || 0) <= 1} title={(samples?.length || 0) <= 1 ? "The last sample cannot be removed" : undefined}
                              onClick={() => removeSample(s.id)}>Remove</button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </section>

      {/* ── unclaimed voices ── */}
      <section style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
            <h4 style={{ margin: 0 }}>Voices waiting for a name · {clusters?.length ?? 0}</h4>
            <span style={{ fontSize: 12, color: muted(55) }}>Audition the montage before you name anyone.</span>
          </div>
          {/* Clustering happens one segment at a time and can only compare against the
              clusters that existed at that moment, so one person drifts across several.
              This is the pass that puts them back together, with every voice in view. */}
          <button className="btn btn-secondary" style={{ ...smallBtn, flex: "none" }}
                  disabled={busy || (clusters?.length ?? 0) < 2}
                  onClick={async () => {
                    setBusy(true);
                    try {
                      const r = await api.consolidateClusters();
                      toast.success(r.merged
                        ? `Merged ${r.merged} duplicate voice${r.merged === 1 ? "" : "s"} · ${r.remaining} left`
                        : "Nothing to merge — no two voices were close enough");
                      load();
                    } catch (e) {
                      toast.error("Could not consolidate", { description: errMessage(e) });
                    } finally { setBusy(false); }
                  }}>
            Merge duplicate voices
          </button>
        </div>
        <div style={{ display: "flex", flexDirection: "column", borderTop: "1px solid var(--color-divider)" }}>
          {(clusters || []).map((c) => {
            const enough = (c.total_speech_s || 0) >= 10;
            return (
              <div key={c.id} style={{
                display: "grid", gridTemplateColumns: "150px minmax(0,1fr) 200px auto", gap: 18,
                alignItems: "center", padding: "12px 0", borderBottom: `1px solid ${muted(8)}`,
              }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  <span style={{ fontFamily: "var(--font-heading)", fontSize: 17 }}>{clusterLabel(c)}</span>
                  <span className="mono" style={{ fontSize: 11, color: c.n_clips ? muted(50) : AMBER }}
                        title={c.n_clips ? undefined : "Every clip voice that formed this group has since been named or reprocessed"}>
                    {c.n_clips ? `in ${c.n_clips} recording${c.n_clips === 1 ? "" : "s"}` : "stale — safe to dismiss"}
                  </span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 12, minWidth: 0 }}>
                  <button className="btn btn-secondary btn-icon" style={{ width: 30, height: 30, flex: "none" }}
                          disabled={loadingId === c.id} onClick={() => playMontage(c.id)}
                          aria-label="Play montage">
                    {loadingId === c.id ? "…" : playingId === c.id ? "■" : "▶"}
                  </button>
                  <div title={peaks[c.id] ? "Peaks read from the montage" : "Press play to read the montage's peaks"}
                       style={{ display: "flex", alignItems: "flex-end", gap: 1, height: 26, flex: 1, maxWidth: 230 }}>
                    <MontageBars peaks={peaks[c.id]} active={playingId === c.id} />
                  </div>
                  {info[c.id] && (
                    <span className="mono" style={{ fontSize: 11, color: muted(50), whiteSpace: "nowrap" }}>
                      {info[c.id].segments} segs · {info[c.id].seconds.toFixed(0)}s
                    </span>
                  )}
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                  <div className="mono" style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: muted(58) }}>
                    <span>{c.n_members} segments · {fmtSpeech(c.total_speech_s)}</span>
                    <span>{c.intra_cohesion == null ? "" : num(c.intra_cohesion)}</span>
                  </div>
                  {/* a single-segment cluster has nothing to agree with itself about, so
                      there is no cohesion to draw — an empty bar would imply zero */}
                  {c.intra_cohesion != null && (
                    <div className="bar" style={{ height: 5 }}>
                      <span style={{ width: `${Math.round(c.intra_cohesion * 100)}%`, background: cohesionColor(c.intra_cohesion) }} />
                    </div>
                  )}
                </div>
                <div style={{ display: "flex", gap: 6, justifyContent: "flex-end" }}>
                  <button className="btn btn-secondary" style={{ ...smallBtn, opacity: enough ? 1 : 0.45 }}
                          disabled={!enough} title={enough ? undefined : "Under 10s of speech — too little to name reliably"}
                          onClick={() => { setPromote(c); setPromoteName(""); setPromoteInto(""); }}>
                    Name
                  </button>
                  <button className="btn btn-ghost" style={{ ...smallBtn, color: muted(60) }}
                          onClick={() => setDismissing(c)}>Dismiss</button>
                </div>
              </div>
            );
          })}
          {clusters?.length === 0 && (
            <p style={{ margin: 0, padding: "16px 0", fontSize: 13, color: muted(55) }}>No unnamed voices right now.</p>
          )}
        </div>
      </section>

      {/* ── dialogs ── */}

      <Dialog open={creating} onClose={() => setCreating(false)} kicker="Speakers" title="New speaker"
              consequence="Creates an empty profile. It starts matching once it has a voice sample — assign one from any clip."
              confirm="Create" busy={busy} confirmDisabled={!createName.trim()} onConfirm={doCreate}>
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div className="field">
            <label>Name</label>
            <input className="input" autoFocus value={createName} placeholder="e.g. Marta Kolar"
                   onChange={(e) => setCreateName(e.target.value)}
                   onKeyDown={(e) => { if (e.key === "Enter") doCreate(); }} />
          </div>
          <div className="field">
            <label>Notes <span style={{ fontWeight: 400, color: muted(50) }}>optional</span></label>
            <textarea className="input" rows={2} value={createNotes} onChange={(e) => setCreateNotes(e.target.value)} />
          </div>
        </div>
      </Dialog>

      <Dialog
        open={merging != null}
        onClose={() => setMerging(null)}
        kicker="Speakers"
        title={mergeAbsorb.length > 1 ? `Merge ${mergeAbsorb.length + 1} profiles` : "Merge two profiles"}
        subject={mergeAbsorb.map((p) => p.display_name).join(", ") + (mergeKeep ? ` → ${mergeKeep.display_name}` : "")}
        consequence="Samples, recordings, tags and notes move to the kept profile. Cohesion is recomputed at once; other clips are re-checked."
        confirm="Merge profiles"
        width="600px"
        busy={busy}
        confirmDisabled={!merging?.keep || !merging.absorb.length || merging.absorb.includes(merging.keep)}
        onConfirm={doMerge}
      >
        {merging && (
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <p style={{ margin: 0, fontSize: 14, lineHeight: 1.55, color: muted(72) }}>
              If these are not the same person, cohesion will drop and matches will get worse —
              the number after the merge tells you which happened.
            </p>
            <div className="field">
              <label>Keep</label>
              <select className="input" value={merging.keep}
                      onChange={(e) => {
                        const keep = e.target.value;
                        // the old keeper joins the merged set rather than vanishing from it
                        const absorb = [...new Set([...merging.absorb, merging.keep].filter((x) => x && x !== keep))];
                        setMerging({ keep, absorb });
                      }}>
                <option value="">— choose who to keep —</option>
                {(profiles || []).map((p) => <option key={p.id} value={p.id}>{p.display_name}</option>)}
              </select>
            </div>
            <div className="field">
              <label>Fold in</label>
              <div style={{ display: "flex", flexDirection: "column", maxHeight: 180, overflowY: "auto",
                            border: "1px solid var(--color-divider)" }}>
                {(profiles || []).filter((p) => p.id !== merging.keep).map((p) => (
                  <label key={p.id} style={{ display: "flex", gap: 10, alignItems: "center", padding: "6px 10px", fontSize: 13.5 }}>
                    <input type="checkbox" checked={merging.absorb.includes(p.id)}
                           onChange={() => setMerging({ ...merging, absorb: merging.absorb.includes(p.id)
                             ? merging.absorb.filter((x) => x !== p.id) : [...merging.absorb, p.id] })} />
                    <span style={{ flex: 1 }}>{p.display_name}</span>
                    <span className="mono" style={{ fontSize: 11, color: muted(50) }}>{p.n_enrollments} samples · {fmtSpeech(p.total_speech_s)}</span>
                  </label>
                ))}
              </div>
            </div>
            {mergeKeep && mergeAbsorb.length > 0 && (
              <div className="blueprint" style={{ padding: 12, display: "flex", flexDirection: "column", gap: 8 }}>
                <span className="kicker-sm" style={{ color: muted(55) }}>After the merge</span>
                {[
                  ["samples", mergeKeep.n_enrollments, mergeKeep.n_enrollments + mergeAbsorb.reduce((n, p) => n + p.n_enrollments, 0)],
                  ["recordings", mergeKeep.n_clips, mergeKeep.n_clips + mergeAbsorb.reduce((n, p) => n + p.n_clips, 0)],
                ].map(([k, a, b]) => (
                  <div key={String(k)} className="mono" style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                    <span style={{ color: muted(60) }}>{k}</span>
                    <span>{a} → <span style={{ color: GREEN }}>{k === "recordings" ? `up to ${b}` : b}</span></span>
                  </div>
                ))}
                <div className="mono" style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                  <span style={{ color: muted(60) }}>cohesion</span>
                  <span>{num(mergeKeep.intra_cohesion)} → recomputed from the combined set</span>
                </div>
              </div>
            )}
          </div>
        )}
      </Dialog>

      <Dialog
        open={deleting != null}
        onClose={() => setDeleting(null)}
        kicker="Speakers"
        title={deleting ? `Delete ${deleting.display_name}?` : ""}
        subject={deleting ? `${deleting.n_enrollments} samples · ${deleting.n_clips} recordings` : ""}
        consequence={reassignTo
          ? "Their recordings and samples move to the chosen profile; this name disappears."
          : "Samples are deleted and every voice they named becomes unattributed. Recordings and transcripts stay."}
        confirm={reassignTo ? "Fold in and delete" : "Delete speaker"}
        danger
        busy={busy}
        onConfirm={doDelete}
      >
        <div className="field">
          <label>Hand their voices to someone else instead <span style={{ fontWeight: 400, color: muted(50) }}>optional</span></label>
          <select className="input" value={reassignTo} onChange={(e) => setReassignTo(e.target.value)}>
            <option value="">— no, just unattribute them —</option>
            {(profiles || []).filter((p) => p.id !== deleting?.id)
              .map((p) => <option key={p.id} value={p.id}>{p.display_name}</option>)}
          </select>
        </div>
      </Dialog>

      <Dialog
        open={promote != null}
        onClose={() => setPromote(null)}
        kicker="Speakers"
        title="Name this recurring voice"
        subject={promote ? `${clusterLabel(promote)} · ${promote.n_members} segments · ${fmtSpeech(promote.total_speech_s)}` : ""}
        consequence="Re-labels every past appearance of this voice, and it is recognised in future clips."
        confirm={promoteInto ? "Add to this person" : "Create profile"}
        width="560px"
        busy={busy}
        confirmDisabled={!promoteName.trim() && !promoteInto}
        onConfirm={doPromote}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div className="blueprint" style={{ padding: 12, display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <button className="btn btn-secondary btn-icon" style={{ width: 30, height: 30, flex: "none" }}
                      onClick={() => promote && playMontage(promote.id)}>
                {playingId === promote?.id ? "■" : "▶"}
              </button>
              <div style={{ display: "flex", alignItems: "flex-end", gap: 1, height: 26, flex: 1 }}>
                <MontageBars peaks={promote ? peaks[promote.id] : undefined} active={playingId === promote?.id} />
              </div>
              <span className="mono" style={{ fontSize: 11, color: muted(55), flex: "none" }}>
                montage · {fmtSpeech(promote?.total_speech_s)}
              </span>
            </div>
            <span style={{ fontSize: 12.5, color: muted(62) }}>
              Listen before you name. Cohesion {num(promote?.intra_cohesion)} — how much the segments agree.
            </span>
          </div>
          <div className="field">
            <label>Someone already enrolled</label>
            <select className="input" value={promoteInto} onChange={(e) => { setPromoteInto(e.target.value); if (e.target.value) setPromoteName(""); }}>
              <option value="">— no, a new person —</option>
              {(profiles || []).map((p) => <option key={p.id} value={p.id}>{p.display_name}</option>)}
            </select>
          </div>
          {!promoteInto && (
            <div className="field">
              <label>New person's name</label>
              <input className="input" autoFocus value={promoteName}
                     onChange={(e) => setPromoteName(e.target.value)} placeholder="e.g. Marta Kolar" />
            </div>
          )}
        </div>
      </Dialog>

      <Dialog
        open={dismissing != null}
        onClose={() => setDismissing(null)}
        kicker="Speakers"
        title="Dismiss this voice?"
        subject={dismissing ? `${clusterLabel(dismissing)} · ${dismissing.n_members} segments` : ""}
        consequence="The grouping is forgotten; the recordings and their transcripts are untouched. If the voice keeps turning up, it will form a new group."
        confirm="Dismiss"
        busy={busy}
        onConfirm={doDismiss}
      />

      <Dialog
        open={finding != null}
        onClose={() => setFinding(null)}
        kicker="Speakers"
        title={finding ? `Where else does ${finding.display_name} speak?` : ""}
        subject={cands ? `${cands.length} unattributed voice${cands.length === 1 ? "" : "s"} close to this profile` : "searching…"}
        consequence="Each confirmed voice is named and, where reliable, added as a sample — then every other clip is re-checked."
        confirm={`Confirm ${picked.size} as ${finding?.display_name ?? ""}`}
        confirmDisabled={!picked.size}
        busy={busy}
        width="720px"
        onConfirm={doClaim}
      >
        {!cands && <p style={{ margin: 0, fontSize: 14, color: muted(60) }}>Comparing against every stored voice…</p>}
        {cands?.length === 0 && (
          <p style={{ margin: 0, fontSize: 14, color: muted(60) }}>
            No other voice in the corpus comes within the suggestion range of this profile.
          </p>
        )}
        {cands && cands.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", borderTop: "1px solid var(--color-divider)" }}>
            {cands.map((c) => {
              const k = candKey(c);
              return (
                <label key={k} style={{
                  display: "grid", gridTemplateColumns: "22px 1fr 150px 80px", gap: 12, alignItems: "center",
                  padding: "10px 2px", borderBottom: `1px solid ${muted(8)}`, cursor: "pointer",
                }}>
                  <input type="checkbox" checked={picked.has(k)} onChange={() => setPicked((s) => {
                    const n = new Set(s); n.has(k) ? n.delete(k) : n.add(k); return n;
                  })} />
                  <span style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
                    <span style={{ fontSize: 13.5, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                      {c.filename}
                    </span>
                    <span className="mono" style={{ fontSize: 11, color: muted(52) }}>
                      {c.local_label.replace("SPEAKER_", "Voice ")} · {fmtSpeech(c.speech_s)}
                      {c.current_name ? ` · now: ${c.current_name} (${c.match_result})` : " · unnamed"}
                      {" · "}
                      <a href={`/clips/${c.clip_id}?t=${c.first_s ?? 0}`} target="_blank" rel="noreferrer"
                         onClick={(e) => e.stopPropagation()}>listen</a>
                    </span>
                  </span>
                  <span style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    <span className="mono" style={{ fontSize: 11, display: "flex", justifyContent: "space-between" }}>
                      <span>SIMILARITY</span><span>{num(c.sim)}</span>
                    </span>
                    <span className="bar"><span style={{
                      width: `${Math.round(Math.max(0, c.sim) * 100)}%`,
                      background: c.band === "likely" ? GREEN : AMBER,
                    }} /></span>
                  </span>
                  <span className="tag mono" style={{
                    fontSize: 10, justifySelf: "end",
                    border: `1px solid ${c.band === "likely" ? GREEN : AMBER}`,
                    color: c.band === "likely" ? GREEN : AMBER,
                  }}>{c.band}</span>
                </label>
              );
            })}
          </div>
        )}
      </Dialog>
    </div>
  );
}
