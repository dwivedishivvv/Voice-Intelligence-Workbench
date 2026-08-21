import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/api/client";
import { Dialog } from "@/components/dialog";
import { audioPeaks } from "@/lib/peaks";
import { AMBER, GREEN, RED, fmtDur, fmtSpeech, muted, num, tagTone } from "@/lib/ui";

type ProfileClip = {
  id: string; filename: string; duration_s: number | null; created_at: string;
  talk_share: number | null; match_result: string | null;
};

type Profile = {
  id: string; display_name: string; status: string; n_enrollments: number;
  total_speech_s: number | null; intra_cohesion: number | null; mean_reliability: number | null;
};
type Cluster = {
  id: string; label: string; n_members: number;
  total_speech_s: number | null; intra_cohesion: number | null;
};

const MONTAGE_BARS = 40;

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

export default function Speakers() {
  const navigate = useNavigate();
  const [profiles, setProfiles] = useState<Profile[] | null>(null);
  const [clusters, setClusters] = useState<Cluster[] | null>(null);
  const [busy, setBusy] = useState(false);

  const [viewing, setViewing] = useState<Profile | null>(null);
  const [viewingClips, setViewingClips] = useState<ProfileClip[] | null>(null);

  const [promote, setPromote] = useState<Cluster | null>(null);
  const [promoteName, setPromoteName] = useState("");
  const [merging, setMerging] = useState(false);
  const [keepId, setKeepId] = useState("");
  const [absorbId, setAbsorbId] = useState("");

  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const [info, setInfo] = useState<Record<string, { segments: number; seconds: number; clips: number }>>({});
  const [peaks, setPeaks] = useState<Record<string, number[]>>({});
  const audioRef = useRef<HTMLAudioElement | null>(null);

  async function load() {
    const [s, c] = await Promise.all([api.listSpeakers(), api.listClusters()]);
    setProfiles(s.items);
    setClusters(c.items);
  }
  useEffect(() => { load().catch((e) => toast.error(String(e))); }, []);

  async function viewClips(p: Profile) {
    setViewing(p);
    setViewingClips(null);
    try {
      const res = await api.speakerClips(p.id);
      setViewingClips(res.items);
    } catch (e) {
      toast.error("Could not load clips", { description: String((e as Error).message) });
      setViewing(null);
    }
  }

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
      // decoded off the same blob we are about to play, so the bars can't disagree with it
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
      toast.error(String((e as Error).message));
    } finally {
      setLoadingId(null);
    }
  }

  async function doPromote() {
    if (!promote || !promoteName.trim()) return;
    setBusy(true);
    try {
      await api.promoteCluster(promote.id, promoteName.trim());
      toast.success(`Promoted to “${promoteName.trim()}”`);
      setPromote(null);
      setPromoteName("");
      load();
    } catch (e) {
      toast.error("Could not promote that voice", { description: String(e) });
    } finally { setBusy(false); }
  }

  async function doMerge() {
    if (!keepId || !absorbId || keepId === absorbId) return;
    setBusy(true);
    try {
      await api.mergeSpeakers(keepId, absorbId);
      toast.success("Profiles merged");
      setMerging(false);
      load();
    } catch (e) {
      toast.error("Merge failed", { description: String(e) });
    } finally { setBusy(false); }
  }

  const keep = profiles?.find((p) => p.id === keepId);
  const absorb = profiles?.find((p) => p.id === absorbId);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 26 }}>
      <header style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        <h2 style={{ margin: 0, fontSize: 30 }}>Speakers</h2>
        <p style={{ margin: 0, maxWidth: "62ch", fontSize: 14, color: muted(68) }}>
          People the system can name, and recurring voices that could become people.
          Promoting a voice re-labels every clip it appears in.
        </p>
      </header>

      <section style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between" }}>
          <h4 style={{ margin: 0 }}>Enrolled · {profiles?.length ?? 0}</h4>
          <button
            className="btn btn-secondary"
            style={{ fontSize: 12, padding: "5px 10px" }}
            disabled={(profiles?.length ?? 0) < 2}
            onClick={() => {
              setKeepId(profiles?.[0]?.id || "");
              setAbsorbId(profiles?.[1]?.id || "");
              setMerging(true);
            }}
          >
            Merge profiles
          </button>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: 16 }}>
          {(profiles || []).map((p) => {
            const tone = statusTone(p.status);
            const low = p.intra_cohesion != null && p.intra_cohesion < 0.55;
            return (
              <article key={p.id} className="blueprint row-hover" onClick={() => viewClips(p)}
                       style={{ padding: 15, display: "flex", flexDirection: "column", gap: 11, cursor: "pointer" }}>
                <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 10 }}>
                  <span style={{ fontFamily: "var(--font-heading)", fontSize: 20, lineHeight: 1.1 }}>
                    {p.display_name}
                  </span>
                  <span className="tag mono" style={{ border: `1px solid ${tone.border}`, color: tone.color }}>
                    {p.status.toUpperCase()}
                  </span>
                </div>
                <div className="mono" style={{ display: "flex", gap: 16, fontSize: 11.5, color: muted(58) }}>
                  <span>{p.n_enrollments} enrollments</span>
                  <span>{fmtSpeech(p.total_speech_s)}</span>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <div className="mono" style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
                    <span>COHESION</span><span>{num(p.intra_cohesion)}</span>
                  </div>
                  <div className="bar">
                    <span style={{
                      width: `${Math.round((p.intra_cohesion || 0) * 100)}%`,
                      background: cohesionColor(p.intra_cohesion),
                    }} />
                  </div>
                </div>
                <p style={{ margin: 0, fontSize: 12.5, lineHeight: 1.5, color: low ? AMBER : muted(62) }}>
                  {low
                    ? "Cohesion warning: some enrollments disagree with the rest. This profile may have been polluted by another voice — merge or prune before trusting its matches."
                    : p.n_enrollments < 3
                      ? "Few samples. Matches will stay suggestions until the profile firms up."
                      : "Samples agree. Identification is attempted on any clip above the reliability floor."}
                </p>
                <div style={{ display: "flex", gap: 6, marginTop: "auto" }}>
                  <button className="btn btn-secondary" style={{ flex: 1, fontSize: 12, padding: 5 }}
                          onClick={(e) => { e.stopPropagation(); viewClips(p); }}>
                    View clips
                  </button>
                  <button className="btn btn-secondary" style={{ flex: 1, fontSize: 12, padding: 5 }}
                          onClick={(e) => { e.stopPropagation(); setKeepId(p.id); setAbsorbId(""); setMerging(true); }}>
                    {low ? "Inspect and merge" : "Merge into another"}
                  </button>
                </div>
              </article>
            );
          })}
          {profiles?.length === 0 && (
            <p style={{ margin: 0, fontSize: 13, color: muted(55) }}>
              No profiles yet — name one of the unclaimed voices below.
            </p>
          )}
        </div>
      </section>

      <section style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
            <h4 style={{ margin: 0 }}>Unclaimed voices · {clusters?.length ?? 0}</h4>
            <span style={{ fontSize: 12, color: muted(55) }}>
              Audition the montage before you name anyone.
            </span>
          </div>
          {/* Clustering happens one segment at a time and can only compare against the
              clusters that existed at that moment, so one person drifts across several.
              This is the pass that puts them back together, with every voice in view. */}
          <button
            className="btn btn-secondary"
            style={{ fontSize: 12, padding: "5px 10px", flex: "none" }}
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
                toast.error("Could not consolidate", { description: String((e as Error).message) });
              } finally { setBusy(false); }
            }}
          >
            Merge duplicate voices
          </button>
        </div>
        <div style={{ display: "flex", flexDirection: "column", borderTop: "1px solid var(--color-divider)" }}>
          {(clusters || []).map((c) => {
            const enough = (c.total_speech_s || 0) >= 10;
            return (
              <div key={c.id} style={{
                display: "grid", gridTemplateColumns: "150px 1fr 220px 170px", gap: 18,
                alignItems: "center", padding: "14px 0", borderBottom: `1px solid ${muted(8)}`,
              }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  <span style={{ fontFamily: "var(--font-heading)", fontSize: 17 }}>{clusterLabel(c)}</span>
                  <span className="mono" style={{ fontSize: 11, color: muted(50) }}>
                    {c.id.slice(0, 8)}
                  </span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <button className="btn btn-secondary btn-icon" style={{ width: 30, height: 30, flex: "none" }}
                          disabled={loadingId === c.id} onClick={() => playMontage(c.id)}
                          aria-label="Play montage">
                    {loadingId === c.id ? "…" : playingId === c.id ? "■" : "▶"}
                  </button>
                  <div title={peaks[c.id] ? "Peaks read from the montage" : "Press play to read the montage's peaks"}
                       style={{ display: "flex", alignItems: "flex-end", gap: 1, height: 26, width: 230, flex: "none" }}>
                    <MontageBars peaks={peaks[c.id]} active={playingId === c.id} />
                  </div>
                  {info[c.id] && (
                    <span className="mono" style={{ fontSize: 11, color: muted(50), whiteSpace: "nowrap" }}>
                      {info[c.id].segments} segs · {info[c.id].seconds.toFixed(0)}s · {info[c.id].clips} clips
                    </span>
                  )}
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                  <div className="mono" style={{
                    display: "flex", justifyContent: "space-between", fontSize: 11, color: muted(58),
                  }}>
                    <span>{c.n_members} segments · {fmtSpeech(c.total_speech_s)}</span>
                    <span>{c.intra_cohesion == null ? "cohesion n/a" : num(c.intra_cohesion)}</span>
                  </div>
                  {/* a single-segment cluster has nothing to agree with itself about, so
                      there is no cohesion to draw — an empty bar would imply zero */}
                  {c.intra_cohesion != null && (
                    <div className="bar" style={{ height: 5 }}>
                      <span style={{
                        width: `${Math.round(c.intra_cohesion * 100)}%`,
                        background: cohesionColor(c.intra_cohesion),
                      }} />
                    </div>
                  )}
                </div>
                <div style={{ display: "flex", justifyContent: "flex-end" }}>
                  <button
                    className="btn btn-secondary"
                    style={{ fontSize: 12, padding: "5px 10px", opacity: enough ? 1 : 0.45 }}
                    disabled={!enough}
                    onClick={() => { setPromote(c); setPromoteName(""); }}
                  >
                    {enough ? "Name this voice" : "Too little speech to name"}
                  </button>
                </div>
              </div>
            );
          })}
          {clusters?.length === 0 && (
            <p style={{ margin: 0, padding: "16px 0", fontSize: 13, color: muted(55) }}>
              No unclaimed voices right now.
            </p>
          )}
        </div>
      </section>

      <Dialog
        open={promote != null}
        onClose={() => setPromote(null)}
        kicker="Speakers"
        title="Name this recurring voice"
        subject={promote ? `${clusterLabel(promote)} · ${promote.n_members} segments · ${fmtSpeech(promote.total_speech_s)}` : ""}
        consequence="Re-labels every past appearance of this voice, and it is recognised in future clips."
        confirm="Create profile"
        width="560px"
        busy={busy}
        confirmDisabled={!promoteName.trim()}
        onConfirm={doPromote}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <p style={{ margin: 0, fontSize: 14, lineHeight: 1.55, color: muted(72) }}>
            This voice has been heard {promote?.n_members} times and never named. Naming it
            writes a profile and re-labels every past appearance.
          </p>
          <div className="blueprint" style={{ padding: 12, display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <button className="btn btn-secondary btn-icon" style={{ width: 30, height: 30, flex: "none" }}
                      onClick={() => promote && playMontage(promote.id)}>
                {playingId === promote?.id ? "■" : "▶"}
              </button>
              <div style={{ display: "flex", alignItems: "flex-end", gap: 1, height: 26, flex: 1 }}>
                <MontageBars peaks={promote ? peaks[promote.id] : undefined}
                             active={playingId === promote?.id} />
              </div>
              <span className="mono" style={{ fontSize: 11, color: muted(55), flex: "none" }}>
                montage · {fmtSpeech(promote?.total_speech_s)}
              </span>
            </div>
            <span style={{ fontSize: 12.5, color: muted(62) }}>
              Listen before you name. Cohesion {num(promote?.intra_cohesion)} — how much the
              segments agree with each other.
            </span>
          </div>
          <div className="field">
            <label>Name</label>
            <input className="input" autoFocus value={promoteName}
                   onChange={(e) => setPromoteName(e.target.value)} placeholder="e.g. Marta Kolar" />
          </div>
        </div>
      </Dialog>

      <Dialog
        open={merging}
        onClose={() => setMerging(false)}
        kicker="Speakers"
        title="Merge two profiles"
        subject={`${profiles?.length ?? 0} profiles`}
        consequence="Cohesion is recomputed immediately; matches change from the next job."
        confirm="Merge profiles"
        width="580px"
        busy={busy}
        confirmDisabled={!keepId || !absorbId || keepId === absorbId}
        onConfirm={doMerge}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <p style={{ margin: 0, fontSize: 14, lineHeight: 1.55, color: muted(72) }}>
            Merging combines enrollments and re-derives cohesion. If the two are not the same
            person, cohesion will drop and matches will get worse — the number after the merge
            tells you which happened.
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 26px 1fr", gap: 12, alignItems: "flex-end" }}>
            <div className="field">
              <label>Keep</label>
              <select className="input" value={keepId} onChange={(e) => setKeepId(e.target.value)}>
                <option value="">—</option>
                {(profiles || []).map((p) => <option key={p.id} value={p.id}>{p.display_name}</option>)}
              </select>
            </div>
            <span style={{ textAlign: "center", paddingBottom: 9, color: muted(45) }}>←</span>
            <div className="field">
              <label>Absorb</label>
              <select className="input" value={absorbId} onChange={(e) => setAbsorbId(e.target.value)}>
                <option value="">—</option>
                {(profiles || []).filter((p) => p.id !== keepId)
                  .map((p) => <option key={p.id} value={p.id}>{p.display_name}</option>)}
              </select>
            </div>
          </div>
          {keep && absorb && (
            <div className="blueprint" style={{ padding: 12, display: "flex", flexDirection: "column", gap: 8 }}>
              <span className="kicker-sm" style={{ color: muted(55) }}>After the merge</span>
              <div className="mono" style={{ display: "flex", justifyContent: "space-between", gap: 12, fontSize: 12 }}>
                <span style={{ color: muted(60) }}>enrollments</span>
                <span>{keep.n_enrollments} → <span style={{ color: GREEN }}>{keep.n_enrollments + absorb.n_enrollments}</span></span>
              </div>
              <div className="mono" style={{ display: "flex", justifyContent: "space-between", gap: 12, fontSize: 12 }}>
                <span style={{ color: muted(60) }}>speech</span>
                <span>{fmtSpeech(keep.total_speech_s)} → {fmtSpeech((keep.total_speech_s || 0) + (absorb.total_speech_s || 0))}</span>
              </div>
              <div className="mono" style={{ display: "flex", justifyContent: "space-between", gap: 12, fontSize: 12 }}>
                <span style={{ color: muted(60) }}>cohesion</span>
                <span>{num(keep.intra_cohesion)} → recomputed from the combined set</span>
              </div>
            </div>
          )}
          <span style={{ fontSize: 12.5, color: muted(60) }}>
            {absorb ? `“${absorb.display_name}” disappears; every clip it named now names “${keep?.display_name}”.` : ""}
          </span>
        </div>
      </Dialog>

      <Dialog
        open={viewing != null}
        onClose={() => { setViewing(null); setViewingClips(null); }}
        kicker="Speakers"
        title={viewing ? `Clips featuring ${viewing.display_name}` : ""}
        subject={viewingClips ? `${viewingClips.length} clip${viewingClips.length === 1 ? "" : "s"}` : "loading…"}
        cancel="Close"
        width="620px"
      >
        {!viewingClips && (
          <p style={{ margin: 0, fontSize: 14, color: muted(60) }}>Loading…</p>
        )}
        {viewingClips?.length === 0 && (
          <p style={{ margin: 0, fontSize: 14, color: muted(60) }}>
            No clips reference this profile yet.
          </p>
        )}
        {viewingClips && viewingClips.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", borderTop: "1px solid var(--color-divider)" }}>
            {viewingClips.map((c) => (
              <div key={c.id} className="row-hover"
                   onClick={() => navigate(`/clips/${c.id}`)}
                   style={{
                     display: "flex", alignItems: "center", justifyContent: "space-between",
                     gap: 14, padding: "11px 4px", borderBottom: `1px solid ${muted(8)}`, cursor: "pointer",
                   }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
                  <span style={{
                    fontSize: 14, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                  }}>
                    {c.filename}
                  </span>
                  <span className="mono" style={{ fontSize: 11, color: muted(50) }}>
                    {new Date(c.created_at).toLocaleString()} · {fmtDur(c.duration_s)}
                    {c.talk_share != null && ` · ${Math.round(c.talk_share * 100)}% talk share`}
                  </span>
                </div>
                <span className="tag mono" style={{
                  fontSize: 10, flex: "none",
                  border: `1px solid ${c.match_result === "confident" ? "var(--color-accent)" : "var(--color-neutral-300)"}`,
                  color: c.match_result === "confident" ? "var(--color-accent-700)" : muted(55),
                }}>
                  {c.match_result || "unknown"}
                </span>
              </div>
            ))}
          </div>
        )}
      </Dialog>
    </div>
  );
}
