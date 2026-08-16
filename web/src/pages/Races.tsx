import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/api/client";
import { Dialog } from "@/components/dialog";
import {
  AMBER_INK, GREEN, MOOD_COLOR, RED, fmtDur, fmtSpeech, muted, tagTone, voiceColor,
} from "@/lib/ui";

type Race = {
  id: string; name: string; circuit: string | null; race_date: string | null;
  session_key: number | null; svg_path: string | null; notes: string | null;
  n_clips?: number; n_complete?: number; avg_sentiment?: number | null;
};
type RaceClip = {
  id: string; filename: string; status: string; duration_s: number | null;
  n_speakers: number | null; language: string | null; sentiment: string | null;
  sentiment_score: number | null; mood: string | null; needs_review: boolean;
  transcript: string | null;
};
type Utterance = {
  id: string; clip_id: string; filename: string; start_s: number; end_s: number; text: string;
  local_label: string | null; display_name: string | null; cluster_id: string | null;
  sentiment: string | null; sentiment_score: number | null;
  text_sentiment: string | null; mood: string | null; match_result: string | null;
};

const TERMINAL = ["COMPLETE", "FAILED", "REJECTED", "DEAD"];

/** Track outline. Fetched rather than <img src> so it inlines into the DOM and inherits
 *  currentColor — an <img> would render it as an opaque bitmap that can't be themed. */
function TrackOutline({ raceId }: { raceId: string }) {
  const [svg, setSvg] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    fetch(api.raceSvgUrl(raceId))
      .then((r) => (r.ok ? r.text() : Promise.reject()))
      .then((t) => alive && setSvg(t))
      .catch(() => {});
    return () => { alive = false; };
  }, [raceId]);
  if (!svg) return null;
  return (
    <div
      style={{ pointerEvents: "none", height: "100%", width: "100%", color: muted(45) }}
      className="track-outline"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}

function RaceForm({ open, race, onClose, onDone }: {
  open: boolean; race: Race | null; onClose: () => void; onDone: (r: Race) => void;
}) {
  const [fields, setFields] = useState({ name: "", circuit: "", race_date: "", session_key: "", notes: "" });
  const [svg, setSvg] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setFields({
      name: race?.name || "", circuit: race?.circuit || "",
      race_date: race?.race_date ? String(race.race_date).slice(0, 10) : "",
      session_key: race?.session_key ? String(race.session_key) : "", notes: race?.notes || "",
    });
    setSvg(null);
  }, [race, open]);

  async function submit() {
    if (!fields.name.trim()) return toast.error("Name is required");
    setBusy(true);
    try {
      onDone(await api.createRace(fields, svg));
    } catch (e) {
      toast.error(String((e as Error).message));
    } finally { setBusy(false); }
  }

  return (
    <Dialog
      open={open}
      onClose={onClose}
      kicker="Races"
      title="New race"
      subject="a race groups its recordings; nothing is reprocessed by filing"
      consequence="Creating a race only makes a grouping. Recordings are added afterwards."
      confirm="Create race"
      width="860px"
      busy={busy}
      onConfirm={submit}
    >
      <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 26, alignItems: "start" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 14, minWidth: 0 }}>
          <div className="field">
            <label>Race name</label>
            <input className="input" autoFocus value={fields.name}
                   placeholder="2026 Dutch Grand Prix"
                   onChange={(e) => setFields({ ...fields, name: e.target.value })} />
          </div>
          <div className="field">
            <label>Circuit</label>
            <input className="input" value={fields.circuit} placeholder="Circuit Zandvoort"
                   onChange={(e) => setFields({ ...fields, circuit: e.target.value })} />
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div className="field">
              <label>Date</label>
              {/* native date input: no picker dependency, correct locale + keyboard free */}
              <input className="input mono" type="date" value={fields.race_date}
                     onChange={(e) => setFields({ ...fields, race_date: e.target.value })} />
            </div>
            <div className="field">
              <label>OpenF1 session key <span style={{ fontWeight: 400, color: muted(50) }}>optional</span></label>
              <input className="input mono" inputMode="numeric" value={fields.session_key}
                     placeholder="9158"
                     onChange={(e) => setFields({ ...fields, session_key: e.target.value })} />
            </div>
          </div>
          <div className="field">
            <label>Notes</label>
            <textarea className="input" rows={2} style={{ resize: "vertical" }} value={fields.notes}
                      onChange={(e) => setFields({ ...fields, notes: e.target.value })} />
          </div>
        </div>
        <div className="blueprint" style={{ padding: 13, display: "flex", flexDirection: "column", gap: 9 }}>
          <span className="kicker-sm" style={{ color: muted(55) }}>Track outline</span>
          <p style={{ margin: 0, fontSize: 12.5, lineHeight: 1.5, color: muted(68) }}>
            An SVG outline is drawn on the race page. Outline only — scripts and external
            references are stripped on upload.
          </p>
          <input className="input" type="file" accept=".svg,image/svg+xml"
                 onChange={(e) => setSvg(e.target.files?.[0] ?? null)} />
          {svg && <span className="mono" style={{ fontSize: 11, color: GREEN }}>✓ {svg.name}</span>}
        </div>
      </div>
    </Dialog>
  );
}

function RaceList() {
  const navigate = useNavigate();
  const [races, setRaces] = useState<Race[] | null>(null);
  const [totalClips, setTotalClips] = useState(0);
  const [creating, setCreating] = useState(false);
  const [deleting, setDeleting] = useState<Race | null>(null);
  const [alsoClips, setAlsoClips] = useState(false);

  const load = useCallback(() => {
    api.listRaces().then((r) => setRaces(r.items)).catch((e) => toast.error(String(e.message)));
    api.listClips({ limit: "1" }).then((r) => setTotalClips(r.total)).catch(() => {});
  }, []);
  useEffect(load, [load]);

  const filed = (races || []).reduce((n, r) => n + (r.n_clips ?? 0), 0);
  const processed = (races || []).reduce((n, r) => n + (r.n_complete ?? 0), 0);
  const tiles = [
    { value: races?.length ?? 0, label: "races" },
    { value: filed, label: "recordings filed under a race" },
    { value: Math.max(0, totalClips - filed), label: "recordings not filed" },
    { value: filed ? `${Math.round((processed / filed) * 100)}%` : "—", label: "of filed audio processed" },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
      <header style={{
        display: "flex", alignItems: "flex-end", justifyContent: "space-between",
        gap: 20, flexWrap: "wrap",
      }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
          <span className="kicker">Grouping</span>
          <h2 style={{ margin: 0, fontSize: 30 }}>Races</h2>
          <p style={{ margin: 0, maxWidth: "64ch", fontSize: 14, color: muted(68) }}>
            A race groups its recordings. Coverage is the share of speech the system could
            attribute — it is a reading of the audio you have, not a score for the weekend.
          </p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn btn-primary" onClick={() => setCreating(true)}>New race</button>
        </div>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
        {tiles.map((t) => (
          <div key={t.label} className="blueprint"
               style={{ padding: "13px 14px", display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={{ fontFamily: "var(--font-heading)", fontSize: 30, lineHeight: 1 }}>{t.value}</span>
            <span style={{ fontSize: 12, color: muted(62) }}>{t.label}</span>
          </div>
        ))}
      </div>

      <div style={{ display: "flex", flexDirection: "column", borderTop: "1px solid var(--color-divider)" }}>
        {(races || []).map((r) => {
          const clips = r.n_clips ?? 0;
          const complete = r.n_complete ?? 0;
          const pending = clips - complete;
          const tone = pending ? tagTone("warn") : tagTone("accent");
          return (
            <div
              key={r.id}
              className="row-hover"
              onClick={() => navigate(`/races/${r.id}`)}
              style={{
                display: "grid",
                gridTemplateColumns: "52px minmax(190px, 1fr) 200px minmax(200px, 1fr) 170px",
                gap: 20, alignItems: "center", padding: "15px 4px",
                borderBottom: `1px solid ${muted(8)}`, cursor: "pointer",
              }}
            >
              <span className="mono" style={{ fontSize: 12, color: muted(45) }}>
                {r.race_date ? String(r.race_date).slice(5, 10) : "—"}
              </span>
              <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
                <span style={{ fontFamily: "var(--font-heading)", fontSize: 19, lineHeight: 1.1 }}>
                  {r.name}
                </span>
                <span style={{ fontSize: 12.5, color: muted(58) }}>
                  {[r.race_date ? String(r.race_date).slice(0, 10) : null, r.circuit]
                    .filter(Boolean).join(" · ") || "no circuit set"}
                </span>
              </div>
              <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
                {r.session_key && (
                  <span className="tag mono" style={{
                    border: "1px solid var(--color-neutral-300)", color: "var(--color-neutral-700)", fontSize: 10.5,
                  }}>
                    session {r.session_key}
                  </span>
                )}
                {r.avg_sentiment != null && (
                  <span className="tag mono" style={{
                    border: `1px solid ${r.avg_sentiment < -0.2 ? RED : r.avg_sentiment > 0.2 ? GREEN : "var(--color-neutral-300)"}`,
                    color: r.avg_sentiment < -0.2 ? RED : r.avg_sentiment > 0.2 ? GREEN : "var(--color-neutral-700)",
                    fontSize: 10.5,
                  }}>
                    text {r.avg_sentiment > 0 ? "+" : ""}{r.avg_sentiment.toFixed(2)}
                  </span>
                )}
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 5, minWidth: 0 }}>
                <div className="mono" style={{
                  display: "flex", justifyContent: "space-between", fontSize: 11, color: muted(55),
                }}>
                  <span>{clips} recording{clips === 1 ? "" : "s"}</span>
                  <span>{clips ? `${Math.round((complete / clips) * 100)}% processed` : "empty"}</span>
                </div>
                <div style={{ display: "flex", height: 6, background: muted(10) }}>
                  <div style={{ width: clips ? `${(complete / clips) * 100}%` : "0%", background: "var(--sig-voice-a)" }} />
                  <div className="hatch" style={{ width: clips ? `${(pending / clips) * 100}%` : "0%" }} />
                </div>
              </div>
              <div style={{ display: "flex", justifyContent: "flex-end", alignItems: "center", gap: 6 }}>
                <span className="tag mono" style={{ border: `1px solid ${tone.border}`, color: tone.color }}>
                  {pending ? `${pending} pending` : "clean"}
                </span>
                <button className="btn btn-ghost" style={{ fontSize: 12, padding: "4px 7px", color: RED }}
                        onClick={(e) => { e.stopPropagation(); setDeleting(r); }}>
                  Delete
                </button>
              </div>
            </div>
          );
        })}
        {races?.length === 0 && (
          <p style={{ padding: "24px 0", fontSize: 13, color: muted(55) }}>
            No races yet — create one, then drop its recordings in.
          </p>
        )}
      </div>

      <div style={{ display: "flex", gap: 18, flexWrap: "wrap", fontSize: 11.5, color: muted(58) }}>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 10, height: 10, background: "var(--sig-voice-a)" }} />processed
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span className="hatch" style={{ width: 10, height: 10, border: `1px solid ${muted(30)}` }} />
          not processed yet
        </span>
      </div>

      <RaceForm open={creating} race={null} onClose={() => setCreating(false)}
                onDone={(r) => { setCreating(false); load(); navigate(`/races/${r.id}`); }} />

      <Dialog
        open={deleting != null}
        onClose={() => { setDeleting(null); setAlsoClips(false); }}
        kicker="Races · destructive"
        title="Delete this race?"
        subject={deleting ? `${deleting.name} · ${deleting.n_clips ?? 0} recordings` : ""}
        consequence={alsoClips
          ? "Not reversible. The audio is removed from disk and every transcript, voice and tone reading with it."
          : "The grouping goes; the recordings stay in the library, unfiled."}
        confirm={alsoClips
          ? `Delete race and ${deleting?.n_clips ?? 0} recording${deleting?.n_clips === 1 ? "" : "s"}`
          : "Delete race only"}
        cancel="Keep race"
        danger
        width="560px"
        onConfirm={async () => {
          if (!deleting) return;
          try {
            const res = await api.deleteRace(deleting.id, alsoClips);
            toast.success(res.clips_deleted
              ? `Race and ${res.clips_deleted} recording${res.clips_deleted === 1 ? "" : "s"} deleted`
              : "Race deleted — its recordings stay in the library");
            setDeleting(null);
            setAlsoClips(false);
            load();
          } catch (e) { toast.error(String((e as Error).message)); }
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <p style={{ margin: 0, fontSize: 14.5, lineHeight: 1.6 }}>
            A race is a grouping. Deleting it unmakes the grouping only: every transcript,
            voice and tone reading survives, and the {deleting?.n_clips ?? 0} recordings
            reappear in the library with no race attached.
          </p>
          {/* The destructive reading of "delete this race" has to be asked for by name.
              Defaulting to it would make an ordinary tidy-up destroy the audio. */}
          <label style={{
            display: "flex", alignItems: "flex-start", gap: 9, fontSize: 13.5,
            cursor: "pointer", padding: 12,
            border: `1px solid ${alsoClips ? RED : "var(--color-divider)"}`,
          }}>
            <input type="checkbox" checked={alsoClips} style={{ marginTop: 2 }}
                   onChange={(e) => setAlsoClips(e.target.checked)} />
            <span>
              Delete the {deleting?.n_clips ?? 0} recording{deleting?.n_clips === 1 ? "" : "s"} too
              <span style={{ display: "block", color: muted(60), fontSize: 12.5, marginTop: 3 }}>
                Audio off disk, transcripts and tone readings gone, and any profile enrolled
                from them loses that enrollment. A deletion receipt is written for each.
                Not reversible.
              </span>
            </span>
          </label>
          {alsoClips && (deleting?.n_clips ?? 0) > 0 && (
            <span className="mono" style={{ fontSize: 11.5, color: RED }}>
              {deleting?.n_clips} clips will be removed from the corpus, and any Ask answer
              citing them will show its citation as deleted.
            </span>
          )}
        </div>
      </Dialog>
    </div>
  );
}

function RaceDetail({ raceId }: { raceId: string }) {
  const navigate = useNavigate();
  const [race, setRace] = useState<(Race & { clips: RaceClip[]; speakers: any[] }) | null>(null);
  const [stats, setStats] = useState<any>(null);
  const [utterances, setUtterances] = useState<Utterance[]>([]);
  const [uploading, setUploading] = useState(0);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    const r = await api.getRace(raceId);
    setRace(r);
    api.raceStats(raceId).then(setStats).catch(() => {});
    if (r.clips.some((c: RaceClip) => c.status === "COMPLETE")) {
      setUtterances((await api.raceUtterances(raceId)).items);
    }
    return r;
  }, [raceId]);

  useEffect(() => { load().catch((e) => toast.error(String(e.message))); }, [load]);

  // poll only while something is still in flight — a race whose clips are all terminal is a
  // static page, and polling it forever is pure waste
  const pending = race?.clips.filter((c) => !TERMINAL.includes(c.status)).length ?? 0;
  useEffect(() => {
    if (!pending) return;
    const t = setInterval(() => load().catch(() => {}), 3000);
    return () => clearInterval(t);
  }, [pending, load]);

  async function uploadFiles(files: File[]) {
    const audio = files.filter((f) =>
      f.type.startsWith("audio/") || /\.(mp3|wav|m4a|ogg|flac|webm|aac)$/i.test(f.name));
    if (!audio.length) return toast.error("No audio files in that drop");
    setUploading(audio.length);
    let ok = 0, failed = 0;
    const queue = [...audio];
    // bounded concurrency, same reason as the batch uploader
    await Promise.all(Array.from({ length: 6 }, async () => {
      while (queue.length) {
        const f = queue.shift()!;
        try { await api.uploadRaceClip(raceId, f); ok++; } catch { failed++; }
        setUploading((n) => n - 1);
      }
    }));
    toast[failed ? "warning" : "success"](`${ok} uploaded${failed ? `, ${failed} failed` : ""}`);
    load().catch(() => {});
  }

  /** Attribution across the weekend, from the utterances themselves: a name the system
   *  stands behind, a voice it grouped but cannot name, and speech it never attributed. */
  const attribution = useMemo(() => {
    if (!utterances.length) return null;
    let named = 0, grouped = 0, unattributed = 0;
    for (const u of utterances) {
      if (u.display_name) named++;
      else if (u.cluster_id) grouped++;
      else unattributed++;
    }
    const total = utterances.length;
    return { named: named / total, grouped: grouped / total, unattributed: unattributed / total, total };
  }, [utterances]);

  /** Moments worth hearing: the two readings disagreeing, or nobody attributed at all. */
  const moments = useMemo(() =>
    utterances
      .filter((u) => u.text?.trim() && (
        (u.text_sentiment && u.sentiment && u.text_sentiment !== u.sentiment)
        || (!u.display_name && !u.cluster_id)
        || u.mood === "stressed"))
      .slice(0, 8),
    [utterances]);

  if (!race) return <p style={{ color: muted(55) }}>Loading the race…</p>;

  const speech = stats?.totals?.speech_s ?? 0;
  const complete = race.clips.filter((c) => c.status === "COMPLETE").length;
  const flagged = race.clips.filter((c) => c.needs_review).length;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
      <header style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 16, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 300, display: "flex", flexDirection: "column", gap: 4 }}>
            <span className="kicker">
              <a href="/races" onClick={(e) => { e.preventDefault(); navigate("/races"); }}>Races</a>
              {race.circuit ? ` / ${race.circuit}` : ""}
            </span>
            <h2 style={{ margin: 0, fontSize: 30 }}>{race.name}</h2>
            <div className="mono" style={{
              display: "flex", alignItems: "center", gap: 14, fontSize: 12,
              color: muted(62), flexWrap: "wrap",
            }}>
              {race.race_date && <span>{String(race.race_date).slice(0, 10)}</span>}
              <span>{race.clips.length} recordings</span>
              <span>{complete} processed</span>
              <span>{fmtSpeech(speech)} of speech</span>
              <span>{utterances.length} utterances</span>
            </div>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button className="btn btn-secondary" onClick={() => inputRef.current?.click()}>
              Add recordings
            </button>
            <button className="btn btn-primary" onClick={() => navigate("/ask")}>Ask about the corpus</button>
          </div>
        </div>
        <input ref={inputRef} type="file" multiple accept="audio/*,.mp3,.wav,.m4a,.ogg,.flac"
               style={{ display: "none" }}
               onChange={(e) => { uploadFiles(Array.from(e.target.files ?? [])); e.target.value = ""; }} />
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: 24, alignItems: "start" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 22, minWidth: 0 }}>
          <div
            className="blueprint hatch"
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); uploadFiles(Array.from(e.dataTransfer.files)); }}
            style={{
              padding: 20, borderStyle: "dashed", display: "flex", flexDirection: "column",
              alignItems: "center", gap: 6,
              borderColor: dragging ? "var(--color-accent)" : undefined,
            }}
          >
            <span style={{ fontFamily: "var(--font-heading)", fontSize: 18 }}>
              {uploading > 0 ? `Uploading ${uploading} file${uploading === 1 ? "" : "s"}…` : "Drop recordings for this race"}
            </span>
            <span style={{ fontSize: 12.5, color: muted(62) }}>
              Each file is transcribed, diarized, matched against enrolled profiles and scored.
            </span>
          </div>

          {stats?.sentiment_timeline?.length > 1 && (
            <section style={{ display: "flex", flexDirection: "column", gap: 11 }}>
              <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
                <h4 style={{ margin: 0 }}>Traffic across the weekend</h4>
                <span style={{ fontSize: 12, color: muted(55) }}>
                  One bar per recording, in upload order · colour is the tone reading
                </span>
              </div>
              <div className="blueprint" style={{ padding: "14px 16px", display: "flex", flexDirection: "column", gap: 10 }}>
                <div style={{ display: "flex", alignItems: "flex-end", gap: 2, height: 96 }}>
                  {stats.sentiment_timeline.map((t: any) => {
                    const clip = race.clips.find((c) => c.id === t.clip_id);
                    const dur = clip?.duration_s || 0;
                    const maxDur = Math.max(...race.clips.map((c) => c.duration_s || 0), 1);
                    return (
                      <div
                        key={t.clip_id}
                        title={`${t.filename} · ${clip?.mood || "unscored"}`}
                        onClick={() => navigate(`/clips/${t.clip_id}`)}
                        className="wf-seg"
                        style={{
                          flex: 1, height: `${Math.max(8, (dur / maxDur) * 90)}px`,
                          background: MOOD_COLOR[clip?.mood || ""] || muted(25),
                          cursor: "pointer",
                        }}
                      />
                    );
                  })}
                </div>
                <div className="mono" style={{
                  display: "flex", justifyContent: "space-between", fontSize: 10.5, color: muted(45),
                }}>
                  <span>first recording</span><span>last</span>
                </div>
              </div>
            </section>
          )}

          <section style={{ display: "flex", flexDirection: "column", gap: 11 }}>
            <h4 style={{ margin: 0 }}>Recordings</h4>
            <table className="table">
              <thead>
                <tr><th>File</th><th>Duration</th><th>Voices</th><th>Tone</th><th>Status</th><th /></tr>
              </thead>
              <tbody>
                {race.clips.map((c) => {
                  const tone = c.status === "COMPLETE" ? tagTone("accent")
                    : TERMINAL.includes(c.status) ? tagTone("warn") : tagTone("neutral");
                  return (
                    <tr key={c.id}>
                      <td className="mono" style={{ fontSize: 12.5 }}>
                        <a href={`/clips/${c.id}`}
                           onClick={(e) => { e.preventDefault(); navigate(`/clips/${c.id}`); }}>
                          {c.filename}
                        </a>
                      </td>
                      <td className="mono" style={{ fontSize: 12.5 }}>{fmtDur(c.duration_s)}</td>
                      <td className="mono" style={{ fontSize: 12.5 }}>{c.n_speakers ?? "—"}</td>
                      <td className="mono" style={{ fontSize: 12.5, color: MOOD_COLOR[c.mood || ""] || muted(55) }}>
                        {c.mood || "—"}
                      </td>
                      <td>
                        <span className="tag mono" style={{ border: `1px solid ${tone.border}`, color: tone.color }}>
                          {c.status}
                        </span>
                      </td>
                      <td style={{ textAlign: "right" }}>
                        <button className="btn btn-ghost" style={{ fontSize: 12, padding: "4px 7px" }}
                                onClick={() => navigate(`/clips/${c.id}`)}>Open</button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {race.clips.length === 0 && (
              <p style={{ margin: 0, fontSize: 13, color: muted(55) }}>
                No recordings filed here yet — drop some audio above.
              </p>
            )}
          </section>

          {moments.length > 0 && (
            <section style={{ display: "flex", flexDirection: "column", gap: 11 }}>
              <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
                <h4 style={{ margin: 0 }}>Moments worth hearing</h4>
                <span style={{ fontSize: 12, color: muted(55) }}>
                  Surfaced because the two readings disagree, the tone reads stressed, or nobody was attributed
                </span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", borderTop: "1px solid var(--color-divider)" }}>
                {moments.map((m) => {
                  const who = m.display_name || (m.cluster_id ? "Grouped voice" : "Unattributed");
                  return (
                    <div key={m.id} style={{
                      display: "grid", gridTemplateColumns: "96px 1fr 190px 90px", gap: 16,
                      alignItems: "center", padding: "13px 2px", borderBottom: `1px solid ${muted(8)}`,
                    }}>
                      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                        <span className="mono" style={{ fontSize: 12, color: "var(--color-accent-700)" }}>
                          {fmtDur(m.start_s)}
                        </span>
                        <span className="mono" style={{
                          fontSize: 10.5, color: muted(45), overflow: "hidden", textOverflow: "ellipsis",
                        }}>
                          {m.filename}
                        </span>
                      </div>
                      <div style={{ display: "flex", flexDirection: "column", gap: 3, minWidth: 0 }}>
                        <span style={{
                          fontFamily: "var(--font-heading)", fontSize: 14.5,
                          color: m.display_name ? "var(--color-text)" : muted(65),
                        }}>
                          {who}
                        </span>
                        <span style={{ fontSize: 13.5, lineHeight: 1.5 }}>{m.text}</span>
                      </div>
                      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                        <span className="mono" style={{
                          fontSize: 11, color: MOOD_COLOR[m.mood || ""] || muted(55),
                        }}>
                          {m.mood ? `voice ${m.mood}` : "not judged"}
                          {m.sentiment_score != null && ` · text ${m.sentiment_score > 0 ? "+" : ""}${m.sentiment_score.toFixed(2)}`}
                        </span>
                        <span style={{ fontSize: 11.5, color: muted(55) }}>
                          {m.text_sentiment && m.sentiment && m.text_sentiment !== m.sentiment
                            ? "text and voice disagree"
                            : !m.display_name && !m.cluster_id ? "no voice attributed" : "tone reads stressed"}
                        </span>
                      </div>
                      <div style={{ display: "flex", justifyContent: "flex-end" }}>
                        <button className="btn btn-secondary" style={{ fontSize: 12, padding: "4px 9px" }}
                                onClick={() => navigate(`/clips/${m.clip_id}?t=${m.start_s}`)}>
                          ▶ Listen
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>
          )}
        </div>

        <aside style={{ display: "flex", flexDirection: "column", gap: 16, position: "sticky", top: 26 }}>
          {race.svg_path && (
            <div className="blueprint" style={{ padding: 13, height: 160 }}>
              <TrackOutline raceId={race.id} />
            </div>
          )}

          {attribution && (
            <div className="blueprint" style={{ padding: 13, display: "flex", flexDirection: "column", gap: 10 }}>
              <span className="kicker-sm" style={{ color: muted(55) }}>Attribution across the weekend</span>
              <div style={{ display: "flex", height: 10 }}>
                <div style={{ width: `${attribution.named * 100}%`, background: "var(--sig-voice-a)" }} />
                <div style={{ width: `${attribution.grouped * 100}%`, background: "var(--sig-voice-b)" }} />
                <div className="hatch" style={{
                  width: `${attribution.unattributed * 100}%`,
                  border: attribution.unattributed ? `1px solid ${muted(28)}` : undefined,
                }} />
              </div>
              <div className="mono" style={{ display: "flex", flexDirection: "column", gap: 5, fontSize: 12 }}>
                <span style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>named</span><span>{Math.round(attribution.named * 100)}%</span>
                </span>
                <span style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>grouped, unnamed</span><span>{Math.round(attribution.grouped * 100)}%</span>
                </span>
                <span style={{ display: "flex", justifyContent: "space-between" }}>
                  <span>unattributed</span><span>{Math.round(attribution.unattributed * 100)}%</span>
                </span>
              </div>
            </div>
          )}

          {stats?.voices?.length > 0 && (
            <div className="blueprint" style={{ padding: 13, display: "flex", flexDirection: "column", gap: 9 }}>
              <span className="kicker-sm" style={{ color: muted(55) }}>Voices heard here</span>
              {stats.voices.slice(0, 8).map((v: any) => {
                const name = v.display_name || `Voice ${String(v.key).slice(0, 4)}`;
                return (
                  <div key={v.key} style={{
                    display: "grid", gridTemplateColumns: "1fr 58px", gap: 8, alignItems: "center",
                  }}>
                    <span style={{ display: "flex", alignItems: "center", gap: 7, minWidth: 0 }}>
                      <span style={{ width: 9, height: 9, flex: "none", background: voiceColor(name) }} />
                      <span style={{
                        fontSize: 13, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                        color: v.display_name ? "var(--color-text)" : muted(70),
                      }}>
                        {name}
                      </span>
                    </span>
                    <span className="mono" style={{ fontSize: 11, textAlign: "right", color: muted(55) }}>
                      {Math.round((v.talk_share || 0) * 100)}%
                    </span>
                  </div>
                );
              })}
              {stats.voices.length > 8 && (
                <span className="mono" style={{ fontSize: 11, color: muted(45) }}>
                  + {stats.voices.length - 8} quieter voices
                </span>
              )}
              <a href="/speakers" style={{ fontSize: 12.5, marginTop: 2 }}
                 onClick={(e) => { e.preventDefault(); navigate("/speakers"); }}>
                Open in Speakers
              </a>
            </div>
          )}

          {flagged > 0 && (
            <div className="blueprint hatch" style={{ padding: 13, display: "flex", flexDirection: "column", gap: 6 }}>
              <span className="kicker-sm" style={{ color: AMBER_INK }}>Needs a curator</span>
              <span style={{ fontSize: 13, lineHeight: 1.5, color: muted(75) }}>
                {flagged} recording{flagged === 1 ? "" : "s"} from this race {flagged === 1 ? "is" : "are"} in
                the review queue — the pipeline hesitated or graded the audio poor.
              </span>
              <a href="/review" style={{ fontSize: 12.5 }}
                 onClick={(e) => { e.preventDefault(); navigate("/review"); }}>
                Open the queue
              </a>
            </div>
          )}

          {stats?.mood_counts && (
            <div className="blueprint" style={{ padding: 13, display: "flex", flexDirection: "column", gap: 9 }}>
              <span className="kicker-sm" style={{ color: muted(55) }}>Tone across the race</span>
              <div className="mono" style={{ display: "flex", flexDirection: "column", gap: 5, fontSize: 12 }}>
                {(["calm", "tired", "stressed"] as const).map((m) => (
                  <span key={m} style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ width: 9, height: 9, background: MOOD_COLOR[m] }} />{m}
                    </span>
                    <span>{stats.mood_counts[m] ?? 0}</span>
                  </span>
                ))}
              </div>
              <span style={{ fontSize: 11, lineHeight: 1.5, color: muted(50) }}>
                Tone labels are heuristic readings of the audio, not measurements of how anyone felt.
              </span>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

export default function Races() {
  const { id } = useParams();
  return id ? <RaceDetail raceId={id} /> : <RaceList />;
}
