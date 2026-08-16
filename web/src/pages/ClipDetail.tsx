import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/api/client";
import { AudioPlayer } from "@/components/audio-player";
import { Dialog } from "@/components/dialog";
import { EditClipDialog, DeleteClipDialog, type ClipRef } from "@/components/clip-dialogs";
import {
  AMBER, AMBER_INK, MOOD_COLOR, MOOD_PCT, OUTCOME, QUALITY_COLOR, RED,
  fmtTime, muted, num, outcomeOf, voiceColor, type Outcome,
} from "@/lib/ui";

const API_KEY = () => localStorage.getItem("api_key") || "change-me";

type Speaker = {
  local_label: string; display_name: string | null; profile_id: string | null;
  match_result: string; match_score: number | null; match_margin: number | null;
  runner_up_score: number | null; reliability: number | null; talk_share: number | null;
  speech_s: number; n_turns: number; cluster_id: string | null; is_manual: boolean;
};

/** One card per voice, one of four outcomes. The four say genuinely different things —
 *  a name, a proposal, "judged and nobody matched", and "refused to judge" — so they get
 *  different cards rather than one card with a confidence number that hides the difference. */
function VoiceCard({ s, thresholds, onAct }: {
  s: Speaker;
  thresholds: { id_threshold: number; reliability_fair: number };
  onAct: (s: Speaker, mode: "correct" | "confirm") => void;
}) {
  const outcome = outcomeOf(s.match_result);
  const meta = OUTCOME[outcome];
  const name = s.display_name || s.local_label.replace("SPEAKER_", "Voice ");
  const abstained = outcome === "abstained";

  const bar = (label: string, value: number | null, color: string, markAt?: number) => (
    <>
      <div className="mono" style={{ display: "flex", justifyContent: "space-between", fontSize: 11 }}>
        <span>{label}</span><span>{num(value)}</span>
      </div>
      <div className="bar" style={{ position: "relative" }}>
        <span style={{ width: `${Math.round((value || 0) * 100)}%`, background: color }} />
        {markAt != null && (
          <div style={{
            position: "absolute", left: `${Math.round(markAt * 100)}%`, top: -3, bottom: -3,
            width: 1, background: "var(--color-text)",
          }} />
        )}
      </div>
    </>
  );

  return (
    <article
      className={`blueprint${abstained ? " hatch" : ""}`}
      style={{
        padding: 14, display: "flex", flexDirection: "column", gap: 10,
        borderLeft: abstained ? `3px dashed ${muted(40)}` : `3px solid ${meta.color}`,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span className="kicker-sm" style={{ color: meta.kicker }}>{meta.label}</span>
        {abstained
          ? <span className="hatch" style={{ width: 11, height: 11, border: `1px solid ${muted(40)}` }} />
          : <span style={{ width: 11, height: 11, background: voiceColor(s.local_label) }} />}
      </div>

      <div style={{
        fontFamily: "var(--font-heading)", fontSize: 20, lineHeight: 1.1,
        fontStyle: meta.italic ? "italic" : "normal",
        color: outcome === "confident" ? "var(--color-text)" : muted(outcome === "abstained" ? 55 : 70),
      }}>
        {abstained ? "Declined to judge" : `${name}${outcome === "suggested" ? "?" : ""}`}
      </div>

      <div style={{ fontSize: 12, color: muted(outcome === "abstained" ? 70 : 62) }}>
        {outcome === "confident" && (
          <>Auto-labelled{s.is_manual ? " — corrected by hand" : ""}.
            {s.runner_up_score != null && s.match_score != null &&
              ` Runner-up profile ${num(s.match_score - s.runner_up_score)} behind.`}</>
        )}
        {outcome === "suggested" && (
          <>Closest profile, but {num((thresholds.id_threshold) - (s.match_score || 0))} under the
            auto-label bar. Proposal only — nothing has been written.</>
        )}
        {/* A zero best-match means the comparison had nothing to compare against — saying
            "best was 0.00" would read as a bad match rather than an empty directory. */}
        {outcome === "unknown" && (s.match_score
          ? <>Audio was good enough to judge. No enrolled profile came close — best was {num(s.match_score)}.</>
          : <>Audio was good enough to judge, but no profile is enrolled to compare it against.</>
        )}
        {abstained && (
          <>Reliability {num(s.reliability)} — under the {num(thresholds.reliability_fair)} floor,
            so identification was never attempted.</>
        )}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 7, marginTop: "auto" }}>
        {!abstained && bar(
          outcome === "unknown" ? "BEST MATCH" : "MATCH",
          s.match_score, meta.color,
          outcome === "suggested" ? thresholds.id_threshold : undefined)}
        {outcome === "suggested" && (
          <div className="mono" style={{ fontSize: 10, color: muted(45) }}>
            │ bar at {num(thresholds.id_threshold)}
          </div>
        )}
        {(outcome !== "suggested") && bar("RELIABILITY", s.reliability,
          abstained ? AMBER : muted(45),
          abstained ? thresholds.reliability_fair : undefined)}
        {abstained && (
          <div className="mono" style={{ fontSize: 10, color: muted(45) }}>no match score exists</div>
        )}
      </div>

      <div style={{ display: "flex", gap: 6 }}>
        {outcome === "suggested" ? (
          <>
            <button className="btn btn-primary" style={{ flex: 1, fontSize: 12, padding: 5 }}
                    onClick={() => onAct(s, "confirm")}>Confirm</button>
            <button className="btn btn-secondary" style={{ flex: 1, fontSize: 12, padding: 5 }}
                    onClick={() => onAct(s, "correct")}>Reject</button>
          </>
        ) : (
          <button className="btn btn-secondary" style={{ flex: 1, fontSize: 12, padding: 5 }}
                  onClick={() => onAct(s, "correct")}>
            {outcome === "confident" ? "Correct" : outcome === "unknown" ? "Name this voice" : "Label manually"}
          </button>
        )}
      </div>
    </article>
  );
}

export default function ClipDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [result, setResult] = useState<any>(null);
  const [profiles, setProfiles] = useState<any[]>([]);
  const [thresholds, setThresholds] = useState({ id_threshold: 0.55, reliability_fair: 0.35 });
  const [variant, setVariant] = useState<"original" | "work">("work");
  const [currentTime, setCurrentTime] = useState(0);
  const [reprocessing, setReprocessing] = useState(false);
  const [assign, setAssign] = useState<Speaker | null>(null);
  const [pick, setPick] = useState<string>("");   // profile id, "" = none chosen
  const [newName, setNewName] = useState("");
  const [enroll, setEnroll] = useState(true);
  const [busy, setBusy] = useState(false);
  const [edit, setEdit] = useState<ClipRef | null>(null);
  const [del, setDel] = useState<ClipRef | null>(null);
  // A nonce so clicking the same word twice still re-fires the player's seek effect.
  const [seekTo, setSeekTo] = useState<{ time: number; nonce: number } | null>(null);
  const seek = (time: number) => setSeekTo((p) => ({ time, nonce: (p?.nonce ?? 0) + 1 }));
  const activeRowRef = useRef<HTMLDivElement>(null);
  const seekedRef = useRef(false);

  async function load() {
    if (id) setResult(await api.getResult(id));
  }
  useEffect(() => { load().catch((e) => toast.error(String(e))); }, [id]); // eslint-disable-line

  useEffect(() => {
    api.listSpeakers().then((r) => setProfiles(r.items)).catch(() => {});
    // The card bars mark the thresholds the pipeline actually ran with, read from settings
    // rather than hard-coded — they are tunable, and a stale mark would be a lie.
    api.getSettings().then((s: any) => {
      const all: any[] = s.categories.flatMap((c: any) => c.fields);
      const get = (k: string, d: number) => Number(all.find((f) => f.key === k)?.value ?? d);
      setThresholds({
        id_threshold: get("id_threshold", 0.55),
        reliability_fair: get("reliability_fair", 0.35),
      });
    }).catch(() => {});
  }, []);

  // ?t=<seconds> deep link: an Ask citation points at a moment, not a file.
  useEffect(() => {
    const t = Number(searchParams.get("t"));
    if (!result || seekedRef.current || !Number.isFinite(t) || t <= 0) return;
    seekedRef.current = true;
    seek(t);
  }, [result, searchParams]);

  const wordsByUtterance = useMemo(() => {
    const m = new Map<string, any[]>();
    for (const w of result?.words || []) {
      if (!m.has(w.utterance_id)) m.set(w.utterance_id, []);
      m.get(w.utterance_id)!.push(w);
    }
    return m;
  }, [result]);

  const activeUtteranceId = useMemo(() => {
    for (const u of result?.utterances || []) {
      if (currentTime >= u.start_s && currentTime < u.end_s) return u.id;
    }
    return null;
  }, [result, currentTime]);

  useEffect(() => {
    activeRowRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [activeUtteranceId]);

  const speakers: Speaker[] = result?.speakers || [];

  // Attribution coverage: talk share grouped by what the pipeline concluded, never merged
  // into a single "accuracy" number — "not judged" is its own answer.
  const coverage = useMemo(() => {
    const acc: Record<Outcome, number> = { confident: 0, suggested: 0, unknown: 0, abstained: 0 };
    let total = 0;
    for (const s of speakers) {
      const share = s.talk_share ?? (s.speech_s || 0);
      acc[outcomeOf(s.match_result)] += share;
      total += share;
    }
    if (!total) return null;
    return {
      named: acc.confident / total, suggested: acc.suggested / total,
      unknown: acc.unknown / total, unjudged: acc.abstained / total,
    };
  }, [speakers]);

  const tone = useMemo(() => {
    const utts: any[] = result?.utterances || [];
    const scored = utts.filter((u) => u.sentiment_score != null);
    const disagree = utts.filter((u) => u.text_sentiment && u.sentiment && u.text_sentiment !== u.sentiment);
    const moods = utts.filter((u) => u.mood);
    const counts: Record<string, number> = {};
    for (const u of moods) counts[u.mood] = (counts[u.mood] || 0) + 1;
    const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    return {
      avg: scored.length ? scored.reduce((n, u) => n + u.sentiment_score, 0) / scored.length : null,
      disagreements: disagree.length,
      mood: top ? top[0] : null,
      moodShare: top && moods.length ? top[1] / moods.length : null,
    };
  }, [result]);

  if (!result) {
    return <p style={{ color: muted(55) }}>Loading the clip…</p>;
  }

  const { clip, quality, turns, utterances, run } = result;
  const warnings: any[] = run?.warnings || [];
  const displayName = (label: string | null) => {
    const s = speakers.find((x) => x.local_label === label);
    if (!s) return label || "Unattributed";
    return s.display_name || (label || "").replace("SPEAKER_", "Voice ");
  };
  const outcomeFor = (label: string | null): Outcome =>
    outcomeOf(speakers.find((x) => x.local_label === label)?.match_result);

  async function reprocess() {
    if (!id) return;
    setReprocessing(true);
    try {
      await api.reprocess(id);
      toast.success("Reprocessing queued");
      setTimeout(() => load().catch(() => {}), 1500);
    } finally { setReprocessing(false); }
  }

  function openAssign(s: Speaker, mode: "correct" | "confirm") {
    if (mode === "confirm" && s.profile_id) {
      saveAssign(s, { profile_id: s.profile_id, enroll: true });
      return;
    }
    setAssign(s);
    setPick(s.profile_id || "");
    setNewName("");
    setEnroll(true);
  }

  async function saveAssign(s: Speaker, body: Record<string, unknown>) {
    if (!id) return;
    setBusy(true);
    try {
      const res = await api.assignSpeaker(id, s.local_label, body);
      if (body.enroll && !res.enrolled) {
        toast.warning("Voice labelled, but not enrolled", {
          description: res.enroll_skip_reason || "This clip's audio wasn't usable for identification.",
        });
      } else {
        toast.success("Correction saved");
      }
      setAssign(null);
      load();
    } catch (e) {
      toast.error("Could not save that correction", { description: String(e) });
    } finally { setBusy(false); }
  }

  function confirmAssign() {
    if (!assign) return;
    if (newName.trim()) return saveAssign(assign, { create_profile: { display_name: newName.trim() }, enroll });
    if (pick === "__unknown__") return saveAssign(assign, { mark_unknown: true });
    if (pick) return saveAssign(assign, { profile_id: pick, enroll });
    toast.error("Pick a voice, type a name, or record an abstention");
  }

  const clipRef: ClipRef = {
    id: clip.id, filename: clip.filename, duration_s: clip.duration_s,
    n_speakers: clip.n_speakers, created_at: clip.created_at,
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
      <header style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <div style={{ display: "flex", alignItems: "flex-start", gap: 16, flexWrap: "wrap" }}>
          <div style={{ flex: 1, minWidth: 320, display: "flex", flexDirection: "column", gap: 4 }}>
            <span className="kicker">
              <a href="/" onClick={(e) => { e.preventDefault(); navigate("/"); }}>Library</a>
              {" / "}Clip {clip.id.slice(0, 6)}
            </span>
            <h2 style={{ margin: 0, fontSize: 30 }}>{clip.filename}</h2>
            <div className="mono" style={{
              display: "flex", alignItems: "center", gap: 14, fontSize: 12,
              color: muted(62), flexWrap: "wrap",
            }}>
              <span>{fmtTime(clip.duration_s)}</span>
              <span>{(clip.language || "—").toUpperCase()}{clip.language_conf != null && ` · ${num(clip.language_conf)}`}</span>
              <span>{speakers.length} voices</span>
              {quality?.snr_db != null && <span>SNR {quality.snr_db.toFixed(1)} dB</span>}
              {quality?.grade && (
                <span style={{ color: QUALITY_COLOR[quality.grade] || undefined }}>
                  QUALITY · {String(quality.grade).toUpperCase()}
                </span>
              )}
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className="tag tag-accent" style={{ border: "1px solid var(--color-accent-300)" }}>
              {clip.status}
            </span>
            <button className="btn btn-secondary" onClick={reprocess} disabled={reprocessing}>
              {reprocessing ? "Queueing…" : "Reprocess"}
            </button>
            <button className="btn btn-secondary" onClick={() => setEdit(clipRef)}>Edit details</button>
            <button className="btn btn-secondary" style={{ color: RED }} onClick={() => setDel(clipRef)}>Delete</button>
            <a className="btn btn-primary blueprint" href={`/v1/clips/${clip.id}/export/srt?key=${API_KEY()}`}>
              Export SRT
              <i className="corner tl" /><i className="corner tr" /><i className="corner bl" /><i className="corner br" />
            </a>
          </div>
        </div>

        {warnings.map((w, i) => (
          <div key={i} style={{
            display: "flex", gap: 12, border: `1px solid ${AMBER}`,
            padding: "10px 12px", alignItems: "flex-start",
          }}>
            <span className="mono" style={{ fontSize: 11, color: AMBER_INK, flex: "none", paddingTop: 2 }}>
              WARN
            </span>
            <div>
              <div style={{ fontFamily: "var(--font-heading)", fontSize: 15 }}>
                {String(w.code || "warning").replaceAll("_", " ")}
              </div>
              <div style={{ fontSize: 13, color: muted(65) }}>
                {Object.entries(w).filter(([k]) => k !== "code").map(([k, v]) => `${k}: ${v}`).join(" · ")
                  || "The pipeline flagged this clip; identification here is lower-confidence."}
              </div>
            </div>
          </div>
        ))}
      </header>

      <AudioPlayer
        key={variant}
        src={`/v1/clips/${clip.id}/audio?variant=${variant}&key=${API_KEY()}`}
        duration={clip.duration_s || 0}
        turns={turns || []}
        speakers={speakers}
        onTimeUpdate={setCurrentTime}
        seekTo={seekTo}
        right={clip.work_path ? (
          <div className="seg">
            <label className="seg-opt">
              <input type="radio" name="audiosrc" checked={variant === "original"}
                     onChange={() => setVariant("original")} />
              Original
            </label>
            <label className="seg-opt">
              <input type="radio" name="audiosrc" checked={variant === "work"}
                     onChange={() => setVariant("work")} />
              Processed
            </label>
          </div>
        ) : null}
      />

      <section style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
          <h4 style={{ margin: 0 }}>Voices in this clip</h4>
          <span style={{ fontSize: 12, color: muted(55) }}>
            Four identification outcomes. Each says something different about what the system knows.
          </span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 16 }}>
          {speakers.map((s) => (
            <VoiceCard key={s.local_label} s={s} thresholds={thresholds} onAct={openAssign} />
          ))}
          {speakers.length === 0 && (
            <p style={{ margin: 0, fontSize: 13, color: muted(55) }}>
              No voices were separated out of this recording.
            </p>
          )}
        </div>
      </section>

      <section style={{ display: "grid", gridTemplateColumns: "1fr 260px", gap: 24, alignItems: "start" }}>
        <div style={{ display: "flex", flexDirection: "column" }}>
          <div style={{
            display: "flex", alignItems: "baseline", justifyContent: "space-between",
            paddingBottom: 10, borderBottom: "1px solid var(--color-divider)",
          }}>
            <h4 style={{ margin: 0 }}>Transcript</h4>
            <span style={{ fontSize: 12, color: muted(55) }}>
              Hover a word for its confidence · click to seek
            </span>
          </div>

          {(utterances || []).map((u: any) => {
            const active = u.id === activeUtteranceId;
            const outcome = outcomeFor(u.local_label);
            const words = wordsByUtterance.get(u.id);
            const abstained = outcome === "abstained";
            return (
              <div
                key={u.id}
                ref={active ? activeRowRef : undefined}
                onClick={() => seek(u.start_s)}
                style={{
                  display: "grid", gridTemplateColumns: "76px 1fr", gap: 16,
                  padding: "14px 12px 14px 0", cursor: "pointer",
                  borderBottom: `1px solid ${muted(8)}`,
                  borderLeft: active ? "2px solid var(--color-accent)" : undefined,
                  background: abstained ? muted(3) : "transparent",
                }}
              >
                <div style={{ display: "flex", flexDirection: "column", gap: 4, alignItems: "flex-start" }}>
                  <span className="mono" style={{
                    fontSize: 12, color: active ? "var(--color-accent)" : "var(--color-accent-700)",
                  }}>
                    {fmtTime(u.start_s)}
                  </span>
                  {abstained
                    ? <span className="hatch" style={{ width: "100%", height: 3, borderTop: `1px solid ${muted(35)}` }} />
                    : <span style={{ width: "100%", height: 3, background: voiceColor(u.local_label || "") }} />}
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                    <span style={{
                      fontFamily: "var(--font-heading)", fontSize: 15,
                      fontStyle: OUTCOME[outcome].italic ? "italic" : "normal",
                      color: outcome === "confident" ? "var(--color-text)" : muted(abstained ? 55 : 70),
                    }}>
                      {abstained ? "Unattributed" : displayName(u.local_label)}
                      {outcome === "suggested" ? "?" : ""}
                    </span>
                    <span className="kicker-sm" style={{
                      letterSpacing: ".1em", color: abstained ? AMBER_INK : muted(50),
                    }}>
                      {outcome}{u.mean_speaker_conf != null && !abstained ? ` ${num(u.mean_speaker_conf)}` : ""}
                      {active ? " · playing" : ""}
                    </span>
                    {u.mood && (
                      <span className="mono" style={{
                        display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: muted(55),
                      }}>
                        <span style={{
                          width: 26, height: 5, background: muted(12),
                          display: "inline-block", position: "relative",
                        }}>
                          <span style={{
                            position: "absolute", left: 0, top: 0, bottom: 0,
                            width: MOOD_PCT[u.mood] || "30%", background: MOOD_COLOR[u.mood] || muted(40),
                          }} />
                        </span>
                        {u.mood}
                      </span>
                    )}
                  </div>
                  <p style={{ margin: 0, fontSize: 15, lineHeight: 1.6, textWrap: "pretty" } as React.CSSProperties}>
                    {words ? words.map((w: any, i: number) => {
                      const spoken = currentTime >= w.end_s;
                      const now = currentTime >= w.start_s && currentTime < w.end_s;
                      const shaky = w.probability != null && w.probability < 0.5;
                      return (
                        <span
                          key={i}
                          title={w.probability != null ? `word confidence ${w.probability.toFixed(2)}` : undefined}
                          onClick={(e) => { e.stopPropagation(); seek(w.start_s); }}
                          style={{
                            background: now
                              ? "color-mix(in srgb, var(--color-accent) 30%, transparent)"
                              : shaky ? "color-mix(in srgb, var(--sig-amber) 14%, transparent)" : undefined,
                            color: spoken && !now ? muted(72) : undefined,
                            textDecoration: shaky ? "underline dotted" : undefined,
                            textDecorationColor: shaky ? AMBER : undefined,
                            textUnderlineOffset: 4,
                          }}
                        >
                          {w.word}{" "}
                        </span>
                      );
                    }) : u.text}
                  </p>
                  {abstained && (
                    <span style={{ fontSize: 12, color: muted(55) }}>
                      The system declined to judge this voice.{" "}
                      <a href="#" onClick={(e) => {
                        e.preventDefault(); e.stopPropagation();
                        const s = speakers.find((x) => x.local_label === u.local_label);
                        if (s) openAssign(s, "correct");
                      }}>Assign manually</a>
                    </span>
                  )}
                </div>
              </div>
            );
          })}
          {(!utterances || utterances.length === 0) && (
            <p style={{ padding: "24px 0", fontSize: 13, color: muted(55) }}>No transcript available.</p>
          )}
        </div>

        <aside style={{ display: "flex", flexDirection: "column", gap: 16, position: "sticky", top: 26 }}>
          {coverage && (
            <div className="blueprint" style={{ padding: 13, display: "flex", flexDirection: "column", gap: 10 }}>
              <span className="kicker-sm" style={{ color: muted(55) }}>Attribution coverage</span>
              <div style={{ display: "flex", height: 10 }}>
                <div style={{ width: `${coverage.named * 100}%`, background: "var(--sig-voice-a)" }} />
                <div style={{ width: `${coverage.suggested * 100}%`, background: "var(--sig-voice-b)" }} />
                <div style={{ width: `${coverage.unknown * 100}%`, background: "var(--color-neutral-600)" }} />
                <div className="hatch" style={{
                  width: `${coverage.unjudged * 100}%`,
                  border: coverage.unjudged ? `1px solid ${muted(30)}` : undefined,
                }} />
              </div>
              <div className="mono" style={{ display: "flex", flexDirection: "column", gap: 5, fontSize: 12 }}>
                {([["named", coverage.named], ["suggested", coverage.suggested],
                   ["unknown", coverage.unknown], ["not judged", coverage.unjudged]] as const).map(([k, v]) => (
                  <span key={k} style={{ display: "flex", justifyContent: "space-between" }}>
                    <span>{k}</span><span>{Math.round(v * 100)}%</span>
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="blueprint" style={{ padding: 13, display: "flex", flexDirection: "column", gap: 8 }}>
            <span className="kicker-sm" style={{ color: muted(55) }}>Sentiment vs tone</span>
            <div style={{ fontSize: 13, color: muted(75) }}>
              {tone.disagreements > 0
                ? `Two readings disagree on ${tone.disagreements} utterance${tone.disagreements === 1 ? "" : "s"}. They are kept apart, not averaged.`
                : "The two readings agree across this clip. They are still stored apart."}
            </div>
            <div className="mono" style={{ display: "flex", flexDirection: "column", gap: 6, fontSize: 12 }}>
              <span style={{ display: "flex", justifyContent: "space-between" }}>
                <span>text</span>
                <span>{tone.avg == null ? "—" : `${tone.avg > 0 ? "+" : ""}${tone.avg.toFixed(2)}`}</span>
              </span>
              <span style={{ display: "flex", justifyContent: "space-between" }}>
                <span>voice</span>
                <span>{tone.mood ? `${tone.mood} ${num(tone.moodShare)}` : "—"}</span>
              </span>
            </div>
          </div>

          <p style={{ margin: 0, fontSize: 11, lineHeight: 1.5, color: muted(50) }}>
            Tone labels are heuristic readings of the audio, not measurements of how anyone felt.
          </p>
        </aside>
      </section>

      <Dialog
        open={assign != null}
        onClose={() => setAssign(null)}
        kicker="Correction"
        title="Who is this voice?"
        subject={assign
          ? `clip ${clip.id.slice(0, 6)} · ${assign.local_label} · ${assign.n_turns} turns · ${assign.speech_s?.toFixed(1)}s`
          : ""}
        consequence="Corrections are attributed to you and kept in the clip's history. Every clip processed after this benefits."
        confirm="Save correction"
        width="900px"
        busy={busy}
        onConfirm={confirmAssign}
      >
        <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: 26, alignItems: "start" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
            <p style={{ margin: 0, fontSize: 14, lineHeight: 1.55, color: muted(72) }}>
              {assign && assign.match_score != null
                ? <>The system scored its best profile at {num(assign.match_score)}, against a
                    {" "}{num(thresholds.id_threshold)} bar.</>
                : <>The system never got a usable voiceprint here.</>}
              {" "}Pick who this is, or say the audio is not good enough to tell — an honest
              abstention is a valid answer and is recorded as one.
            </p>
            <div style={{ display: "flex", flexDirection: "column", borderTop: "1px solid var(--color-divider)" }}>
              {profiles.map((p: any) => (
                <label key={p.id} style={{
                  display: "grid", gridTemplateColumns: "18px 1fr 120px", gap: 12, alignItems: "center",
                  padding: "11px 4px", borderBottom: `1px solid ${muted(8)}`, cursor: "pointer",
                  background: pick === p.id ? "color-mix(in srgb, var(--color-accent) 7%, transparent)" : "transparent",
                }}>
                  <input type="radio" name="assign" checked={pick === p.id}
                         onChange={() => { setPick(p.id); setNewName(""); }} />
                  <div style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
                    <span style={{
                      fontFamily: "var(--font-heading)", fontSize: 15,
                      color: p.status === "locked" ? AMBER_INK : "var(--color-text)",
                    }}>
                      {p.display_name}
                    </span>
                    <span style={{ fontSize: 12, color: muted(60) }}>
                      {p.status} profile · {p.n_enrollments} enrollment{p.n_enrollments === 1 ? "" : "s"}
                      {p.intra_cohesion != null && ` · cohesion ${num(p.intra_cohesion)}`}
                      {assign?.profile_id === p.id ? " · the system's own proposal" : ""}
                    </span>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                    <span className="mono" style={{ fontSize: 11, color: muted(55), textAlign: "right" }}>
                      {assign?.profile_id === p.id ? num(assign?.match_score) : "—"}
                    </span>
                    <div className="bar" style={{ height: 5 }}>
                      <span style={{
                        width: assign?.profile_id === p.id ? `${(assign?.match_score || 0) * 100}%` : "0%",
                        background: "var(--sig-voice-b)",
                      }} />
                    </div>
                  </div>
                </label>
              ))}
              <label style={{
                display: "grid", gridTemplateColumns: "18px 1fr 120px", gap: 12, alignItems: "center",
                padding: "11px 4px", borderBottom: `1px solid ${muted(8)}`, cursor: "pointer",
              }}>
                <input type="radio" name="assign" checked={pick === "__unknown__"}
                       onChange={() => { setPick("__unknown__"); setNewName(""); }} />
                <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  <span style={{ fontFamily: "var(--font-heading)", fontSize: 15, color: muted(55) }}>
                    Not clear enough to say
                  </span>
                  <span style={{ fontSize: 12, color: muted(60) }}>
                    Records an abstention. The turn stays unattributed and leaves the queue.
                  </span>
                </div>
                <span />
              </label>
            </div>
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <input className="input" placeholder="…or type a new name" style={{ flex: 1 }}
                     value={newName}
                     onChange={(e) => { setNewName(e.target.value); if (e.target.value) setPick(""); }} />
              <span className="mono" style={{ fontSize: 11, color: muted(48) }}>creates a profile</span>
            </div>
          </div>

          <div className="blueprint" style={{ padding: 13, display: "flex", flexDirection: "column", gap: 10 }}>
            <span className="kicker-sm" style={{ color: muted(55) }}>What saving does</span>
            <span style={{ fontSize: 13.5 }}>
              Applies to all {assign?.n_turns ?? 0} turns of this voice in this clip.
            </span>
            <label style={{ display: "flex", alignItems: "flex-start", gap: 9, fontSize: 13.5, cursor: "pointer" }}>
              <input type="checkbox" checked={enroll} onChange={(e) => setEnroll(e.target.checked)}
                     style={{ marginTop: 2 }} />
              <span>
                Add this speech as an enrollment{" "}
                <span style={{ color: muted(55) }}>
                  — {assign?.speech_s?.toFixed(1)}s, only if the voiceprint is reliable enough
                </span>
              </span>
            </label>
            <span style={{ fontSize: 12, color: muted(55) }}>
              Enrollment is skipped, with the reason shown, when the audio can't support it.
            </span>
          </div>
        </div>
      </Dialog>

      <EditClipDialog clip={edit} onClose={() => setEdit(null)} onSaved={load} />
      <DeleteClipDialog clip={del} onClose={() => setDel(null)} onDeleted={() => navigate("/")} />
    </div>
  );
}
