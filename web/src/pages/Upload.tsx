import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { api } from "@/api/client";
import { Dialog } from "@/components/dialog";
import { AMBER, AMBER_INK, muted, tagTone } from "@/lib/ui";

type Outcome = "QUEUED" | "PROCESSING" | "COMPLETE" | "REJECTED" | "FAILED" | "DUPLICATE";
type Item = {
  key: string; name: string; size: number; clipId: string | null;
  stage: string; pct: number; outcome: Outcome;
  voices: number | null; code: string | null; message: string | null;
  // Held only for a file that has not been sent yet. A File handle cannot be serialised,
  // so it is the one part of a batch that a refresh genuinely destroys.
  file?: File;
};

const STORE = "upload_batch";
const TERMINAL: Outcome[] = ["COMPLETE", "REJECTED", "FAILED", "DUPLICATE"];

/** A batch outlives the page. Processing happens on the box and takes minutes, so a
 *  refresh — or a closed laptop — must not lose the record of what was submitted.
 *
 *  Only what the API already knows is persisted: once a file has a clip id, its progress
 *  can be recovered by asking. A file still waiting to be sent cannot be, because its File
 *  handle dies with the page, so those are dropped on restore and said so out loud rather
 *  than restored as rows that would never move. */

const OUTCOME_TONE: Record<Outcome, ReturnType<typeof tagTone>> = {
  COMPLETE: tagTone("accent"), PROCESSING: tagTone("neutral"), QUEUED: tagTone("neutral"),
  REJECTED: tagTone("warn"), FAILED: tagTone("bad"), DUPLICATE: tagTone("neutral"),
};

const BAR_COLOR: Record<Outcome, string> = {
  COMPLETE: "var(--color-accent-700)", PROCESSING: "var(--color-accent-400)",
  QUEUED: "var(--color-neutral-400)", REJECTED: AMBER, FAILED: "var(--sig-red)",
  DUPLICATE: "var(--color-neutral-400)",
};

const CONCURRENCY = 3;

/** What a clip row says about itself, in the shape the table renders. Used when catching
 *  up after a refresh, where the stage-level events are long gone and the clip's own
 *  status is all that is left. */
function statusOf(clip: {
  status: string; n_speakers: number | null; error_code: string | null;
}): Partial<Item> {
  const terminal = TERMINAL.includes(clip.status as Outcome);
  return {
    outcome: (terminal ? clip.status : "PROCESSING") as Outcome,
    stage: clip.status.toLowerCase(),
    pct: terminal ? 100 : 50,
    voices: clip.n_speakers,
    code: clip.error_code,
  };
}

export default function Upload() {
  const navigate = useNavigate();
  const [items, setItems] = useState<Item[]>([]);
  // Nothing is written back to storage until the first read has happened, or the empty
  // initial state would erase the batch before it is restored.
  const [restoring, setRestoring] = useState(true);
  const [dragging, setDragging] = useState(false);
  const [races, setRaces] = useState<any[]>([]);
  const [raceId, setRaceId] = useState("");
  const [running, setRunning] = useState(false);
  const [batchStart, setBatchStart] = useState<Date | null>(null);
  const [reject, setReject] = useState<Item | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const batchId = useRef(Math.random().toString(16).slice(2, 6));

  const patch = useCallback((key: string, p: Partial<Item>) => {
    setItems((prev) => prev.map((i) => (i.key === key ? { ...i, ...p } : i)));
  }, []);

  useEffect(() => { api.listRaces().then((r) => setRaces(r.items)).catch(() => {}); }, []);

  // Persist on every change. The File is stripped: it cannot be serialised, and a row that
  // claims to hold a file it no longer has is worse than an honest gap.
  useEffect(() => {
    if (restoring) return;
    const keep = items.map(({ file, ...rest }) => rest);
    if (keep.length) localStorage.setItem(STORE, JSON.stringify(keep));
    else localStorage.removeItem(STORE);
  }, [items, restoring]);

  // Restore, then reconcile against the API. Redis pub/sub has no replay, so anything that
  // finished while the page was gone was never delivered — the truth is the clip row, and
  // asking for it is the only way to catch up.
  useEffect(() => {
    let saved: Item[] = [];
    try { saved = JSON.parse(localStorage.getItem(STORE) || "[]"); } catch { saved = []; }

    const sent = saved.filter((i) => i.clipId);
    const lost = saved.length - sent.length;
    setItems(sent);
    setRestoring(false);
    if (lost) {
      toast.info(`${lost} file${lost === 1 ? "" : "s"} never left the browser`, {
        description: "A queued file cannot survive a refresh — drop those in again.",
      });
    }

    let alive = true;
    (async () => {
      for (const item of sent) {
        if (TERMINAL.includes(item.outcome)) continue;
        try {
          const clip = await api.getClip(item.clipId!);
          if (!alive) return;
          patch(item.key, statusOf(clip));
        } catch {
          // A clip the API no longer has was deleted while we were away; say so rather
          // than leaving the row spinning on a job that will never report.
          if (alive) patch(item.key, {
            outcome: "FAILED", pct: 100, stage: "clip no longer exists",
            message: "This clip was deleted after it was uploaded.",
          });
        }
      }
    })();
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Poll whatever is still in flight. The websocket carries stage-level progress for a
  // batch this page started; after a refresh there is no socket, and a clip that finishes
  // must still stop saying "processing".
  useEffect(() => {
    const watching = items.filter((i) => i.clipId && !TERMINAL.includes(i.outcome));
    if (!watching.length) return;
    const t = setInterval(async () => {
      for (const item of watching) {
        try {
          patch(item.key, statusOf(await api.getClip(item.clipId!)));
        } catch { /* transient — the next tick tries again */ }
      }
    }, 4000);
    return () => clearInterval(t);
  }, [items, patch]);


  function addFiles(files: File[]) {
    const audio = files.filter((f) =>
      f.type.startsWith("audio/") || f.type.startsWith("video/")
      || /\.(mp3|wav|m4a|ogg|flac|webm|aac|mp4|mov)$/i.test(f.name));
    if (!audio.length) return toast.error("No audio or video files in that drop");
    setItems((prev) => [...prev, ...audio.map((file) => ({
      key: `${file.name}-${file.size}-${Math.random().toString(16).slice(2)}`,
      name: file.name, size: file.size,
      file, clipId: null, stage: "waiting", pct: 0,
      outcome: "QUEUED" as Outcome, voices: null, code: null, message: null,
    }))]);
  }

  /** One file, start to finish. The worker always emits {type:"stage", stage, state, …} —
   *  terminal outcomes are told apart by stage/state, not a dedicated event type. */
  function processOne(item: Item) {
    return new Promise<void>(async (resolve) => {
      try {
        const res = raceId
          ? await api.uploadRaceClip(raceId, item.file!)
          : await api.uploadClip(item.file!);
        patch(item.key, { clipId: res.clip_id });
        if (res.duplicate) {
          patch(item.key, {
            outcome: "DUPLICATE", stage: "same audio as an existing clip", pct: 100,
          });
          return resolve();
        }
        patch(item.key, { outcome: "PROCESSING", stage: "queued", pct: 2 });

        const ws = new WebSocket(`${location.origin.replace("http", "ws")}${res.ws_url}`);
        ws.onmessage = (e) => {
          const msg = JSON.parse(e.data);
          if (msg.type !== "stage") return;
          const stage = String(msg.stage || "").toLowerCase();
          patch(item.key, {
            stage: msg.state === "rejected" ? `rejected — ${msg.message || msg.code}` : stage,
            pct: msg.progress ?? undefined,
          });
          if (msg.state === "rejected") {
            patch(item.key, { outcome: "REJECTED", pct: 100, code: msg.code, message: msg.message });
            ws.close(); resolve();
          } else if (msg.stage === "FAILED") {
            patch(item.key, { outcome: "FAILED", pct: 100, code: msg.code, message: msg.message });
            ws.close(); resolve();
          } else if (msg.stage === "COMPLETE") {
            patch(item.key, { outcome: "COMPLETE", stage: "complete", pct: 100 });
            api.getClip(res.clip_id).then((c) => patch(item.key, { voices: c.n_speakers })).catch(() => {});
            ws.close(); resolve();
          }
        };
        ws.onerror = () => {
          patch(item.key, { outcome: "FAILED", message: "lost connection to the processing job" });
          resolve();
        };
      } catch (e) {
        patch(item.key, { outcome: "FAILED", message: String(e), pct: 100 });
        resolve();
      }
    });
  }

  async function runBatch() {
    const queue = items.filter((i) => i.outcome === "QUEUED" && i.file);
    if (!queue.length) return;
    setRunning(true);
    setBatchStart(new Date());
    // bounded concurrency: the API streams each file to disk, and firing everything at once
    // just buries the event loop and the browser's connection pool for no extra throughput
    const pending = [...queue];
    await Promise.all(Array.from({ length: CONCURRENCY }, async () => {
      while (pending.length) await processOne(pending.shift()!);
    }));
    setRunning(false);
  }

  const counts = {
    complete: items.filter((i) => i.outcome === "COMPLETE").length,
    processing: items.filter((i) => i.outcome === "PROCESSING").length,
    rejected: items.filter((i) => i.outcome === "REJECTED" || i.outcome === "FAILED").length,
    duplicate: items.filter((i) => i.outcome === "DUPLICATE").length,
    queued: items.filter((i) => i.outcome === "QUEUED").length,
  };
  const totalDone = counts.complete + counts.rejected + counts.duplicate;
  const share = (n: number) => (items.length ? `${(n / items.length) * 100}%` : "0%");

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <header style={{
        display: "flex", alignItems: "flex-end", justifyContent: "space-between",
        gap: 20, flexWrap: "wrap",
      }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
          <span className="kicker">
            Batch {batchId.current}
            {batchStart && ` · started ${batchStart.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}`}
          </span>
          <h2 style={{ margin: 0, fontSize: 30 }}>Upload &amp; process</h2>
          <p style={{ margin: 0, maxWidth: "64ch", fontSize: 14, color: muted(68) }}>
            Drop a folder of audio and watch it through the pipeline. Files are processed with
            the settings in force at submit time; duplicates are recognised by audio, not
            filename, and jump to the existing result.
          </p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn btn-secondary" disabled={!items.length || running}
                  onClick={() => { setItems([]); localStorage.removeItem(STORE); }}>
            Clear batch
          </button>
          <button className="btn btn-primary" onClick={() => navigate("/")}>Open in Library</button>
        </div>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: 20, alignItems: "start" }}>
        <div
          className="blueprint hatch"
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); addFiles(Array.from(e.dataTransfer.files)); }}
          onClick={() => inputRef.current?.click()}
          style={{
            padding: 34, display: "flex", flexDirection: "column", alignItems: "center",
            gap: 8, borderStyle: "dashed", cursor: "pointer",
            borderColor: dragging ? "var(--color-accent)" : undefined,
          }}
        >
          <input
            ref={inputRef} type="file" multiple accept="audio/*,video/*" style={{ display: "none" }}
            onChange={(e) => { addFiles(Array.from(e.target.files ?? [])); e.target.value = ""; }}
          />
          <span style={{ fontFamily: "var(--font-heading)", fontSize: 22 }}>Drop audio or video here</span>
          <span style={{ fontSize: 13, color: muted(62) }}>
            wav · flac · m4a · mp3 · mp4 · mov
          </span>
          <button className="btn btn-secondary" style={{ marginTop: 6 }}
                  onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}>
            Browse files
          </button>
        </div>

        <div className="blueprint" style={{ padding: 14, display: "flex", flexDirection: "column", gap: 12 }}>
          <span className="kicker-sm" style={{ color: muted(55) }}>
            Applies to every file in this batch
          </span>
          <div className="field">
            <label>Add to race</label>
            <select className="input" value={raceId} onChange={(e) => setRaceId(e.target.value)}>
              <option value="">None</option>
              {races.map((r) => <option key={r.id} value={r.id}>{r.name}</option>)}
            </select>
          </div>
          <button className="btn btn-primary" disabled={!counts.queued || running} onClick={runBatch}>
            {running ? "Processing…" : `Process ${counts.queued || ""} file${counts.queued === 1 ? "" : "s"}`}
          </button>
          <div style={{
            paddingTop: 10, borderTop: "1px solid var(--color-divider)",
            display: "flex", flexDirection: "column", gap: 5,
          }}>
            <span className="mono" style={{
              display: "flex", justifyContent: "space-between", fontSize: 11.5, color: muted(58),
            }}>
              <span>done</span><span>{totalDone} of {items.length}</span>
            </span>
            <span style={{ fontSize: 11.5, color: muted(55) }}>
              Files that are rejected state their reason and are not retried silently.
            </span>
          </div>
        </div>
      </div>

      {items.length > 0 && (
        <>
          <section style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{
              display: "flex", alignItems: "center", justifyContent: "space-between",
              gap: 16, flexWrap: "wrap",
            }}>
              <div style={{ display: "flex", alignItems: "baseline", gap: 14 }}>
                <h4 style={{ margin: 0 }}>Batch progress</h4>
                <span className="mono" style={{ fontSize: 12, color: muted(58) }}>
                  {totalDone} of {items.length} done
                </span>
              </div>
              <div className="mono" style={{ display: "flex", gap: 14, fontSize: 11.5 }}>
                {([["complete", counts.complete, "var(--color-accent-700)"],
                   ["processing", counts.processing, "var(--color-accent-400)"],
                   ["rejected", counts.rejected, AMBER],
                   ["duplicate", counts.duplicate, "var(--color-neutral-500)"]] as const).map(([label, n, color]) => (
                  <span key={label} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <span style={{ width: 9, height: 9, background: color }} />{n} {label}
                  </span>
                ))}
              </div>
            </div>
            <div style={{ height: 12, display: "flex", background: muted(10) }}>
              <div style={{ width: share(counts.complete), background: "var(--color-accent-700)" }} />
              <div style={{ width: share(counts.processing), background: "var(--color-accent-400)" }} />
              <div style={{ width: share(counts.rejected), background: AMBER }} />
              <div style={{ width: share(counts.duplicate), background: "var(--color-neutral-500)" }} />
            </div>
          </section>

          <table className="table">
            <thead>
              <tr>
                <th>File</th><th>Size</th><th style={{ width: "34%" }}>Stage</th>
                <th>Outcome</th><th>Voices</th><th />
              </tr>
            </thead>
            <tbody>
              {items.map((i) => {
                const tone = OUTCOME_TONE[i.outcome];
                return (
                  <tr key={i.key}>
                    <td className="mono" style={{ fontSize: 12.5 }}>{i.name}</td>
                    <td className="mono" style={{ fontSize: 12.5, color: muted(60) }}>
                      {(i.size / 1024 / 1024).toFixed(1)} MB
                    </td>
                    <td>
                      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                        <div className="mono" style={{
                          display: "flex", justifyContent: "space-between", fontSize: 11, color: muted(60),
                        }}>
                          <span>{i.stage}</span><span>{Math.round(i.pct)}%</span>
                        </div>
                        <div className="bar" style={{ height: 5 }}>
                          <span style={{ width: `${i.pct}%`, background: BAR_COLOR[i.outcome] }} />
                        </div>
                      </div>
                    </td>
                    <td>
                      <span className="tag mono" style={{ border: `1px solid ${tone.border}`, color: tone.color }}>
                        {i.outcome}
                      </span>
                    </td>
                    <td className="mono" style={{ fontSize: 12.5 }}>{i.voices ?? "—"}</td>
                    <td style={{ textAlign: "right" }}>
                      {(i.outcome === "REJECTED" || i.outcome === "FAILED") ? (
                        <button className="btn btn-ghost" style={{ fontSize: 12, padding: "4px 7px" }}
                                onClick={() => setReject(i)}>Why?</button>
                      ) : i.clipId ? (
                        <button className="btn btn-ghost" style={{ fontSize: 12, padding: "4px 7px" }}
                                onClick={() => navigate(`/clips/${i.clipId}`)}>
                          {i.outcome === "DUPLICATE" ? "Go to original" : "Open clip"}
                        </button>
                      ) : null}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p style={{ margin: 0, fontSize: 12, color: muted(55) }}>
            Rejections state their reason and are not retried silently. A rejected file is a
            fact about the audio, not a failure of the run.
          </p>
        </>
      )}

      <Dialog
        open={reject != null}
        onClose={() => setReject(null)}
        kicker="Upload · batch"
        title="Why this file was rejected"
        subject={reject ? `${reject.name} · ${(reject.size / 1024 / 1024).toFixed(1)} MB` : ""}
        cancel="Close"
        width="560px"
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <p style={{ margin: 0, fontSize: 14, lineHeight: 1.55, color: muted(75) }}>
            The file was rejected on its result, not lost to an error. Nothing is retried in
            the background.
          </p>
          <div className="blueprint" style={{ padding: 12, display: "flex", flexDirection: "column", gap: 8 }}>
            <span className="kicker-sm" style={{ color: AMBER_INK }}>Reason from the worker</span>
            <span className="mono" style={{ fontSize: 12.5 }}>{reject?.code || "unspecified"}</span>
            <span style={{ fontSize: 13, color: muted(70) }}>
              {reject?.message || "The worker gave no further detail."}
            </span>
          </div>
          <span style={{ fontSize: 13, color: muted(65) }}>
            You can re-drop the file after fixing the audio — the pipeline recognises duplicates
            by content, so an unchanged file will jump to its existing result rather than
            processing twice.
          </span>
        </div>
      </Dialog>
    </div>
  );
}
