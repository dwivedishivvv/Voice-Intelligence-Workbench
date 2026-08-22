const API_KEY = localStorage.getItem("api_key") || "change-me";

async function req(path: string, opts: RequestInit = {}) {
  const res = await fetch(path, {
    ...opts,
    headers: { ...opts.headers, Authorization: `Bearer ${API_KEY}` },
  });
  if (!res.ok) throw new Error((await res.json().catch(() => ({ detail: res.statusText }))).detail || res.statusText);
  return res.json();
}

export const api = {
  listClips: (params: Record<string, string> = {}) =>
    req(`/v1/clips?${new URLSearchParams(params)}`),
  getClip: (id: string) => req(`/v1/clips/${id}`),
  getResult: (id: string) => req(`/v1/clips/${id}/result`),
  uploadClip: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req("/v1/clips", { method: "POST", body: fd });
  },
  deleteClip: (id: string) => req(`/v1/clips/${id}`, { method: "DELETE" }),
  patchClip: (id: string, body: { tags?: string[]; notes?: string; needs_review?: boolean }) =>
    req(`/v1/clips/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  uploadLiveChunk: (sessionId: string, seq: number, blob: Blob) => {
    const fd = new FormData();
    fd.append("file", blob, `${seq}.webm`);
    return req(`/v1/live/${sessionId}/chunk?seq=${seq}`, { method: "POST", body: fd });
  },
  // Names an unmatched (or only-suggested) live turn from the embedding the worker already
  // computed for it — no re-recording needed. Enrolls into an existing profile of that name
  // if one exists, otherwise creates one; every later chunk's identify() call can then match it.
  tagLiveSpeaker: (body: { display_name: string; embedding: number[]; duration_s: number; reliability: number }) =>
    req("/v1/live/tag", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  reprocess: (id: string) => req(`/v1/clips/${id}/reprocess`, { method: "POST" }),
  assignSpeaker: (clipId: string, label: string, body: object) =>
    req(`/v1/clips/${clipId}/speakers/${label}/assign`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  listSpeakers: () => req("/v1/speakers"),
  getSpeaker: (id: string) => req(`/v1/speakers/${id}`),
  speakerClips: (id: string) => req(`/v1/speakers/${id}/clips`),
  // keep_id absorbs source_id: the kept profile gains the other's enrollments and its
  // centroid (and cohesion) are recomputed from the combined set.
  mergeSpeakers: (keepId: string, sourceId: string) =>
    req(`/v1/speakers/${keepId}/merge`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source_id: sourceId }),
    }),
  listClusters: () => req("/v1/clusters"),
  getCluster: (id: string) => req(`/v1/clusters/${id}`),
  // <audio src> can't carry an Authorization header, hence the ?key= form get_current_user
  // also accepts. First hit stitches the montage with ffmpeg; later hits are served cached.
  clusterMontageUrl: (id: string) =>
    `/v1/clusters/${id}/montage?key=${encodeURIComponent(API_KEY)}`,
  promoteCluster: (id: string, display_name: string) =>
    req(`/v1/clusters/${id}/promote`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name }),
    }),
  search: (q: string, mode = "hybrid") =>
    req("/v1/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ q, mode }),
    }),
  stats: () => req("/v1/admin/stats"),
  getSettings: () => req("/v1/admin/settings"),
  patchSettings: (values: Record<string, boolean | number | string>) =>
    req("/v1/admin/settings", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values }),
    }),
  resetSetting: (key: string) => req(`/v1/admin/settings/${key}`, { method: "DELETE" }),
  // Measures thresholds against your own enrollments and returns suggestions. It writes a
  // calibration_runs row, never a setting — applying is a separate, explicit PATCH.
  calibrate: () => req("/v1/admin/calibrate", { method: "POST" }),
  // Merges unclaimed clusters that are the same voice. Online clustering can only compare
  // against the clusters that existed at the time, so one person drifts across several.
  consolidateClusters: () => req("/v1/admin/clusters/consolidate", { method: "POST" }),
  // The agent's own config. Keys are write-only: the GET reports whether one is installed
  // and its last four characters, never the key itself.
  getLLM: () => req("/v1/admin/llm"),
  patchLLM: (body: {
    enabled?: boolean; provider?: string; model?: string;
    api_key_provider?: string; api_key?: string;
  }) =>
    req("/v1/admin/llm", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  listModels: () => req("/v1/admin/models"),
  pullModel: (category: string, repo_id: string) =>
    req("/v1/admin/models/pull", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ category, repo_id }),
    }),
  activateModel: (category: string, repo_id: string) =>
    req("/v1/admin/models/activate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ category, repo_id }),
    }),
  agentAsk: (question: string, history: unknown[] | null, conversationId: string, signal?: AbortSignal) =>
    req("/v1/agent/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history, conversation_id: conversationId }),
      signal,
    }),
  // Tool-less one-shot note (e.g. a live lap comparison) — deliberately not agentAsk,
  // whose corpus-QA system prompt forces tool use and has no live-session context to search.
  agentAnalyze: (prompt: string) =>
    req("/v1/agent/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    }),
  graphSummary: () => req("/v1/admin/graph/summary"),

  listRaces: () => req("/v1/races"),
  getRace: (id: string) => req(`/v1/races/${id}`),
  raceUtterances: (id: string) => req(`/v1/races/${id}/utterances`),
  raceStats: (id: string) => req(`/v1/races/${id}/stats`),
  // Without deleteClips the recordings survive, unfiled — the race is only a grouping.
  // With it, each one goes through the same delete path as a single clip: files off disk,
  // row gone, deletion receipt written.
  deleteRace: (id: string, deleteClips = false) =>
    req(`/v1/races/${id}?delete_clips=${deleteClips}`, { method: "DELETE" }),
  createRace: (fields: Record<string, string>, svg: File | null) => {
    const fd = new FormData();
    for (const [k, v] of Object.entries(fields)) fd.append(k, v);
    if (svg) fd.append("svg", svg);
    return req("/v1/races", { method: "POST", body: fd });
  },
  nameRaceVoice: (raceId: string, clusterId: string, display_name: string) =>
    req(`/v1/races/${raceId}/voices/${clusterId}/name`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name }),
    }),
  uploadRaceClip: (raceId: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req(`/v1/races/${raceId}/clips`, { method: "POST", body: fd });
  },
  // the SVG route needs auth but is used as an <img>/fetch source, hence the ?key= form
  raceSvgUrl: (id: string) => `/v1/races/${id}/svg?key=${encodeURIComponent(API_KEY)}`,

  // Demo hook for Live: an official F1 livetiming radio clip, run through the same
  // transcribe+tone analysis as a live mic chunk (see api/app/routers/f1.py::ingest).
  // Cached calls come back inline; a fresh one returns a ws_url on the shared job:{id}
  // channel — same contract every other job publishes on.
  ingestF1Radio: (recording_url: string) =>
    req("/v1/f1/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ recording_url }),
    }),
  // Same analysis, for a radio clip that only exists as a local mp3 (no livetiming URL).
  uploadF1Radio: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req("/v1/f1/ingest/upload", { method: "POST", body: fd });
  },
};
