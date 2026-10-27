const API_KEY = localStorage.getItem("api_key") || "change-me";

async function req(path: string, opts: RequestInit = {}) {
  const res = await fetch(path, {
    ...opts,
    headers: { ...opts.headers, Authorization: `Bearer ${API_KEY}` },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    // A structured detail ({message, existing_id}) rides along on the error so a caller can
    // act on it — e.g. offer a merge when a rename collides with an existing speaker.
    const detail = body.detail;
    const err = new Error(typeof detail === "string" ? detail
      : detail?.message || res.statusText) as Error & { status?: number; detail?: unknown };
    err.status = res.status;
    err.detail = detail;
    throw err;
  }
  return res.json();
}

const json = (method: string, body: unknown): RequestInit => ({
  method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

export const api = {
  // `tags` repeats the ?tag= parameter; the server requires every one of them.
  listClips: (params: Record<string, string> = {}, tags: string[] = []) => {
    const qs = new URLSearchParams(params);
    for (const t of tags) qs.append("tag", t);
    return req(`/v1/clips?${qs}`);
  },
  // Every tag in use with its clip count, operator tags and auto tags reported apart.
  tagCounts: () => req("/v1/clips/tags"),
  retagClip: (id: string) => req(`/v1/clips/${id}/retag`, { method: "POST" }),
  retagAll: () => req("/v1/admin/retag", { method: "POST" }),
  maintenanceStatus: () => req("/v1/admin/maintenance"),
  runMaintenance: () => req("/v1/admin/maintenance/run", { method: "POST" }),
  getClip: (id: string) => req(`/v1/clips/${id}`),
  getResult: (id: string) => req(`/v1/clips/${id}/result`),
  uploadClip: (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return req("/v1/clips", { method: "POST", body: fd });
  },
  deleteClip: (id: string) => req(`/v1/clips/${id}`, { method: "DELETE" }),
  patchClip: (id: string, body: { filename?: string; tags?: string[]; notes?: string; needs_review?: boolean }) =>
    req(`/v1/clips/${id}`, json("PATCH", body)),
  // One action over many clips: delete | reprocess | add_tag | remove_tag | mark_reviewed | flag_review
  bulkClips: (ids: string[], action: string, tag?: string) =>
    req("/v1/clips/bulk", json("POST", { ids, action, tag })),
  // Correct one transcript line; the search index follows.
  editUtterance: (clipId: string, utteranceId: string, text: string) =>
    req(`/v1/clips/${clipId}/utterances/${utteranceId}`, json("PATCH", { text })),
  uploadLiveChunk: (sessionId: string, seq: number, blob: Blob) => {
    const fd = new FormData();
    fd.append("file", blob, `${seq}.webm`);
    return req(`/v1/live/${sessionId}/chunk?seq=${seq}`, { method: "POST", body: fd });
  },
  reprocess: (id: string) => req(`/v1/clips/${id}/reprocess`, { method: "POST" }),
  assignSpeaker: (clipId: string, label: string, body: object) =>
    req(`/v1/clips/${clipId}/speakers/${label}/assign`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  listSpeakers: () => req("/v1/speakers"),
  createSpeaker: (body: { display_name: string; notes?: string; tags?: string[] }) =>
    req("/v1/speakers", json("POST", body)),
  patchSpeaker: (id: string, body: { display_name?: string; notes?: string; status?: string; tags?: string[] }) =>
    req(`/v1/speakers/${id}`, json("PATCH", body)),
  // reassign_to merges the profile's voices into another instead of unassigning them.
  deleteSpeaker: (id: string, reassignTo?: string) =>
    req(`/v1/speakers/${id}${reassignTo ? `?reassign_to=${reassignTo}` : ""}`, { method: "DELETE" }),
  deleteEnrollment: (id: string, eid: string) =>
    req(`/v1/speakers/${id}/enrollments/${eid}`, { method: "DELETE" }),
  speakerRelations: (id: string) => req(`/v1/speakers/${id}/relations`),
  speakerUtterances: (id: string, limit = 50, offset = 0) =>
    req(`/v1/speakers/${id}/utterances?limit=${limit}&offset=${offset}`),
  // Profile pairs whose voiceprints say they are probably one person.
  mergeSuggestions: () => req("/v1/speakers/merge-suggestions"),
  // Voices in other clips that sound like this profile and are not attributed to it.
  speakerCandidates: (id: string) => req(`/v1/speakers/${id}/candidates`),
  // Confirm candidate voices as this person — each as if assigned from its own clip page.
  claimVoices: (id: string, items: { clip_id: string; local_label: string }[], enroll = true) =>
    req(`/v1/speakers/${id}/claim`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items, enroll }),
    }),
  // Search by speaking: what was said (speech), who is speaking (speaker), or both.
  voiceSearch: (audio: Blob, mode: "both" | "speech" | "speaker" = "both", filename = "query.webm") => {
    const fd = new FormData();
    fd.append("file", audio, filename);
    fd.append("mode", mode);
    return req("/v1/search/voice", { method: "POST", body: fd });
  },
  getSpeaker: (id: string) => req(`/v1/speakers/${id}`),
  speakerClips: (id: string) => req(`/v1/speakers/${id}/clips`),
  // keep_id absorbs source_id: the kept profile gains the other's enrollments and its
  // centroid (and cohesion) are recomputed from the combined set.
  mergeSpeakers: (keepId: string, sourceIds: string | string[]) =>
    req(`/v1/speakers/${keepId}/merge`,
        json("POST", { source_ids: Array.isArray(sourceIds) ? sourceIds : [sourceIds] })),
  listClusters: () => req("/v1/clusters"),
  getCluster: (id: string) => req(`/v1/clusters/${id}`),
  // <audio src> can't carry an Authorization header, hence the ?key= form get_current_user
  // also accepts. First hit stitches the montage with ffmpeg; later hits are served cached.
  clusterMontageUrl: (id: string) =>
    `/v1/clusters/${id}/montage?key=${encodeURIComponent(API_KEY)}`,
  // Name a voice: a new profile by name, or into someone already enrolled by profile_id.
  promoteCluster: (id: string, target: { display_name?: string; profile_id?: string }) =>
    req(`/v1/clusters/${id}/promote`, json("POST", target)),
  dismissCluster: (id: string) => req(`/v1/clusters/${id}`, { method: "DELETE" }),
  search: (q: string, opts: { mode?: string; tags?: string[]; speaker_id?: string; clip_id?: string; limit?: number } = {}) =>
    req("/v1/search", json("POST", {
      q, mode: opts.mode || "hybrid", tags: opts.tags || [], speaker_id: opts.speaker_id || null,
      clip_id: opts.clip_id || null, limit: opts.limit || 40,
    })),
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
  graphSummary: () => req("/v1/admin/graph/summary"),

  graphSync: () => req("/v1/admin/graph/sync", { method: "POST" }),
  // The whole graph minus individual lines of speech, for the Graph page.
  graphMap: (limit = 600) => req(`/v1/graph/map?limit=${limit}`),
  graphNode: (id: string) => req(`/v1/graph/node/${encodeURIComponent(id)}`),
};
