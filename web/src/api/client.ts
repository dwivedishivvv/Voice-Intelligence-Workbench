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
  reprocess: (id: string) => req(`/v1/clips/${id}/reprocess`, { method: "POST" }),
  assignSpeaker: (clipId: string, label: string, body: object) =>
    req(`/v1/clips/${clipId}/speakers/${label}/assign`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  listSpeakers: () => req("/v1/speakers"),
  getSpeaker: (id: string) => req(`/v1/speakers/${id}`),
  listClusters: () => req("/v1/clusters"),
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
  f1Sessions: (year?: number) =>
    req(`/v1/f1/sessions${year ? `?year=${year}` : ""}`),
  f1Drivers: (sessionKey: number) => req(`/v1/f1/drivers?session_key=${sessionKey}`),
  f1Laps: (sessionKey: number, driverNumber: number) =>
    req(`/v1/f1/laps?session_key=${sessionKey}&driver_number=${driverNumber}`),
  f1TeamRadio: (sessionKey: number, driverNumber: number) =>
    req(`/v1/f1/team_radio?session_key=${sessionKey}&driver_number=${driverNumber}`),
  f1TeamRadioAll: (sessionKey: number) => req(`/v1/f1/team_radio?session_key=${sessionKey}`),
  f1Ingest: (recording_url: string, session_key?: number, driver_number?: number) =>
    req("/v1/f1/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ recording_url, session_key, driver_number }),
    }),
};
