# Settings reference

Every knob in `common/config.py`, what it does, what values are legal, and how the Settings
page should take them. Three tiers decide where a setting can be edited:

| Tier | Set defined in | Editable where | When it bites |
|---|---|---|---|
| **Tunable** | `TUNABLE_FIELDS` | Settings page, live | Next job — no restart |
| **Restart-tunable** | `RESTART_TUNABLE_FIELDS` | Settings page → System tab | Next worker start (stored as *pending*) |
| **Read-only** | everything else | `.env` only | Next process start |

Storage: `settings_overrides (key, value TEXT, updated_at, updated_by)`. Values are stored as
strings and coerced back by `_coerce()` in `common/config.py`. `get_effective_settings()`
layers them over the env defaults; `get_settings()` returns env-only. Reset = `DELETE
/v1/admin/settings/{key}`, which drops the row and falls back to the env default.

Validation today is **type-only** — `patch_settings()` runs `Settings.model_validate()`, so a
`float` field accepts `-999.0` as happily as `0.5`. The "Valid range" column below is what the
UI should enforce, not what the API currently enforces. See [Gaps](#gaps-and-bugs) at the end.

---

## Current UI control mapping

`web/src/pages/Settings.tsx` picks a control from `field.type` alone:

| `type` | Control | Entry format |
|---|---|---|
| `bool` | `<Switch>` | immediate save on toggle |
| `int` | `<Input type="number">` | integer string, `parseInt`, 600 ms debounce |
| `float` | `<Input type="number" step="0.01">` | decimal string, `parseFloat`, 600 ms debounce |
| `str` | `<Input type="text">` | free text, 600 ms debounce |
| Literal (System tab only) | `<Select>` | options come from `get_args(annotation)` |

`NaN` is dropped silently by `debouncedSave`, so a half-typed `-` or `.` doesn't fire a save.

---

## Tunable — live, applies to the next job

### Ingest

| Key | Type | Default | Valid range | Entry format | What it does |
|---|---|---|---|---|---|
| `max_upload_mb` | int | 500 | 1–2000 | number, step 1, suffix "MB" | Rejects larger uploads at the API boundary |
| `max_duration_s` | float | 3600.0 | 1–3600, `> target_duration_s` | number, step 1, suffix "s" | Hard ceiling; longer audio is rejected |
| `target_duration_s` | float | 60.0 | 1–`max_duration_s` | number, step 1, suffix "s" | Preferred clip length; longer clips get trimmed |
| `min_duration_s` | float | 0.5 | 0.1–10 | number, step 0.1, suffix "s" | Floor below which a file isn't worth processing |

### Pre-processing

| Key | Type | Default | Valid range | Entry format | What it does |
|---|---|---|---|---|---|
| `highpass_hz` | float | 70.0 | 0–300 | number, step 5, suffix "Hz" | High-pass cutoff; 0 disables |
| `target_lufs` | float | -23.0 | -40 to -10 | number, step 0.5, suffix "LUFS" | Loudness normalisation target (EBU R128 is -23) |
| `denoise_enabled` | bool | true | — | switch | Toggles spectral-gate denoise |
| `denoise_prop_decrease` | float | 0.75 | 0.0–1.0 | **slider** 0–1 step 0.05 + numeric readout | Denoise strength; 1.0 is maximally aggressive |
| `clipping_threshold` | float | 0.99 | 0.5–1.0 | number, step 0.01 | Sample amplitude counted as clipped |
| `vad_min_speech_ms` | int | 200 | 50–2000 | number, step 50, suffix "ms" | Shortest run VAD will call speech |
| `vad_min_silence_ms` | int | 150 | 50–2000 | number, step 50, suffix "ms" | Silence needed to split two speech runs |
| `vad_speech_pad_ms` | int | 100 | 0–1000 | number, step 25, suffix "ms" | Padding added either side of each region |
| `min_total_speech_s` | float | 0.5 | 0.1–30 | number, step 0.1, suffix "s" | Below this → `NO_SPEECH_DETECTED`. Don't raise past ~0.6: real radio calls ("Box, box.") are under a second |

### Quality grading

| Key | Type | Default | Valid range | Entry format | What it does |
|---|---|---|---|---|---|
| `quality_good_snr_db` | float | 18.0 | 0–60, `>` fair | number, step 1, suffix "dB" | SNR at or above → GOOD |
| `quality_fair_snr_db` | float | 10.0 | 0–`good` | number, step 1, suffix "dB" | SNR at or above → FAIR, else POOR |
| `quality_max_clipping` | float | 0.02 | 0.0–1.0 | number, step 0.005, render as % | Clipped-sample fraction that downgrades the grade |
| `quality_min_bandwidth_hz` | float | 3400.0 | 1000–8000 | number, step 100, suffix "Hz" | Bandwidth floor; telephone-band audio sits at ~3400 |

### Diarization

| Key | Type | Default | Valid range | Entry format | What it does |
|---|---|---|---|---|---|
| `diar_min_speakers` | int | 1 | 1–`diar_max_speakers` | number, step 1 | Lower bound handed to pyannote |
| `diar_max_speakers` | int | 50 | `diar_min_speakers`–50 | number, step 1 | Upper bound handed to pyannote; 50 stands in for unlimited (see common/config.py) |
| `min_turn_s` | float | 0.4 | 0.05–5 | number, step 0.05, suffix "s" | Turns shorter than this are dropped |
| `merge_gap_s` | float | 0.6 | 0.0–5 | number, step 0.1, suffix "s" | Same-speaker turns closer than this are merged |
| `vad_snap_tol_s` | float | 0.15 | 0.0–1 | number, step 0.05, suffix "s" | Tolerance for snapping diarization edges to VAD edges |
| `overlap_warn_ratio` | float | 0.30 | 0.0–1.0 | number, step 0.05, render as % | Overlapped-speech fraction that raises a warning |
| `smooth_min_conf` | float | 0.60 | 0.0–1.0 | number, step 0.05 | Minimum confidence to keep a smoothed label |

### Speaker identification

The comment block above these in `config.py` is load-bearing: the defaults are **calibrated on
70 F1 team-radio clips**, not universal constants. `POST /v1/admin/calibrate` re-derives
`id_threshold` / `verify_threshold` / `cluster_threshold` from your own enrolled profiles (EER)
once you have ≥3 profiles with ≥2 enrollments each. Prefer that over hand-tuning.

| Key | Type | Default | Valid range | Entry format | What it does |
|---|---|---|---|---|---|
| `embed_min_s` | float | 1.5 | 0.5–10 | number, step 0.1, suffix "s" | Shortest speech that yields an embedding |
| `embed_target_s` | float | 8.0 | `embed_min_s`–60 | number, step 0.5, suffix "s" | Speech length aimed for when pooling |
| `reliability_good` | float | 0.75 | 0.0–1.0, `>` fair | number, step 0.05 | Embedding-reliability band: GOOD |
| `reliability_fair` | float | 0.45 | 0.0–`good`, `>` poor | number, step 0.05 | Band: FAIR |
| `reliability_poor` | float | 0.20 | 0.0–`fair` | number, step 0.05 | Band: POOR (below → unusable) |
| `id_threshold` | float | 0.55 | 0.0–1.0 | number, step 0.01 | Cosine similarity needed to attach a name |
| `id_suggest_delta` | float | 0.08 | 0.0–0.5 | number, step 0.01 | Below-threshold window that still shows as a *suggestion* |
| `id_min_margin` | float | 0.04 | 0.0–0.5 | number, step 0.01 | Required gap between best and runner-up, else abstain |
| `id_threshold_penalty` | float | 0.10 | 0.0–0.5 | number, step 0.01 | Threshold raised by this much on low-reliability embeddings |
| `verify_threshold` | float | 0.62 | 0.0–1.0 | number, step 0.01 | 1:1 verification cutoff (stricter than identification) |
| `cluster_threshold` | float | 0.52 | 0.0–1.0 | number, step 0.01 | Cross-clip clustering cutoff. At 0.76 the same driver re-clustered as a new person 95% of the time |
| `auto_enroll` | bool | false | — | switch | Auto-enrol a cluster as a profile when confident. **Off by default on purpose** — a wrong auto-enrolment poisons every later match |
| `auto_enroll_min_sim` | float | 0.85 | 0.0–1.0 | number, step 0.01; disable unless `auto_enroll` | Similarity required to auto-enrol |
| `auto_enroll_min_reliability` | float | 0.75 | 0.0–1.0 | number, step 0.01; disable unless `auto_enroll` | Reliability required to auto-enrol |

### Transcription

| Key | Type | Default | Valid range | Entry format | What it does |
|---|---|---|---|---|---|
| `asr_beam_size` | int | 5 | 1–10 | number, step 1 | Whisper beam width. Higher = slower, marginally better |
| `asr_language` | str | `auto` | `auto` \| ISO-639-1 code | **should be a `<Select>`**, not free text | `auto` → language detection; anything else pins it. Read in `worker/main.py:119` and `stages/transcribe.py:36` as `None if == "auto"` |

Suggested `asr_language` option list: `auto, en, es, fr, de, it, pt, nl, ja, zh, ko, ar, ru, hi,
tr, pl, sv, fi, da, no`. A typo here silently pins ASR to a wrong language — the exact failure
`device` was made a `Literal` to avoid.

### Jobs & retention

| Key | Type | Default | Valid range | Entry format | What it does |
|---|---|---|---|---|---|
| `job_timeout_s` | int | 300 | 30–3600 | number, step 30, suffix "s" | arq per-job timeout |
| `job_max_attempts` | int | 3 | 1–10 | number, step 1 | Retries before a job is marked failed |
| `retention_days` | int | 0 | 0–3650, 0 = keep forever | number, step 1, suffix "days" | **Currently does nothing** — see [Gaps](#gaps-and-bugs) |

### Graph

| Key | Type | Default | Valid range | Entry format | What it does |
|---|---|---|---|---|---|
| `graph_lap_match_tolerance_s` | float | 2.0 | 0.0–30 | number, step 0.5, suffix "s" | Slack when matching speech to the lap it happened on. Speech inside the overlap links to **both** laps rather than being assigned by coin flip. Applies to the next *graph sync*, not the next job — the tab should say so |

---

## Restart-tunable — System tab, applies after a worker restart

| Key | Type | Default | Options | Entry format |
|---|---|---|---|---|
| `device` | Literal | `auto` | `auto`, `cpu`, `cuda` | `<Select>`, options from `get_args()`. Saved as *pending*; UI shows `current → pending` and a clear button |

---

## Read-only in the UI — `.env` only

Shown on the System tab so it's obvious *why* they aren't editable
(`RESTART_REQUIRED_FIELDS` in `api/app/routers/admin.py`), because `ModelPool` bakes them in at
worker startup.

| Key | Type | Default | Legal values |
|---|---|---|---|
| `precision` | str | `int8` | `int8`, `int8_float16`, `float16`, `float32` — CTranslate2 `compute_type`. Forced to `int8` on CPU (`worker/pool.py:53`) |
| `worker_concurrency` | int | 2 | 1–16. Also sizes the DB pool (`max_size = concurrency + 4`) |
| `model_dir` | str | `/models` | Absolute path. `C:/models` on Windows — `models/REGISTRY.yaml` hardcodes `/models/...` |
| `asr_model` | str | `large-v3-turbo` | Registry key |
| `vad_model` | str | `silero-vad` | Registry key |
| `diar_model` | str | `pyannote-3.1` | Registry key |
| `embed_model` | str | `ecapa` | Registry key |
| `text_embed_model` | str | `minilm` | Registry key |
| `sentiment_model` | str | `xlmr-sentiment` | Registry key — **missing from `RESTART_REQUIRED_FIELDS`**, so it's the one model the System tab doesn't list |

Model *downloads* and activation are handled by the Models tab (`ModelsPanel`), not these
fields.

### Infrastructure — never in the UI

| Key | Type | Default | Notes |
|---|---|---|---|
| `app_env` | str | `production` | Currently read nowhere |
| `log_level` | str | `info` | `debug`/`info`/`warning`/`error` |
| `log_transcripts` | bool | false | Leave off outside debugging — writes transcript text to logs |
| `api_key` | str | `change-me` | Stripped from `config_snapshot()`. Web UI reads it from `localStorage` |
| `data_dir` | str | `/data` | Paths are stored **relative** to this (`common/storage.py`), so API and worker may differ |
| `postgres_host/port/db/user/password` | — | localhost:5432 | Password stripped from `config_snapshot()` |
| `redis_url` | str | `redis://localhost:6379/0` | arq queue |
| `hf_hub_offline` / `transformers_offline` | int | 1 | Worker must never fetch weights mid-job |

---

## Want — settings that exist in config but have no UI at all

These are all real fields today, reachable only by editing `.env` and restarting. Each needs a
deliberate decision about whether it *should* be web-editable — two of them change where data
goes, which is exactly the argument for leaving them out.

### Agent / LLM (`AGENT_LAYER_PLAN.md`)

| Key | Type | Default | Options | Proposed control |
|---|---|---|---|---|
| `llm_enabled` | bool | false | — | Switch **with a confirmation dialog**. This is the one setting that changes where data goes: on, transcript excerpts leave the box for the provider API. Arguably `.env`-only — a deployment decision, not a threshold |
| `llm_provider` | str | `anthropic` | `anthropic`, `nvidia` | `<Select>`. Should be a `Literal` in config so a typo fails at validation, not at request time (`agent.py:401` currently raises `LLMUnavailable` at runtime) |
| `llm_model` | str | `""` | free text; empty = provider default | Combobox: presets + free entry. Defaults are `claude-opus-5` (anthropic) and the NVIDIA model in `DEFAULT_MODELS`. Empty on purpose so switching provider doesn't keep pointing at the other one's model id |
| `anthropic_api_key` | str | `""` | — | Password input, write-only, never returned by GET. Show `sk-…abcd` masked + "replace" button |
| `nvidia_api_key` | str | `""` | — | Same |
| `nvidia_base_url` | str | `https://integrate.api.nvidia.com/v1` | any OpenAI-compatible URL | Text input, URL-validated. This is what lets a local vLLM/Ollama serve the same path |
| `llm_max_iterations` | int | 12 | 1–50 | number, step 1. Bounds a runaway tool loop |
| `llm_max_tokens` | int | 4096 | 256–32000 | number, step 256. **OpenAI path only** — see Gaps |
| `llm_effort` | str | `high` | `low`, `medium`, `high` | `<Select>`. Anthropic path only (`output_config.effort`) |

### Graph (`GRAPH_RAG_PLAN.md`)

| Key | Type | Default | Proposed control |
|---|---|---|---|
| `graph_enabled` | bool | false | Read-only status row + a "Sync now" button. Neo4j holds a *derived* read model, so the honest UI is health + last-sync + re-project, not a toggle |
| `graph_uri` | str | `bolt://localhost:7687` | `.env` only — connection wiring |
| `graph_user` | str | `neo4j` | `.env` only |
| `graph_password` | str | `change-me` | `.env` only. Note the `GRAPH_*` prefix is deliberate: the neo4j container reads every `NEO4J_*` var as a server config setting and refuses to boot on an unrecognised one |

### Non-config settings the page should probably grow

Not fields in `Settings` — new work, listed so the scope is visible:

- **Data management** — total disk under `DATA_DIR`, per-directory breakdown (`raw`, `work`, `exports`, `montages`), and a purge button. `retention_days` needs this to mean anything.
- **API key rotation** — generate/revoke, since the web UI already stores one in `localStorage`.
- **Calibration** — `POST /v1/admin/calibrate` has no button. It's the *recommended* way to set the three identification thresholds and it's currently curl-only.
- **Export defaults** — format and redaction defaults for the export path.

---

## Gaps and bugs

1. **`retention_days` is dead.** Exposed under "Jobs & retention", stored, never read by any
   code path (`grep 'cfg.retention_days'` → nothing). Either wire it to a purge job or drop it
   from `SETTINGS_CATEGORIES` — a knob that silently does nothing is worse than no knob.
2. **`llm_max_tokens` is ignored on the Anthropic path.** `agent.py:457` hardcodes
   `max_tokens=16000`; the setting only reaches the OpenAI-compatible call at line 515.
3. **No range validation anywhere.** `patch_settings()` validates type only. `diar_min_speakers`
   > `diar_max_speakers`, a negative `highpass_hz`, or `quality_fair_snr_db` above
   `quality_good_snr_db` all save cleanly and misbehave at job time. Pydantic `Field(ge=, le=)`
   on the numeric fields fixes this at both API and (via the schema) the UI.
4. **`asr_language` is free text.** Same class of typo bug that `device` was made a `Literal` to
   prevent, and it fails silently rather than loudly.
5. **`sentiment_model` is missing from `RESTART_REQUIRED_FIELDS`**, so the System tab lists eight
   of the nine baked-in model fields.
6. **The float input step is a flat `0.01`.** Fine for thresholds, wrong for `target_lufs`
   (0.5), `merge_gap_s` (0.1) and `quality_min_bandwidth_hz` (100). Per-field `step` would come
   free if the API returned it alongside `type`.
