-- Speaker Intelligence Workbench schema (spec §5, trimmed to what the code uses)
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TYPE clip_status AS ENUM (
  'QUEUED','VALIDATING','PREPROCESSING','TRANSCRIBING','DIARIZING',
  'RECONCILING','EMBEDDING','IDENTIFYING','POSTPROCESSING','SENTIMENT','INDEXING',
  'COMPLETE','FAILED','REJECTED','DEAD'
);
CREATE TYPE quality_grade AS ENUM ('good','fair','poor');
CREATE TYPE profile_status AS ENUM ('provisional','confirmed','locked');
CREATE TYPE enroll_source AS ENUM ('manual','auto_confirmed','imported','promoted');
CREATE TYPE id_result AS ENUM ('confident','suggested','unknown','abstained');

CREATE TABLE clips (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  sha256          CHAR(64) NOT NULL,
  filename        TEXT NOT NULL,
  mime            TEXT,
  duration_s      REAL,
  sample_rate     INT,
  channels        INT,
  size_bytes      BIGINT,
  raw_path        TEXT NOT NULL,
  work_path       TEXT,
  status          clip_status NOT NULL DEFAULT 'QUEUED',
  error_code      TEXT,
  error_detail    TEXT,
  language        TEXT,
  language_conf   REAL,
  n_speakers      INT,
  needs_review    BOOLEAN NOT NULL DEFAULT false,
  tags            TEXT[] DEFAULT '{}',
  notes           TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX clips_sha_uniq ON clips(sha256);
CREATE INDEX clips_status_created ON clips(status, created_at DESC);
CREATE INDEX clips_review ON clips(needs_review) WHERE needs_review;

CREATE TABLE quality_metrics (
  clip_id           UUID PRIMARY KEY REFERENCES clips(id) ON DELETE CASCADE,
  snr_db            REAL,
  clipping_ratio    REAL,
  silence_ratio     REAL,
  speech_duration_s REAL,
  bandwidth_hz      REAL,
  dc_offset         REAL,
  input_lufs        REAL,
  grade             quality_grade
);

CREATE TABLE vad_regions (
  id       BIGSERIAL PRIMARY KEY,
  clip_id  UUID NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
  idx      INT NOT NULL,
  start_s  REAL NOT NULL,
  end_s    REAL NOT NULL
);
CREATE INDEX vad_clip ON vad_regions(clip_id, idx);

CREATE TABLE speaker_turns (
  id             BIGSERIAL PRIMARY KEY,
  clip_id        UUID NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
  idx            INT NOT NULL,
  local_label    TEXT NOT NULL,
  start_s        REAL NOT NULL,
  end_s          REAL NOT NULL,
  is_overlap     BOOLEAN NOT NULL DEFAULT false,
  snapped        BOOLEAN NOT NULL DEFAULT false
);
CREATE INDEX turns_clip ON speaker_turns(clip_id, idx);

CREATE TABLE transcripts (
  id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  clip_id        UUID NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
  run_id         UUID,
  engine         TEXT NOT NULL,
  language       TEXT,
  language_conf  REAL,
  text           TEXT NOT NULL,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX transcripts_clip ON transcripts(clip_id, created_at DESC);

CREATE TABLE utterances (
  id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  transcript_id     UUID NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
  clip_id           UUID NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
  idx               INT NOT NULL,
  start_s           REAL NOT NULL,
  end_s             REAL NOT NULL,
  local_label       TEXT,
  profile_id        UUID,
  cluster_id        UUID,
  text              TEXT NOT NULL,
  mean_word_conf    REAL,
  mean_speaker_conf REAL,
  embedding         VECTOR(384),
  tsv               TSVECTOR
);
CREATE INDEX utt_clip ON utterances(clip_id, idx);
CREATE INDEX utt_profile ON utterances(profile_id);
CREATE INDEX utt_tsv ON utterances USING GIN(tsv);
CREATE INDEX utt_emb ON utterances USING hnsw (embedding vector_cosine_ops);

CREATE FUNCTION utt_tsv_update() RETURNS trigger AS $$
BEGIN
  NEW.tsv := to_tsvector('simple', coalesce(NEW.text,''));
  RETURN NEW;
END $$ LANGUAGE plpgsql;
CREATE TRIGGER utt_tsv_trg BEFORE INSERT OR UPDATE OF text ON utterances
  FOR EACH ROW EXECUTE FUNCTION utt_tsv_update();

CREATE TABLE words (
  id            BIGSERIAL PRIMARY KEY,
  utterance_id  UUID NOT NULL REFERENCES utterances(id) ON DELETE CASCADE,
  clip_id       UUID NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
  idx           INT NOT NULL,
  word          TEXT NOT NULL,
  start_s       REAL NOT NULL,
  end_s         REAL NOT NULL,
  probability   REAL,
  local_label   TEXT,
  speaker_conf  REAL,
  smoothed      BOOLEAN NOT NULL DEFAULT false
);
CREATE INDEX words_utt ON words(utterance_id, idx);
CREATE INDEX words_clip_time ON words(clip_id, start_s);

CREATE TABLE speaker_profiles (
  id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  display_name     TEXT NOT NULL,
  notes            TEXT,
  tags             TEXT[] DEFAULT '{}',
  centroid         VECTOR(192),
  n_enrollments    INT NOT NULL DEFAULT 0,
  total_speech_s   REAL NOT NULL DEFAULT 0,
  mean_reliability REAL,
  intra_cohesion   REAL,
  status           profile_status NOT NULL DEFAULT 'provisional',
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX profiles_name ON speaker_profiles(lower(display_name));
CREATE INDEX profiles_centroid ON speaker_profiles USING hnsw (centroid vector_cosine_ops);

CREATE TABLE speaker_enrollments (
  id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  profile_id   UUID NOT NULL REFERENCES speaker_profiles(id) ON DELETE CASCADE,
  clip_id      UUID REFERENCES clips(id) ON DELETE SET NULL,
  local_label  TEXT,
  embedding    VECTOR(192) NOT NULL,
  duration_s   REAL NOT NULL,
  n_segments   INT,
  reliability  REAL NOT NULL,
  quality      quality_grade,
  source       enroll_source NOT NULL,
  is_outlier   BOOLEAN NOT NULL DEFAULT false,
  cos_to_centroid REAL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX enroll_profile ON speaker_enrollments(profile_id, is_outlier);

CREATE TABLE speaker_clusters (
  id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  label          TEXT NOT NULL,
  centroid       VECTOR(192),
  n_members      INT NOT NULL DEFAULT 0,
  total_speech_s REAL NOT NULL DEFAULT 0,
  intra_cohesion REAL,
  promoted_to    UUID REFERENCES speaker_profiles(id),
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX clusters_centroid ON speaker_clusters USING hnsw (centroid vector_cosine_ops);

CREATE TABLE clip_speakers (
  id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  clip_id        UUID NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
  local_label    TEXT NOT NULL,
  embedding      VECTOR(192),
  speech_s       REAL NOT NULL,
  n_turns        INT NOT NULL,
  n_segments_used INT,
  reliability    REAL,
  reliability_reason TEXT,
  profile_id     UUID REFERENCES speaker_profiles(id) ON DELETE SET NULL,
  cluster_id     UUID REFERENCES speaker_clusters(id) ON DELETE SET NULL,
  match_score    REAL,
  match_margin   REAL,
  match_result   id_result NOT NULL DEFAULT 'unknown',
  runner_up_id   UUID,
  runner_up_score REAL,
  is_manual      BOOLEAN NOT NULL DEFAULT false,
  talk_share     REAL,
  longest_turn_s REAL,
  interruptions  INT,
  overlap_s      REAL,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX clipspk_uniq ON clip_speakers(clip_id, local_label);
CREATE INDEX clipspk_profile ON clip_speakers(profile_id);
CREATE INDEX clipspk_cluster ON clip_speakers(cluster_id);
CREATE INDEX clipspk_emb ON clip_speakers USING hnsw (embedding vector_cosine_ops);

CREATE TABLE processing_runs (
  id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  clip_id          UUID NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
  attempt          INT NOT NULL DEFAULT 1,
  started_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at      TIMESTAMPTZ,
  total_ms         INT,
  outcome          TEXT,
  pipeline_version TEXT NOT NULL,
  config_snapshot  JSONB NOT NULL,
  model_versions   JSONB NOT NULL,
  stage_timings    JSONB,
  device           TEXT,
  warnings         JSONB DEFAULT '[]',
  error_code       TEXT,
  error_detail     TEXT,
  corr_id          TEXT
);
CREATE INDEX runs_clip ON processing_runs(clip_id, started_at DESC);

CREATE TABLE audit_log (
  id          BIGSERIAL PRIMARY KEY,
  ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
  actor_id    TEXT,
  action      TEXT NOT NULL,
  object_type TEXT NOT NULL,
  object_id   TEXT,
  before      JSONB,
  after       JSONB,
  corr_id     TEXT,
  prev_hash   CHAR(64),
  row_hash    CHAR(64)
);
CREATE INDEX audit_ts ON audit_log(ts DESC);
CREATE INDEX audit_object ON audit_log(object_type, object_id);
REVOKE UPDATE, DELETE ON audit_log FROM PUBLIC;

CREATE TABLE calibration_runs (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  ran_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
  n_profiles      INT,
  n_genuine       INT,
  n_impostor      INT,
  eer             REAL,
  eer_threshold   REAL,
  genuine_mean    REAL,
  impostor_mean   REAL,
  separation      REAL,
  suggested_id_threshold     REAL,
  suggested_verify_threshold REAL,
  histogram       JSONB
);

CREATE TABLE deletion_receipts (
  id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  object_type TEXT NOT NULL,
  object_id   TEXT NOT NULL,
  sha256      CHAR(64),
  deleted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  reason      TEXT
);

-- Live-tunable pipeline settings (thresholds etc.) — overrides Settings' env defaults.
-- Model/device fields are intentionally not stored here: ModelPool loads them once at
-- worker startup, so changing them requires a restart and belongs in .env, not this table.
CREATE TABLE settings_overrides (
  key         TEXT PRIMARY KEY,
  value       TEXT NOT NULL,
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_by  TEXT
);

-- Tracks alternative models pulled live via the Settings > Models page (separate from the
-- ones baked into the image at build time via models/prefetch.py).
CREATE TABLE downloaded_models (
  category      TEXT NOT NULL,
  repo_id       TEXT NOT NULL,
  dir_name      TEXT NOT NULL,
  status        TEXT NOT NULL DEFAULT 'pending',
  error         TEXT,
  downloaded_at TIMESTAMPTZ,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (category, repo_id)
);

-- Races + sentiment columns. Kept separate and idempotent so it can also be replayed
-- against an already-initialized database (init.sql itself only runs on a fresh volume).
-- Absolute: init.sql is mounted into initdb.d, its siblings into /db (see docker-compose).
\i /db/races.sql

-- OpenF1 sessions/drivers/laps + radio-call results. Same reasoning as races.sql, and
-- ordered after it because f1.sql alters clips, which races.sql also touches.
\i /db/f1.sql

-- On-track events read out of speech (common/ontrack.py). Same reasoning as races.sql,
-- and ordered after f1.sql because it alters radio_calls, which f1.sql creates.
\i /db/ontrack.sql

-- Diarization/fingerprinting quality columns. Same reasoning as races.sql; only alters
-- tables init.sql itself creates, so its position among the includes does not matter.
\i /db/speaker_quality.sql
