-- OpenF1 race data + radio-call analysis results, persisted rather than proxied.
-- Written idempotently (like races.sql) so it can be replayed against a live database;
-- init.sql \ir's it so a fresh volume gets it too.
--
-- Before this, api/app/routers/f1.py was a pure pass-through and analyze_f1_radio_job
-- emitted its result over a websocket and deleted the audio: nothing about a race or a
-- driver survived the request that fetched it. Everything here exists so that history
-- accumulates instead of evaporating.

-- Wall-clock time the audio was actually recorded, as distinct from clips.created_at
-- (upload time). Any query that lines an utterance up against something else on a
-- timeline -- a lap, another driver's call -- needs this, and cannot derive it from
-- upload time. NULL for manually uploaded clips, where it is genuinely unknown.
ALTER TABLE clips ADD COLUMN IF NOT EXISTS recorded_at TIMESTAMPTZ;

-- OpenF1's own session_key is the natural primary key: it is stable, globally unique,
-- and already the identifier every other OpenF1 endpoint is keyed by, so inventing a
-- surrogate would just mean carrying both.
CREATE TABLE IF NOT EXISTS f1_sessions (
  session_key   INT PRIMARY KEY,
  meeting_key   INT,
  year          INT,
  name          TEXT,
  session_type  TEXT,
  circuit       TEXT,
  country       TEXT,
  location      TEXT,
  date_start    TIMESTAMPTZ,
  synced_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS f1_sessions_year ON f1_sessions(year, date_start DESC);

-- A driver number is only unique within a session (numbers are reassigned between
-- seasons, and reserve drivers appear mid-season), so the PK is the pair.
CREATE TABLE IF NOT EXISTS f1_drivers (
  session_key   INT NOT NULL REFERENCES f1_sessions(session_key) ON DELETE CASCADE,
  driver_number INT NOT NULL,
  full_name     TEXT,
  name_acronym  TEXT,
  team_name     TEXT,
  team_colour   TEXT,
  headshot_url  TEXT,
  -- the seam that ties a driver to a voice we can actually recognize. Set by hand (or
  -- by promoting a cluster) once a profile has been enrolled from that driver's radio.
  profile_id    UUID REFERENCES speaker_profiles(id) ON DELETE SET NULL,
  PRIMARY KEY (session_key, driver_number)
);
CREATE INDEX IF NOT EXISTS f1_drivers_profile ON f1_drivers(profile_id) WHERE profile_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS f1_laps (
  session_key   INT NOT NULL,
  driver_number INT NOT NULL,
  lap_number    INT NOT NULL,
  date_start    TIMESTAMPTZ,
  lap_duration  REAL,
  sector_1      REAL,
  sector_2      REAL,
  sector_3      REAL,
  is_pit_out    BOOLEAN,
  PRIMARY KEY (session_key, driver_number, lap_number)
);
-- The lookup the (future) utterance-to-lap alignment does: given a session and a
-- wall-clock instant, which lap was running.
CREATE INDEX IF NOT EXISTS f1_laps_time ON f1_laps(session_key, date_start);

-- One row per team-radio recording. recording_url is the natural key -- OpenF1 serves
-- each call at a stable livetiming URL -- and the unique index on it is what makes
-- re-analysis a cache hit instead of a re-download.
CREATE TABLE IF NOT EXISTS radio_calls (
  id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  session_key   INT,
  driver_number INT,
  recording_url TEXT NOT NULL,
  recorded_at   TIMESTAMPTZ,
  -- set only if the call is ever put through the full clip pipeline (diarize, identify,
  -- sentiment, index). Null while analysis is the lightweight transcribe+tone path.
  clip_id       UUID REFERENCES clips(id) ON DELETE SET NULL,
  text          TEXT,
  mood          TEXT,
  features      JSONB,
  error         TEXT,
  -- NULL means queued-but-not-finished. Distinguishes "never analyzed" from "analyzed
  -- and genuinely produced no text", which a null `text` alone cannot.
  analyzed_at   TIMESTAMPTZ,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS radio_url_uniq ON radio_calls(recording_url);
CREATE INDEX IF NOT EXISTS radio_session ON radio_calls(session_key, recorded_at);

-- Entity-linking vocabulary for MENTIONS edges. A dictionary, not an NER model: the
-- vocabulary is closed and tiny (~20 drivers, ~10 teams, ~24 circuits a season) and
-- already sits in the tables above, so a matcher over known aliases beats a general NER
-- that has never seen the ways ASR mangles a driver's name.
--
-- The PK is the whole triple, not the alias: a driver can carry a number in one season
-- and a different one in another, so "verstappen" legitimately maps to more than one
-- entity_key. Which one is meant is resolved at match time by the session the speech
-- belongs to, rather than guessed here.
CREATE TABLE IF NOT EXISTS f1_aliases (
  alias       TEXT NOT NULL,        -- lowercased; matched on word boundaries
  entity_type TEXT NOT NULL,        -- driver | team | circuit
  entity_key  TEXT NOT NULL,        -- driver_number, team name, or circuit name
  -- 'auto' rows are derived from f1_drivers/f1_sessions and refreshed on every graph
  -- sync; 'manual' rows are hand-added (ASR mishearings, nicknames) and never touched
  -- by the refresh. Keeping them apart is what makes re-seeding safe.
  source      TEXT NOT NULL DEFAULT 'auto',
  PRIMARY KEY (alias, entity_type, entity_key)
);
CREATE INDEX IF NOT EXISTS f1_aliases_entity ON f1_aliases(entity_type, entity_key);

-- Full-text index over radio transcripts, so a search can anchor on the F1 corpus at all.
-- utterances get their tsvector from a trigger (written before generated columns were
-- available); this one is GENERATED, which needs no trigger and cannot drift from `text`.
-- Radio calls have no *embedding* — the lightweight transcribe+tone path skips the
-- INDEXING stage that produces one — so they are keyword-searchable but not semantically
-- searchable until they go through the full pipeline.
ALTER TABLE radio_calls ADD COLUMN IF NOT EXISTS tsv TSVECTOR
  GENERATED ALWAYS AS (to_tsvector('simple', coalesce(text, ''))) STORED;
CREATE INDEX IF NOT EXISTS radio_tsv ON radio_calls USING GIN(tsv);
