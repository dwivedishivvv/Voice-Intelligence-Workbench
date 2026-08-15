-- Races: a user-created grouping over clips, plus the sentiment columns the
-- SENTIMENT pipeline stage writes. Kept in its own file (and written idempotently)
-- because db/init.sql only ever runs on a *fresh* postgres volume — this one can be
-- replayed against a live database. init.sql \ir's it so a fresh volume gets it too.

-- new pipeline stage between POSTPROCESSING and INDEXING (see worker/worker/pipeline.py).
-- clips.status is this enum, so the stage can't be written without it.
ALTER TYPE clip_status ADD VALUE IF NOT EXISTS 'SENTIMENT' BEFORE 'INDEXING';

CREATE TABLE IF NOT EXISTS races (
  id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name         TEXT NOT NULL,
  circuit      TEXT,
  race_date    DATE,
  -- optional OpenF1 link: when set, the race page can pull drivers/lap traces
  session_key  INT,
  -- track outline, display only (see races router: stored under DATA_DIR, served back raw)
  svg_path     TEXT,
  notes        TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS races_created ON races(created_at DESC);

-- A clip belongs to at most one race. ON DELETE SET NULL: deleting a race must not
-- take the audio and its transcripts with it — the clips stay in the library.
ALTER TABLE clips ADD COLUMN IF NOT EXISTS race_id UUID REFERENCES races(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS clips_race ON clips(race_id, created_at) WHERE race_id IS NOT NULL;

-- Per-utterance sentiment. Text model + acoustic tone are stored separately, not just
-- their fusion, so the UI can show *disagreement* (words calm, voice stressed) instead
-- of hiding it behind one averaged number.
ALTER TABLE utterances ADD COLUMN IF NOT EXISTS sentiment       TEXT;   -- fused: negative|neutral|positive
ALTER TABLE utterances ADD COLUMN IF NOT EXISTS sentiment_score REAL;   -- signed -1..1, negative = negative
ALTER TABLE utterances ADD COLUMN IF NOT EXISTS text_sentiment  TEXT;
ALTER TABLE utterances ADD COLUMN IF NOT EXISTS text_score      REAL;
ALTER TABLE utterances ADD COLUMN IF NOT EXISTS mood            TEXT;   -- acoustic: calm|stressed|tired
ALTER TABLE utterances ADD COLUMN IF NOT EXISTS mood_features   JSONB;

-- Clip-level rollup, so the race page can render a card per recording without
-- re-aggregating every utterance on each request.
ALTER TABLE clips ADD COLUMN IF NOT EXISTS sentiment       TEXT;
ALTER TABLE clips ADD COLUMN IF NOT EXISTS sentiment_score REAL;
ALTER TABLE clips ADD COLUMN IF NOT EXISTS mood            TEXT;
