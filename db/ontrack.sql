-- On-track events read out of speech (common/ontrack.py).
--
-- Idempotent and replayable, like races.sql and f1.sql: db/init.sql only runs on a fresh
-- volume, so an existing database needs this applied on its own.
--
-- Postgres is the source of truth. Neo4j is wiped and re-projected by graph_sync, so an
-- event that existed only as a relationship would be destroyed by the next sync.

-- Radio calls skip the INDEXING stage that gives utterances their embedding, so they had
-- no vector to compare against exemplars. Same model, same dimensionality as
-- utterances.embedding — a second model here would put the two halves of the corpus in
-- different spaces and make their scores incomparable.
ALTER TABLE radio_calls ADD COLUMN IF NOT EXISTS embedding VECTOR(384);

CREATE TABLE IF NOT EXISTS speech_events (
  id            BIGSERIAL PRIMARY KEY,
  -- Exactly one of these. Two nullable references rather than a polymorphic id column so
  -- the cascade still works: deleting a clip must take its inferences with it, or the
  -- graph would keep projecting events about speech that no longer exists.
  utterance_id  UUID REFERENCES utterances(id)  ON DELETE CASCADE,
  radio_call_id UUID REFERENCES radio_calls(id) ON DELETE CASCADE,
  CONSTRAINT speech_events_one_source
    CHECK ((utterance_id IS NULL) <> (radio_call_id IS NULL)),

  event_type    TEXT NOT NULL,   -- common/ontrack.py EVENT_TYPES
  family        TEXT NOT NULL,   -- overtaking | proximity | incident | flags
  -- Kept, not just used and discarded: a threshold that turns out to be wrong is
  -- re-evaluated against stored scores instead of re-running the whole corpus, and a
  -- reader deciding whether to trust an edge can see how close it was.
  score         REAL NOT NULL,
  margin        REAL NOT NULL,
  -- The other car, when the line named one. Resolved by the same session-scoped alias
  -- table that produces MENTIONS; NULL means nobody was named, which is common and is not
  -- the same as "no other car was involved".
  target_driver_number INT,
  target_session_key   INT,
  -- Which embedding model produced the score. A model swap invalidates every score in
  -- this table, and silently comparing across models is the failure that would hide it.
  model         TEXT NOT NULL,
  extracted_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per (speech, event type): re-running extraction updates rather than duplicates.
CREATE UNIQUE INDEX IF NOT EXISTS speech_events_utt_uniq
  ON speech_events (utterance_id, event_type) WHERE utterance_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS speech_events_radio_uniq
  ON speech_events (radio_call_id, event_type) WHERE radio_call_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS speech_events_type ON speech_events (event_type);
CREATE INDEX IF NOT EXISTS speech_events_family ON speech_events (family);
