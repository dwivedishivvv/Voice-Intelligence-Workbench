-- Which speaker-embedding model produced the voiceprints currently stored.
--
-- Vectors from two embedding models live in unrelated spaces: a cosine between an ECAPA
-- voiceprint and a ResNet293 one is a number, not a similarity. The column width only
-- catches a change when the dimensions differ; two 256-dim models would mix silently. So
-- the worker stamps the model here and re-embeds everything when the active one differs
-- (worker/worker/reembed.py). Replayable: init.sql only runs on a fresh volume.
CREATE TABLE IF NOT EXISTS embedding_spaces (
  name        TEXT PRIMARY KEY,
  model       TEXT NOT NULL,
  dim         INT NOT NULL,
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
