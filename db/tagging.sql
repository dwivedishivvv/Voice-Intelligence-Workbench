-- Automatic clip tags and the pipeline stage that writes them (worker/worker/stages/tag.py).
--
-- Kept separate and idempotent because init.sql only ever runs on
-- a fresh postgres volume. Unlike the older replayable files this one never needs a manual
-- psql replay -- the api applies every replayable file at startup (common/schema.py).

ALTER TYPE clip_status ADD VALUE IF NOT EXISTS 'TAGGING' AFTER 'INDEXING';

-- Machine tags live apart from clips.tags, which belongs to the operator. A reprocess
-- replaces these wholesale; it must never touch something a person typed. NULL means "not
-- tagged yet" (the backfill looks for exactly that), an empty array means "tagged, nothing
-- applied".
ALTER TABLE clips ADD COLUMN IF NOT EXISTS auto_tags TEXT[];
ALTER TABLE clips ADD COLUMN IF NOT EXISTS tagged_at TIMESTAMPTZ;

-- Tag filters are `tag = ANY(...)` membership tests, which a GIN index serves directly.
CREATE INDEX IF NOT EXISTS clips_tags_gin ON clips USING GIN (tags);
CREATE INDEX IF NOT EXISTS clips_auto_tags_gin ON clips USING GIN (auto_tags);
