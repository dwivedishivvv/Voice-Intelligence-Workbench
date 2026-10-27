-- Retires the F1 layer (OpenF1 sessions, drivers, laps, team radio, on-track events,
-- races) and per-utterance sentiment/tone analysis. The workbench is a general speech
-- corpus tool; both features were domain-specific and have been removed end to end.
--
-- Idempotent, so the api can replay it at startup like every other file in this directory
-- (common/schema.py). On a fresh volume it is a no-op: init.sql no longer creates any of
-- this. The SENTIMENT value stays in clip_status because Postgres cannot drop an enum
-- value; nothing sets it any more.

DROP TABLE IF EXISTS speech_events CASCADE;
DROP TABLE IF EXISTS radio_calls   CASCADE;
DROP TABLE IF EXISTS f1_aliases    CASCADE;
DROP TABLE IF EXISTS f1_laps       CASCADE;
DROP TABLE IF EXISTS f1_drivers    CASCADE;
DROP TABLE IF EXISTS f1_sessions   CASCADE;

ALTER TABLE clips DROP COLUMN IF EXISTS race_id;
DROP TABLE IF EXISTS races CASCADE;

ALTER TABLE clips DROP COLUMN IF EXISTS recorded_at;
ALTER TABLE clips DROP COLUMN IF EXISTS sentiment;
ALTER TABLE clips DROP COLUMN IF EXISTS sentiment_score;
ALTER TABLE clips DROP COLUMN IF EXISTS mood;

ALTER TABLE utterances DROP COLUMN IF EXISTS sentiment;
ALTER TABLE utterances DROP COLUMN IF EXISTS sentiment_score;
ALTER TABLE utterances DROP COLUMN IF EXISTS text_sentiment;
ALTER TABLE utterances DROP COLUMN IF EXISTS text_score;
ALTER TABLE utterances DROP COLUMN IF EXISTS mood;
ALTER TABLE utterances DROP COLUMN IF EXISTS mood_features;

-- Auto tags derived from the removed signals would otherwise linger on clips until each
-- one is reprocessed.
UPDATE clips
   SET auto_tags = ARRAY(SELECT t FROM unnest(auto_tags) t
                          WHERE t NOT LIKE 'tone:%' AND t NOT LIKE 'sentiment:%'
                            AND t <> 'mixed-signals')
 WHERE auto_tags && ARRAY['tone:stressed','tone:tired','sentiment:negative',
                          'sentiment:positive','mixed-signals'];
