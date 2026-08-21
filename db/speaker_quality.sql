-- Speaker diarization/fingerprinting quality columns.
--
-- Kept separate and idempotent for the same reason as races.sql and f1.sql: init.sql only
-- ever runs on a fresh postgres volume, so anything that has to reach an already-populated
-- database has to be replayable on its own. tests/unit/test_sql_replayable.py enforces it.

-- Weighted mass behind a cluster centroid. speaker_clusters used an unweighted running
-- mean, so one scrap of audio moved the centroid as far as a long clean turn, while
-- speaker_profiles has always been reliability-weighted. Existing rows land on 0 and
-- common/speaker.py::assign_cluster falls back to their recorded seconds, so a centroid
-- built before this column existed keeps its mass instead of being wiped by the first
-- weighted update.
ALTER TABLE speaker_clusters ADD COLUMN IF NOT EXISTS weight_sum REAL NOT NULL DEFAULT 0;

-- Why identification landed where it did, kept so failures can be read back rather than
-- guessed at. match_z is the AS-Norm score (how far the winning similarity sat above this
-- embedding's own impostor cohort); split_half_sim is the agreement between embeddings of
-- two disjoint halves of the same speaker's audio, which is low exactly when a diarization
-- label has more than one voice in it.
ALTER TABLE clip_speakers ADD COLUMN IF NOT EXISTS match_z REAL;
ALTER TABLE clip_speakers ADD COLUMN IF NOT EXISTS split_half_sim REAL;

-- Fraction of an utterance's words that fell inside a region diarization flagged as
-- overlapping speech. The flag was already computed per turn and used to keep overlapped
-- audio out of embeddings, but never reached the transcript, so nothing downstream could
-- tell a clean attribution from one made over cross-talk.
ALTER TABLE utterances ADD COLUMN IF NOT EXISTS overlap_ratio REAL;
