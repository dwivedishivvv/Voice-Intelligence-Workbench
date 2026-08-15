"""Curated alternatives per model category, offered on the Settings > Models page. Every
repo_id here was verified to actually exist on the Hub before being added — see the
worker/models/REGISTRY.yaml comment history for what happens when that step gets skipped.

Selecting a model here downloads it (if not already present) under MODEL_DIR/<category>/
<dir_name>/, alongside the one baked in at build time. Activating a downloaded model updates
the corresponding *_model Settings field, which ModelPool only reads at worker startup — so
activation always requires a restart, deliberately: hot-swapping a loaded torch model without
restarting is a much larger change than this warrants."""

CATALOG = {
    "asr": {
        "settings_field": "asr_model",
        "gated": False,
        "options": [
            {"repo_id": "deepdml/faster-whisper-large-v3-turbo-ct2", "dir_name": "large-v3-turbo",
             "label": "Large v3 Turbo", "note": "Default. Best accuracy, ~1.6GB."},
            {"repo_id": "Systran/faster-whisper-large-v3", "dir_name": "large-v3",
             "label": "Large v3", "note": "Slightly slower than Turbo, marginally more accurate on some languages."},
            {"repo_id": "Systran/faster-whisper-medium", "dir_name": "medium",
             "label": "Medium", "note": "~1.5GB, faster than Large, noticeably less accurate."},
            {"repo_id": "Systran/faster-whisper-small", "dir_name": "small",
             "label": "Small", "note": "~500MB, fast, good for quick iteration."},
            {"repo_id": "Systran/faster-whisper-base", "dir_name": "base",
             "label": "Base", "note": "~150MB, fastest, English-leaning accuracy."},
            {"repo_id": "Systran/faster-whisper-tiny", "dir_name": "tiny",
             "label": "Tiny", "note": "~75MB, near-instant, low accuracy — testing only."},
        ],
    },
    "diarization": {
        "settings_field": "diar_model",
        "gated": True,
        "options": [
            {"repo_id": "pyannote/speaker-diarization-3.1", "dir_name": "pyannote-3.1",
             "label": "pyannote 3.1", "note": "Default. Current-generation pipeline.",
             "depends_on": ["pyannote/segmentation-3.0", "pyannote/wespeaker-voxceleb-resnet34-LM"]},
            {"repo_id": "pyannote/speaker-diarization-3.0", "dir_name": "pyannote-3.0",
             "label": "pyannote 3.0", "note": "Previous generation — kept for comparison/rollback.",
             "depends_on": ["pyannote/segmentation-3.0", "pyannote/wespeaker-voxceleb-resnet34-LM"]},
        ],
    },
    "embedding": {
        "settings_field": "embed_model",
        "gated": False,
        "options": [
            {"repo_id": "speechbrain/spkrec-ecapa-voxceleb", "dir_name": "ecapa",
             "label": "ECAPA-TDNN", "note": "Default. Strong general-purpose speaker embeddings.",
             "depends_on": ["speechbrain/spkrec-ecapa-voxceleb"]},
            {"repo_id": "speechbrain/spkrec-xvect-voxceleb", "dir_name": "xvect",
             "label": "X-Vector", "note": "Older architecture, smaller, less discriminative.",
             "depends_on": ["speechbrain/spkrec-xvect-voxceleb"]},
        ],
    },
    "text_embedding": {
        "settings_field": "text_embed_model",
        "gated": False,
        "options": [
            {"repo_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "dir_name": "minilm",
             "label": "Multilingual MiniLM", "note": "Default. 384-dim, ~50 languages, fast."},
            {"repo_id": "sentence-transformers/all-MiniLM-L6-v2", "dir_name": "minilm-en",
             "label": "MiniLM (English)", "note": "384-dim, smaller and faster, English-only."},
            {"repo_id": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "dir_name": "mpnet-multilingual",
             "label": "Multilingual MPNet", "note": "768-dim, higher quality, ~3x slower than MiniLM."},
        ],
    },
    "sentiment": {
        "settings_field": "sentiment_model",
        "gated": False,
        "options": [
            {"repo_id": "cardiffnlp/twitter-xlm-roberta-base-sentiment", "dir_name": "xlmr-sentiment",
             "label": "XLM-R Sentiment", "note": "Default. Multilingual, 3-class, ~1.1GB."},
            {"repo_id": "cardiffnlp/twitter-roberta-base-sentiment-latest", "dir_name": "roberta-sentiment-en",
             "label": "RoBERTa Sentiment (English)", "note": "~500MB, more accurate on English, English-only."},
            {"repo_id": "tabularisai/multilingual-sentiment-analysis", "dir_name": "distil-sentiment-5class",
             "label": "5-class Multilingual", "note": "Very negative to very positive. Smaller and faster; label set is remapped to 3 classes."},
        ],
    },
}


def find_option(category: str, repo_id: str) -> dict | None:
    cat = CATALOG.get(category)
    if not cat:
        return None
    return next((o for o in cat["options"] if o["repo_id"] == repo_id), None)
