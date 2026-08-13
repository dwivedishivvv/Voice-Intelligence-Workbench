from common import storage


def write(cfg, clip_id, utterances):
    lines = [f"{u['local_label']}: {u['text']}" for u in utterances]
    path = storage.export_path(cfg, clip_id, "transcript.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
