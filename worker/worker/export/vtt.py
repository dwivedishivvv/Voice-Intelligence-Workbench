from common import storage


def _ts(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    ms = int((s - int(s)) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{ms:03d}"


def write(cfg, clip_id, utterances):
    lines = ["WEBVTT", ""]
    for u in utterances:
        lines.append(f"{_ts(u['start'])} --> {_ts(u['end'])}")
        lines.append(f"{u['local_label']}: {u['text']}")
        lines.append("")
    path = storage.export_path(cfg, clip_id, "transcript.vtt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
