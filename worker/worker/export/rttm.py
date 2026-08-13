from common import storage


def write(cfg, clip_id, turns, speaker_names):
    out = []
    for t in turns:
        name = speaker_names.get(t["label"], t["label"]).replace(" ", "_")
        out.append(f"SPEAKER {clip_id} 1 {t['start']:.3f} {t['end']-t['start']:.3f} "
                   f"<NA> <NA> {name} <NA> <NA>")
    path = storage.export_path(cfg, clip_id, "diarization.rttm")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return path
