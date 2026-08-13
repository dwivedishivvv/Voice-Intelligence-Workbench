import json
from dataclasses import asdict
from common import storage


def write(cfg, clip_id, ctx):
    data = {
        "clip_id": clip_id, "language": ctx.language, "duration_s": ctx.duration_s,
        "quality": ctx.quality, "warnings": ctx.warnings,
        "speakers": {k: {kk: (list(map(float, vv)) if kk == "embedding" and vv is not None else vv)
                         for kk, vv in asdict(v).items()} for k, v in ctx.speakers.items()},
        "utterances": [{k: v for k, v in u.items() if k != "words"} for u in ctx.utterances],
    }
    path = storage.export_path(cfg, clip_id, "full.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path
