from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from common.speaker import reliability_score
from ..ctx import SR, SpeakerResult


def extract(audio_norm, turns, cfg, pool, min_s=None):
    # min_s overrides cfg.embed_min_s — live mode passes a lower floor (see live.py) since
    # its turns are naturally shorter than a whole clip's pooled best segments; reliability_score
    # already penalizes a short embedding on its own, so this only changes whether an attempt
    # is made at all, not how much weight a short embedding gets once identify() sees it
    floor = cfg.embed_min_s if min_s is None else min_s
    usable = sorted([t for t in turns if not t["is_overlap"]],
                     key=lambda t: t["end"] - t["start"], reverse=True)
    chunks, total = [], 0.0
    for t in usable:
        a, b = int(t["start"] * SR), int(t["end"] * SR)
        chunks.append(audio_norm[a:b])
        total += (b - a) / SR
        if total >= cfg.embed_target_s:
            break

    if total < floor:
        return None, total, len(chunks)

    x = torch.from_numpy(np.concatenate(chunks)).unsqueeze(0)
    with torch.no_grad():
        emb = pool.embedder.encode_batch(x).squeeze()
    emb = F.normalize(emb, dim=0).cpu().numpy().astype(np.float32)
    return emb, total, len(chunks)


async def run(ctx):
    by_label = defaultdict(list)
    for t in ctx.turns:
        by_label[t["label"]].append(t)

    for label, turns in by_label.items():
        emb, total, n_seg = extract(ctx.audio_norm, turns, ctx.cfg, ctx.pool)
        rel = reliability_score(total, ctx.quality, ctx.cfg) if emb is not None else 0.0
        ctx.speakers[label] = SpeakerResult(
            local_label=label, embedding=emb, speech_s=total, n_turns=len(turns),
            n_segments_used=n_seg, reliability=rel,
            reliability_reason=None if emb is not None else "insufficient_speech")
        if emb is None:
            ctx.warn("SPEAKER_TOO_SHORT", label=label, duration_s=total)
        elif rel < ctx.cfg.reliability_fair:
            ctx.warn("LOW_RELIABILITY_EMBEDDING", label=label, reliability=rel)
