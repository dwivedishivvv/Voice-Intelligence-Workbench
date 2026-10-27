"""Re-cluster pyannote's turns with the pipeline's own speaker embedder.

pyannote 3.1 does two jobs: segmentation (where speech is and where a speaker change
happens) and clustering (which turns are the same person). Its segmentation is good. Its
clustering runs on its internal ResNet34 embeddings with a centroid-linkage threshold and a
minimum cluster size, and on short multi-speaker recordings it got the number of speakers
wrong about half the time on a synthetic benchmark of known conversations — merging two
people into one label as often as it split one person in two. Lowering min_cluster_size
only traded merges for splits (measured worse on both clean and degraded audio).

The pipeline already holds a much stronger embedder (ResNet293, ~3x fewer cross-session
errors than ResNet34's class of model on degraded audio — worker/speaker_models.py). So
pyannote's turn boundaries are kept and the "who is who" question is answered again:

  1. every usable turn (not overlapped, not too short) is cut into pieces of at most
     PIECE_S, each embedded — equal-length pieces batch exactly (speaker_models.py)
  2. average-linkage agglomerative clustering on cosine, stopping at `threshold`
  3. clusters holding less than MIN_CLUSTER_S of speech are folded into their nearest
     neighbour: a few seconds is not enough evidence to assert an extra person
  4. each turn takes the cluster holding most of its pieces' duration; turns too short to
     cut get the nearest centroid if there is enough audio to embed, else keep the cluster
     their original pyannote label mostly went to

The measured effect is in config.py (diar_recluster_threshold).
"""
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from ..ctx import SR
from .embed import encode_many

PIECE_S = 2.5
MIN_PIECE_S = 1.0
MIN_CLUSTER_S = 3.0
MIN_EMBED_S = 0.5


def pieces(start: float, end: float) -> list[tuple[float, float]]:
    """Cut [start, end) into pieces of PIECE_S, the last absorbing any remainder shorter
    than MIN_PIECE_S so no piece is too short to embed."""
    out, s = [], start
    while end - s >= MIN_PIECE_S:
        e = min(end, s + PIECE_S)
        if end - e < MIN_PIECE_S:
            e = end
        out.append((s, e))
        s = e
    return out


def _centroids(E: np.ndarray, lab: np.ndarray) -> dict[int, np.ndarray]:
    out = {}
    for c in set(lab.tolist()):
        v = E[lab == c].mean(0)
        out[c] = v / max(float(np.linalg.norm(v)), 1e-6)
    return out


def fold_small(E: np.ndarray, lab: np.ndarray, durs: np.ndarray, min_s: float) -> np.ndarray:
    """Merge clusters with under `min_s` of speech into their nearest cluster, smallest
    first, until none is left or only one cluster remains."""
    lab = lab.copy()
    while True:
        tot = {c: float(durs[lab == c].sum()) for c in set(lab.tolist())}
        small = [c for c, s in tot.items() if s < min_s]
        if not small or len(tot) == 1:
            return lab
        C = _centroids(E, lab)
        c = min(small, key=tot.get)
        nearest = max((k for k in C if k != c), key=lambda k: float(C[c] @ C[k]))
        lab[lab == c] = nearest


def recluster(turns: list[dict], audio: np.ndarray, pool, threshold: float) -> list[dict]:
    """Relabel `turns` in place (and return them). Leaves them untouched when there is too
    little usable speech to cluster at all."""
    usable = [i for i, t in enumerate(turns) if not t.get("too_short") and not t.get("is_overlap")]
    owners, segs, durs = [], [], []
    for i in usable:
        for s, e in pieces(turns[i]["start"], turns[i]["end"]):
            owners.append(i)
            segs.append(audio[int(s * SR):int(e * SR)])
            durs.append(e - s)
    if len(segs) < 2:
        return turns

    E = encode_many(pool, segs)
    lab = fcluster(linkage(E, method="average", metric="cosine"),
                   t=1.0 - threshold, criterion="distance")
    lab = fold_small(E, np.asarray(lab), np.asarray(durs), MIN_CLUSTER_S)
    C = _centroids(E, lab)

    votes: dict[int, dict[int, float]] = {}
    for o, c, d in zip(owners, lab.tolist(), durs):
        votes.setdefault(o, {}).setdefault(c, 0.0)
        votes[o][c] += d
    # what each original pyannote label mostly became, for turns with nothing to embed
    by_orig: dict[str, dict[int, float]] = {}
    for i, v in votes.items():
        for c, d in v.items():
            by_orig.setdefault(turns[i]["label"], {}).setdefault(c, 0.0)
            by_orig[turns[i]["label"]][c] += d

    new_label = {}
    for i, v in votes.items():
        new_label[i] = max(v, key=v.get)
    for i, t in enumerate(turns):
        if i in new_label:
            continue
        seg = audio[int(t["start"] * SR):int(t["end"] * SR)]
        if seg.size >= int(MIN_EMBED_S * SR):
            e = encode_many(pool, [seg])[0]
            new_label[i] = max(C, key=lambda k: float(e @ C[k]))
        elif t["label"] in by_orig:
            new_label[i] = max(by_orig[t["label"]], key=by_orig[t["label"]].get)
    # renumber in order of first appearance, in pyannote's own label format
    order: dict[int, str] = {}
    for i in sorted(new_label, key=lambda i: turns[i]["start"]):
        order.setdefault(new_label[i], f"SPEAKER_{len(order):02d}")
    for i, c in new_label.items():
        turns[i]["label"] = order[c]
    return turns
