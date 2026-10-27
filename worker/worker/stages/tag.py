"""Automatic clip tags: the signal tags from common/tagging.py, plus topic tags.

Topic tags are keyphrases from the transcript, ranked KeyBERT-style: candidate n-grams are
embedded with the same text model the search index uses, scored against the clip's own
utterance embeddings, then picked with MMR so three tags are three different things rather
than three spellings of one. They live here rather than in common/ because they need the
encoder, which only the worker loads.

Inputs are read back from Postgres rather than taken from the pipeline ctx, so the
maintenance backfill can tag clips processed before this stage existed without re-running
their audio through the pipeline.
"""
import asyncio
import re
from collections import Counter

import numpy as np

from common import db, tagging

TOPIC_MIN_WORDS = 20       # below this a transcript has no topic worth naming
TOPIC_CANDIDATES = 40      # phrases embedded per clip; caps encoder work on long recordings
TOPIC_DIVERSITY = 0.5      # MMR weight on "different from what is already picked"

# Function words plus the filler that dominates transcribed speech. Small on purpose: a
# phrase only has to not *start or end* with one of these, so "point of view" survives.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are aren't as at be because been
before being below between both but by can can't cannot could couldn't did didn't do does
doesn't doing don't down during each few for from further get gets getting go goes going
gonna got gotta had hadn't has hasn't have haven't having he he'd he'll he's her here here's
hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its
itself just know let let's like look me might more most much must mustn't my myself need no
nor not now of off ok okay on once one only or other ought our ours ourselves out over own
really right said same say says see shan't she she'd she'll she's should shouldn't so some
such sure than that that's the their theirs them themselves then there there's these they
they'd they'll they're they've thing things think this those though through to too two under
until up us very want was wasn't way we we'd we'll we're we've well were weren't what what's
when when's where where's which while who who's whom why why's will with won't would wouldn't
yeah yes yep you you'd you'll you're you've your yours yourself yourselves um uh hmm mm oh ah
actually basically wanna kind sort lot lots bit stuff anyway maybe probably
good great nice fine bad better best new old next last first latest little big long short
today tomorrow yesterday morning afternoon evening everyone everybody someone something
thanks thank please hello hi bye sorry talk talking call start started done close mostly
slightly clearly already still many quite pretty even ever never always again around
second seconds minute minutes hour hours day days week weeks month months year years time
""".split())

# Relevance credit per extra word in a phrase. Single words score close to a clip's mean
# embedding almost by construction (they are what it is averaged from), so without a small
# nudge "pipeline" always beats "pipeline rewrite", which says more.
PHRASE_BONUS = 0.04

# The only stopwords allowed *inside* a three-word phrase: connectors that hold a noun phrase
# together ("point of view", "speech to text"). A verb in the middle ("recording is
# estimated") makes it a clause fragment, not a topic.
INNER_OK = frozenset({"of", "and", "to", "for", "in", "on", "vs"})

_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return [t.strip("'") for t in _WORD.findall(text.lower()) if t.strip("'")]


def candidate_phrases(texts: list[str], limit: int = TOPIC_CANDIDATES) -> list[str]:
    """1-3 word phrases worth embedding, most frequent first.

    A phrase may not start or end on a stopword or a bare number, and a single word needs
    four letters — "car" is rarely a topic, "tyres" usually is. On a long transcript a
    phrase has to recur: a one-off trigram in an hour of speech is noise, not a theme.
    """
    counts: Counter[str] = Counter()
    n_words = 0
    for text in texts:
        toks = _tokens(text)
        n_words += len(toks)
        for n in (1, 2, 3):
            for i in range(len(toks) - n + 1):
                g = toks[i:i + n]
                if g[0] in STOPWORDS or g[-1] in STOPWORDS:
                    continue
                if n == 3 and g[1] in STOPWORDS and g[1] not in INNER_OK:
                    continue
                if g[0].isdigit() or g[-1].isdigit():
                    continue
                if n == 1 and len(g[0]) < 4:
                    continue
                counts[" ".join(g)] += 1
    min_count = 2 if n_words >= 150 else 1
    ranked = [(p, c) for p, c in counts.items() if c >= min_count]
    # frequency first, then longer phrases — "tyre temperatures" over "tyre" at equal count
    ranked.sort(key=lambda pc: (pc[1], len(pc[0].split())), reverse=True)
    return [p for p, _ in ranked[:limit]]


def pick_topics(doc_vec: np.ndarray, cand_vecs: np.ndarray, phrases: list[str], k: int,
                min_sim: float, diversity: float = TOPIC_DIVERSITY) -> list[str]:
    """Maximal-marginal-relevance selection over unit vectors.

    Each pick maximises similarity to the clip minus similarity to what is already picked,
    so the second tag has to add something. A phrase whose words are a subset (or a
    superset) of an already-picked phrase is skipped outright — "tyre" next to "tyre wear"
    is the same tag twice however the vectors fall.
    """
    if not phrases:
        return []
    rel = cand_vecs @ doc_vec
    chosen: list[int] = []
    while len(chosen) < k:
        best, best_score = None, -np.inf
        for i, p in enumerate(phrases):
            if i in chosen or rel[i] < min_sim:
                continue
            words = set(p.split())
            if any(words <= set(phrases[j].split()) or set(phrases[j].split()) <= words
                   for j in chosen):
                continue
            redundancy = max((float(cand_vecs[i] @ cand_vecs[j]) for j in chosen), default=0.0)
            # the floor above is on raw similarity; the bonus only orders what cleared it
            relevance = float(rel[i]) + PHRASE_BONUS * (len(words) - 1)
            score = (1 - diversity) * relevance - diversity * redundancy
            if score > best_score:
                best, best_score = i, score
        if best is None:
            break
        chosen.append(best)
    return [phrases[i] for i in chosen]


def topic_tags(pool, utterances: list[dict], language: str | None, cfg) -> list[str]:
    # The stopword list and the default text model are English; on other languages the
    # candidates would be function words with nothing filtering them.
    if not (language or "").startswith("en") or cfg.auto_tag_max_topics <= 0:
        return []
    texts = [u["text"] for u in utterances if u.get("text")]
    if sum(len(t.split()) for t in texts) < TOPIC_MIN_WORDS:
        return []
    phrases = candidate_phrases(texts)
    if not phrases:
        return []

    # The clip's meaning as the mean of its utterance embeddings (already computed by the
    # INDEXING stage). Encoding the joined transcript instead would truncate at the text
    # model's 256-token window and describe only the first minute of a long recording.
    stored = [np.asarray(u["embedding"], dtype=np.float32) for u in utterances
              if u.get("embedding") is not None]
    if stored:
        doc = np.mean(np.stack(stored), axis=0)
    else:
        doc = np.asarray(pool.text_embedder.encode([" ".join(texts)], normalize_embeddings=True)[0])
    doc = (doc / max(float(np.linalg.norm(doc)), 1e-6)).astype(np.float32)

    cand = np.asarray(pool.text_embedder.encode(phrases, normalize_embeddings=True, batch_size=64),
                      dtype=np.float32)
    picked = pick_topics(doc, cand, phrases, cfg.auto_tag_max_topics, cfg.auto_tag_min_topic_sim)
    return [f"{tagging.TOPIC_PREFIX}{p}" for p in picked]


async def tag_clip(clip_id: str, pool, cfg) -> list[str] | None:
    """Compute and store auto_tags for one clip from what is in the database."""
    inputs = await tagging.load_inputs(clip_id, with_embeddings=True)
    if inputs is None:
        return None
    clip, quality, speakers, utterances = inputs
    tags = tagging.derive_tags(clip, quality, speakers, utterances, cfg)
    tags += await asyncio.to_thread(topic_tags, pool, utterances, clip.get("language"), cfg)
    await db.execute("UPDATE clips SET auto_tags=$2, tagged_at=now() WHERE id=$1", clip_id, tags)
    return tags


async def run(ctx):
    if not ctx.cfg.auto_tag_enabled:
        return
    try:
        await tag_clip(ctx.clip_id, ctx.pool, ctx.cfg)
    except Exception as e:
        # Tags are an enrichment over a finished analysis. Failing the clip over them would
        # throw away a good transcript; leaving auto_tags NULL lets the backfill retry later.
        ctx.warn("AUTO_TAG_FAILED", error=str(e)[:200])
