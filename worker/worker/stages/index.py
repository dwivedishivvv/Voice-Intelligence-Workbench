import asyncio

from common import db


async def run(ctx):
    if not ctx.utterances:
        return
    texts = [u["text"] for u in ctx.utterances]
    embs = await asyncio.to_thread(
        ctx.pool.text_embedder.encode, texts, normalize_embeddings=True, batch_size=32)
    # one round trip for the clip rather than one UPDATE per utterance
    await db.pool().executemany(
        "UPDATE utterances SET embedding=$2 WHERE id=$1",
        [(u["id"], list(map(float, e))) for u, e in zip(ctx.utterances, embs)])
