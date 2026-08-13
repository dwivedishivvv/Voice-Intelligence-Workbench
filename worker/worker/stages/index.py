from common import db


async def run(ctx):
    if not ctx.utterances:
        return
    texts = [u["text"] for u in ctx.utterances]
    embs = ctx.pool.text_embedder.encode(texts, normalize_embeddings=True, batch_size=32)
    for u, e in zip(ctx.utterances, embs):
        await db.execute("UPDATE utterances SET embedding=$2 WHERE id=$1", u["id"], list(map(float, e)))
