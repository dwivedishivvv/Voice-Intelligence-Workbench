import asyncio, sys, time
from common import db
from common.config import get_settings, Settings
from api.app.services import ontrack_extract as ox, ontrack_llm as ol

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta/llama-3.1-8b-instruct"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

async def main():
    base = get_settings()
    await db.init_pool(base)
    cfg = Settings(**{**base.model_dump(), "llm_enabled": True,
                      "ontrack_llm_enabled": True, "llm_model": MODEL,
                      "llm_provider": "nvidia", "llm_max_tokens": 2000})
    cands = await ox.candidates(cfg)
    if LIMIT:
        cands = cands[:LIMIT]
    print(f"model={MODEL}  candidates={len(cands)}")
    t = time.perf_counter()
    done = {"n": 0}
    async def progress(seen, total, failed):
        done["n"] = seen
        print(f"  {seen}/{total}{' (batch failed)' if failed else ''}", flush=True)
    confirmed, failures = await ol.adjudicate(cands, cfg=cfg, on_progress=progress)
    print(f"\nconfirmed {len(confirmed)} of {len(cands)} in {time.perf_counter()-t:.0f}s")
    if not LIMIT:
        stored = await ox.store(confirmed, source="llm", cfg=cfg)
        print("stored:", stored["events"], stored["by_type"])
    else:
        for c in confirmed[:40]:
            print(f"  [{c['score']:.2f}] {c['event_type']:<19} {' '.join(c['text'].split())[:64]}")
    await db.close_pool()

asyncio.run(main())
