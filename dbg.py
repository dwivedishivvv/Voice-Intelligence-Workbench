import asyncio, time
from common import db
from common.config import get_settings, Settings
from api.app.services import ontrack_extract as ox, ontrack_llm as ol
from common import ontrack

async def main():
    base = get_settings()
    await db.init_pool(base)
    cfg = Settings(**{**base.model_dump(), "llm_enabled": True, "ontrack_llm_enabled": True,
                      "llm_model": "meta/llama-3.1-70b-instruct", "llm_provider": "nvidia",
                      "llm_max_tokens": 2000})
    cands = (await ox.candidates(cfg))[10:20]
    prompt = ontrack.build_prompt([c["text"] for c in cands])
    print("prompt chars:", len(prompt))
    t = time.perf_counter()
    try:
        reply = await ol._complete(cfg, prompt)
        print(f"ok in {time.perf_counter()-t:.0f}s, reply chars {len(reply)}")
        print(reply[:600])
    except Exception as e:
        print(f"FAILED after {time.perf_counter()-t:.0f}s: {type(e).__name__}: {str(e)[:400]}")
    await db.close_pool()

asyncio.run(main())
