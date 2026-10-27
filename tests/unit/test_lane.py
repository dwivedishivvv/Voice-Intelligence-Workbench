"""The model lane (worker/lane.py): one holder at a time, interactive waiters first, and a
batch holder that can step aside between units of work. A bug here either lets two jobs
into one model (the heap-corruption crash) or makes the microphone wait on a whole upload."""
import asyncio
import threading

import pytest

from worker.lane import ModelLane, INTERACTIVE, BATCH


def test_interactive_waiter_goes_before_earlier_batch_waiters():
    async def main():
        lane, order = ModelLane(), []
        await lane.acquire(BATCH)                       # someone holds it

        async def job(name, prio):
            async with lane.hold(prio):
                order.append(name)

        tasks = [asyncio.create_task(job("batch-1", BATCH)), asyncio.create_task(job("batch-2", BATCH))]
        await asyncio.sleep(0)
        tasks.append(asyncio.create_task(job("live", INTERACTIVE)))
        await asyncio.sleep(0)
        lane.release()
        await asyncio.gather(*tasks)
        return order

    assert asyncio.run(main()) == ["live", "batch-1", "batch-2"]


def test_never_two_holders():
    async def main():
        lane, inside, peak = ModelLane(), 0, 0

        async def job(prio):
            nonlocal inside, peak
            async with lane.hold(prio):
                inside += 1; peak = max(peak, inside)
                await asyncio.sleep(0.001)
                inside -= 1

        await asyncio.gather(*(job(INTERACTIVE if i % 3 == 0 else BATCH) for i in range(30)))
        return peak

    assert asyncio.run(main()) == 1


def test_batch_holder_steps_aside_for_a_live_job_between_units():
    """The transcription stage's worker thread checks between windows; a live chunk that
    arrived mid-stage runs at the next boundary, then the stage resumes."""
    async def main():
        lane, order = ModelLane(), []
        loop = asyncio.get_running_loop()
        live_queued = threading.Event()

        def batch_work():
            for unit in range(4):
                order.append(f"batch-{unit}")
                if unit == 1:
                    live_queued.wait(5)
                lane.step_aside_from_thread(loop)

        async def live():
            await asyncio.sleep(0.05)
            task = asyncio.create_task(_live_inner())
            await asyncio.sleep(0.01)
            live_queued.set()
            await task

        async def _live_inner():
            async with lane.hold(INTERACTIVE):
                order.append("live")

        async def batch():
            async with lane:
                await asyncio.to_thread(batch_work)

        await asyncio.gather(batch(), live())
        return order

    assert asyncio.run(main()) == ["batch-0", "batch-1", "live", "batch-2", "batch-3"]


def test_a_cancelled_waiter_does_not_wedge_the_lane():
    async def main():
        lane = ModelLane()
        await lane.acquire()
        waiter = asyncio.create_task(lane.acquire(INTERACTIVE))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        lane.release()
        await asyncio.wait_for(lane.acquire(), 1)      # free again, not stuck
        lane.release()

    asyncio.run(main())
