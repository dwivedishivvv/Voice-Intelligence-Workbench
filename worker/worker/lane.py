"""The one lane through the model pool, with a fast lane for people who are waiting.

Every job shares one ModelPool whose models are not thread-safe (see main.startup), so
exactly one job may be inside the models at a time — that rule does not change here.

What changes is who goes next. A plain semaphore is first-come-first-served, and clip
processing used to hold it for the whole pipeline: a live microphone chunk arriving while
an 8-minute upload was being processed waited for every remaining stage of that upload,
about a minute, before its ~0.4s of transcription could start. The pipeline now takes the
lane per stage (pipeline.process_clip), and a waiting interactive job — a live chunk, a
voice query — is handed the lane ahead of the next batch stage. The batch work loses one
chunk's worth of time; the person talking into the microphone stops waiting on it.

Stages turned out to be too coarse on their own: on an 8-minute clip a single stage held
the lane for 12-50s. So the Whisper model has a lane of its own (ModelPool.asr_lane): a
live chunk needs nothing but Whisper and takes only that one — it never waits for
diarization or voiceprints, which run in other models. The rule that protects the process
still holds, because it is about one *model* being entered twice at once: everything that
calls Whisper holds asr_lane, and batch transcription steps aside between 30s windows
whenever a live chunk is waiting (step_aside_from_thread).
"""
import asyncio
import heapq
import itertools
from contextlib import asynccontextmanager

INTERACTIVE = 0
BATCH = 1


class ModelLane:
    def __init__(self):
        self._held = False
        self._waiters: list[tuple[int, int, asyncio.Future]] = []
        self._seq = itertools.count()   # FIFO within a priority

    async def acquire(self, priority: int = BATCH):
        if not self._held and not self._waiters:
            self._held = True
            return
        fut = asyncio.get_running_loop().create_future()
        entry = (priority, next(self._seq), fut)
        heapq.heappush(self._waiters, entry)
        try:
            await fut
        except asyncio.CancelledError:
            # cancelled while waiting (arq job timeout): give the lane on if it was already
            # handed to us, otherwise just leave the queue
            if fut.done() and not fut.cancelled():
                self.release()
            else:
                self._waiters.remove(entry)
                heapq.heapify(self._waiters)
            raise

    def release(self):
        while self._waiters:
            _, _, fut = heapq.heappop(self._waiters)
            if not fut.done():
                fut.set_result(None)       # ownership passes straight to the next waiter
                return
        self._held = False

    @asynccontextmanager
    async def hold(self, priority: int = BATCH):
        await self.acquire(priority)
        try:
            yield
        finally:
            self.release()

    def interactive_waiting(self) -> bool:
        return any(p == INTERACTIVE and not f.done() for p, _, f in self._waiters)

    async def _step_aside(self):
        self.release()                     # hands the lane straight to the waiter
        await self.acquire(BATCH)          # and queues behind it (and any other interactive)

    def step_aside_from_thread(self, loop: asyncio.AbstractEventLoop):
        """Called by a batch holder from its worker thread between units of work (e.g.
        between transcription windows): if an interactive job is waiting, let it through
        and block until the lane comes back. A cheap no-op otherwise.

        Reading _waiters from the thread without the loop is a benign race: a waiter that
        arrives a moment after the check is simply served at the next unit boundary."""
        if self.interactive_waiting():
            asyncio.run_coroutine_threadsafe(self._step_aside(), loop).result()

    # `async with lane:` — batch priority, so existing call sites read as before
    async def __aenter__(self):
        await self.acquire(BATCH)

    async def __aexit__(self, *exc):
        self.release()
