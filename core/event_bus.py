"""
In-process asynchronous event bus for broadcasting pipeline execution events to WebSockets and subscribers.
"""

import asyncio
from collections import defaultdict
from typing import Dict, List, Optional
from utils.logger import logger


class EventBus:
    """
    Singleton in-memory event bus that handles real-time event streaming for audio jobs.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[asyncio.Queue]] = defaultdict(list)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        if self._loop is not None and not self._loop.is_closed():
            return self._loop
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        return self._loop

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Explicitly set the running asyncio event loop."""
        self._loop = loop

    def subscribe(self, job_id: str) -> asyncio.Queue:
        """Subscribe a new queue to receive events for a specific job_id."""
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[job_id].append(queue)
        logger.debug(f"EventBus: Subscribed queue for job {job_id}. Total listeners: {len(self._subscribers[job_id])}")
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        """Unsubscribe a queue from receiving events for a job_id."""
        if job_id in self._subscribers:
            if queue in self._subscribers[job_id]:
                self._subscribers[job_id].remove(queue)
            if not self._subscribers[job_id]:
                del self._subscribers[job_id]
        logger.debug(f"EventBus: Unsubscribed queue for job {job_id}")

    async def emit(self, job_id: str, event_data: dict) -> None:
        """Asynchronously emit an event dictionary to all listeners for job_id."""
        listeners = list(self._subscribers.get(job_id, []))
        for q in listeners:
            try:
                await q.put(event_data)
            except Exception as e:
                logger.warning(f"EventBus: Failed to put event into queue for job {job_id}: {e}")

    def emit_sync(self, job_id: str, event_data: dict) -> None:
        """
        Thread-safe synchronous emitter to be called from pipeline worker threads.
        Dispatches coroutine into running event loop.
        """
        loop = self._get_loop()
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(self.emit(job_id, event_data), loop)
        else:
            # Fallback if no loop is running in main thread yet
            logger.debug(f"EventBus: No running loop available for emit_sync for job {job_id}")


# Global singleton instance
event_bus = EventBus()
