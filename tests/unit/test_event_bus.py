"""
Unit tests for the EventBus async broadcasting mechanism.
"""

import asyncio
import pytest

from core.event_bus import EventBus


def test_event_bus_subscribe_emit_unsubscribe():
    async def _run():
        bus = EventBus()
        job_id = "test_job_123"

        queue = bus.subscribe(job_id)
        assert len(bus._subscribers[job_id]) == 1

        event_payload = {"stage": "vad", "status": "running", "progress": 50}
        await bus.emit(job_id, event_payload)

        received = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert received == event_payload

        bus.unsubscribe(job_id, queue)
        assert job_id not in bus._subscribers

    asyncio.run(_run())


def test_event_bus_multiple_subscribers():
    async def _run():
        bus = EventBus()
        job_id = "test_job_multi"

        q1 = bus.subscribe(job_id)
        q2 = bus.subscribe(job_id)
        assert len(bus._subscribers[job_id]) == 2

        event_payload = {"stage": "whisper", "status": "completed", "progress": 100}
        await bus.emit(job_id, event_payload)

        rec1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        rec2 = await asyncio.wait_for(q2.get(), timeout=1.0)

        assert rec1 == event_payload
        assert rec2 == event_payload

        bus.unsubscribe(job_id, q1)
        bus.unsubscribe(job_id, q2)
        assert job_id not in bus._subscribers

    asyncio.run(_run())
