"""Core module package for pipeline orchestration and event handling."""

from core.event_bus import EventBus, event_bus

__all__ = ["EventBus", "event_bus"]
