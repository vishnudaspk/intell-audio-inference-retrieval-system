"""
Engines package — V3
Exposes core engine abstractions and the factory resolver.
"""

from engines.base import AudioSource, TranscriptionEngine, VADEngine
from engines.factory import EngineFactory

__all__ = [
    "TranscriptionEngine",
    "VADEngine",
    "AudioSource",
    "EngineFactory",
]
