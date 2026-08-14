"""
Base abstract interfaces and protocols for replaceable AI components and audio ingestion sources.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol

from schemas.models import AlignmentResult, AudioAsset


class AudioSource(Protocol):
    """Protocol for audio acquisition sources."""

    def acquire(self) -> AudioAsset:
        ...


class TranscriptionEngine(ABC):
    """Abstract interface for Speech-to-Text ASR engines."""

    @abstractmethod
    def transcribe(self, audio_path: Path) -> str:
        """Convert audio file to plain transcript text."""
        pass


class AlignmentEngine(ABC):
    """Abstract interface for forced-alignment engines."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if forced alignment server/engine is reachable."""
        pass

    @abstractmethod
    def align(self, audio_path: Path, transcript: str) -> AlignmentResult:
        """Perform forced alignment between audio and transcript."""
        pass
