"""
Speaker Diarization and Turn Segmentation Engine Abstractions.
"""

from abc import ABC, abstractmethod
from typing import List

from schemas.models import SpeakerSegment


class DiarizationEngine(ABC):
    """Abstract interface for speaker-turn segmentation and diarization engines."""

    @abstractmethod
    def segment(self, audio_path: str) -> List[SpeakerSegment]:
        """
        Detect speaker-turn boundaries from audio file.
        Must return empty list on any failure or if no turns detected. Must never raise.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if engine and required dependencies are available."""
        pass
