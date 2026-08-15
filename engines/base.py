"""
Base abstract interfaces and protocols for replaceable AI components and audio ingestion sources.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from schemas.models import AudioAsset


class AudioSource(Protocol):
    """Protocol for audio acquisition sources."""

    def acquire(self) -> AudioAsset:
        ...


class VADEngine(ABC):
    """Abstract interface for Voice Activity Detection (VAD) engines."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if VAD engine model is loaded and ready."""
        pass

    @abstractmethod
    def detect_speech_segments(
        self,
        audio_path: Path,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 100,
    ) -> List[Tuple[float, float, float]]:
        """
        Detect speech segments in an audio file.
        Returns a list of tuples: (start_time_sec, end_time_sec, confidence).
        """
        pass


class TranscriptionEngine(ABC):
    """Abstract interface for Speech-to-Text ASR engines."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if ASR engine model is loaded and ready."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Convert audio file to transcript with segment timestamps, language, and metadata.
        Returns a dictionary containing 'text', 'language', 'segments', and 'duration'.
        """
        pass
