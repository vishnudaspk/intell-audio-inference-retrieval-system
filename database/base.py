"""
Abstract repository interface for database persistence.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from schemas.models import AudioAsset, ProcessingJob, Transcript, TranscriptWord


class BaseRepository(ABC):
    """Abstract interface defining storage contract."""

    @abstractmethod
    def save_audio_asset(self, asset: AudioAsset) -> AudioAsset:
        pass

    @abstractmethod
    def get_audio_asset(self, audio_id: str) -> Optional[AudioAsset]:
        pass

    @abstractmethod
    def save_job(self, job: ProcessingJob) -> ProcessingJob:
        pass

    @abstractmethod
    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        pass

    @abstractmethod
    def save_transcript(self, transcript: Transcript) -> Transcript:
        pass

    @abstractmethod
    def get_transcript(self, audio_id: str) -> Optional[Transcript]:
        pass

    @abstractmethod
    def save_alignment_words(self, audio_id: str, words: List[TranscriptWord]) -> None:
        pass

    @abstractmethod
    def get_alignment_words(self, audio_id: str) -> List[TranscriptWord]:
        pass
