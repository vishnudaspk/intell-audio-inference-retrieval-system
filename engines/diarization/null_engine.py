"""
Null no-op diarization engine.
"""

from typing import List

from engines.diarization.base import DiarizationEngine
from schemas.models import SpeakerSegment


class NullDiarizationEngine(DiarizationEngine):
    """No-op diarization engine used when diarization is disabled."""

    def segment(self, audio_path: str) -> List[SpeakerSegment]:
        return []

    def is_available(self) -> bool:
        return True
