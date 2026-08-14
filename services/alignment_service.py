"""
Alignment Service managing forced-alignment engine delegation.
"""

from pathlib import Path
from typing import Optional

from engines.base import AlignmentEngine
from engines.factory import EngineFactory
from schemas.models import AlignmentResult
from utils.logger import logger


class AlignmentService:
    """Service interfacing forced alignment requests to configured alignment engine."""

    def __init__(self, engine: Optional[AlignmentEngine] = None):
        self.engine = engine or EngineFactory.get_alignment_engine()

    def is_available(self) -> bool:
        return self.engine.is_available()

    def align_transcript(self, wav_path: Path, transcript_text: str) -> AlignmentResult:
        logger.info(f"Aligning transcript for {wav_path.name} using {self.engine.__class__.__name__}")
        return self.engine.align(wav_path, transcript_text)
