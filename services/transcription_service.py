"""
Transcription Service managing ASR engine delegation.
"""

from pathlib import Path
from typing import Optional

from engines.base import TranscriptionEngine
from engines.factory import EngineFactory
from schemas.enums import LanguageCode
from schemas.models import Transcript
from utils.logger import logger


class TranscriptionService:
    """Service interfacing high-level transcription requests to configured ASR engine."""

    def __init__(self, engine: Optional[TranscriptionEngine] = None):
        self.engine = engine or EngineFactory.get_transcription_engine()

    def transcribe_audio(self, audio_id: str, wav_path: Path, language: LanguageCode = LanguageCode.ENGLISH) -> Transcript:
        logger.info(f"Transcribing audio {audio_id} using {self.engine.__class__.__name__}")
        raw_text = self.engine.transcribe(wav_path)

        return Transcript(
            audio_id=audio_id,
            text=raw_text,
            language=language,
        )
