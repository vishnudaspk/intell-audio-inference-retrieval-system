"""
Engine factory for dynamic resolution based on application settings.
"""

from config.settings import settings
from engines.base import AlignmentEngine, TranscriptionEngine
from engines.gentle_engine import GentleAlignmentEngine
from engines.pocketsphinx_engine import PocketSphinxEngine
from utils.exceptions import IntellAudioError


class EngineFactory:
    """Factory resolving AI engines based on environment configuration."""

    @staticmethod
    def get_transcription_engine() -> TranscriptionEngine:
        engine_name = settings.ASR_ENGINE.lower()
        if engine_name == "pocketsphinx":
            return PocketSphinxEngine()
        else:
            raise IntellAudioError(f"Unsupported ASR engine configured: {engine_name}")

    @staticmethod
    def get_alignment_engine() -> AlignmentEngine:
        engine_name = settings.ALIGNMENT_ENGINE.lower()
        if engine_name == "gentle":
            return GentleAlignmentEngine()
        else:
            raise IntellAudioError(f"Unsupported alignment engine configured: {engine_name}")
