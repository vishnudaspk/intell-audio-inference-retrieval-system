"""
Engine factory for dynamic resolution based on application settings.
V3: Supports Whisper (ASR) and Silero (VAD). Legacy Gentle/PocketSphinx removed.
"""

from config.settings import settings
from engines.base import TranscriptionEngine, VADEngine
from utils.exceptions import IntellAudioError


class EngineFactory:
    """Factory resolving AI engines based on environment configuration."""

    @staticmethod
    def get_transcription_engine() -> TranscriptionEngine:
        """Resolve and return the configured ASR transcription engine."""
        engine_name = settings.ASR_ENGINE.lower()

        if engine_name == "whisper":
            from engines.whisper_engine import WhisperTranscriptionEngine

            return WhisperTranscriptionEngine(
                model_size=settings.WHISPER_MODEL_SIZE,
                device=settings.WHISPER_DEVICE,
                compute_type=settings.WHISPER_COMPUTE_TYPE,
                beam_size=settings.WHISPER_BEAM_SIZE,
                models_dir=settings.models_dir,
            )

        raise IntellAudioError(f"Unsupported ASR engine configured: '{engine_name}'. Valid: whisper")

    @staticmethod
    def get_vad_engine() -> VADEngine:
        """Resolve and return the configured VAD engine."""
        engine_name = settings.VAD_ENGINE.lower()

        if engine_name == "silero":
            from engines.vad_engine import SileroVADEngine

            return SileroVADEngine()

        raise IntellAudioError(f"Unsupported VAD engine configured: '{engine_name}'. Valid: silero")
