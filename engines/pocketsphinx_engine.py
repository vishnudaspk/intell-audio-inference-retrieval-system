"""
PocketSphinx offline ASR transcription engine implementation.
"""

from pathlib import Path

import speech_recognition as sr

from engines.base import TranscriptionEngine
from utils.exceptions import TranscriptionError
from utils.logger import logger


class PocketSphinxEngine(TranscriptionEngine):
    """PocketSphinx speech recognition engine wrapping speech_recognition library."""

    def transcribe(self, audio_path: Path) -> str:
        if not audio_path.exists():
            raise TranscriptionError(f"Audio file not found: {audio_path}")

        logger.info(f"Starting PocketSphinx transcription for file: {audio_path.name}")
        recognizer = sr.Recognizer()

        try:
            with sr.AudioFile(str(audio_path)) as source:
                audio_data = recognizer.record(source)

            text = recognizer.recognize_sphinx(audio_data)
            logger.info(f"PocketSphinx transcription completed successfully ({len(text)} chars)")
            return text

        except sr.UnknownValueError as exc:
            logger.error("PocketSphinx could not understand the audio signal")
            raise TranscriptionError("PocketSphinx could not understand audio") from exc
        except Exception as exc:
            logger.error(f"PocketSphinx transcription failed: {exc}")
            raise TranscriptionError(f"PocketSphinx ASR error: {exc}") from exc
