"""
Transcription Service — V3 Phase 1E
High-level ASR service wrapping the configured TranscriptionEngine.
Converts raw Whisper output into Transcript / TranscriptSegment domain objects.
"""

from pathlib import Path
from typing import List, Optional

from engines.base import TranscriptionEngine
from engines.factory import EngineFactory
from schemas.enums import LanguageCode
from schemas.models import Transcript, TranscriptSegment, TranscriptWord
from utils.logger import logger


def _map_language_code(lang: str) -> LanguageCode:
    """Map a BCP-47 language string to our LanguageCode enum, falling back to UNKNOWN."""
    mapping = {
        "en": LanguageCode.ENGLISH,
        "ar": LanguageCode.ARABIC,
        "hi": LanguageCode.HINDI,
        "ml": LanguageCode.MALAYALAM,
    }
    return mapping.get(lang.lower(), LanguageCode.UNKNOWN)


class TranscriptionService:
    """Service interfacing high-level transcription requests to the configured ASR engine."""

    def __init__(self, engine: Optional[TranscriptionEngine] = None):
        self.engine = engine or EngineFactory.get_transcription_engine()

    def transcribe_audio(
        self,
        audio_id: str,
        wav_path: Path,
        language: Optional[str] = None,
        word_timestamps: bool = True,
    ) -> Transcript:
        """
        Transcribe a normalized WAV file and return a Transcript domain object.

        Args:
            audio_id: Asset ID to associate with the transcript.
            wav_path: Path to the normalized 16kHz mono WAV.
            language: Optional BCP-47 language hint (None = auto-detect).
            word_timestamps: Request word-level timestamps from Whisper.

        Returns:
            Transcript with segments, words, duration, and detected language.
        """
        logger.info(f"[Transcription] Transcribing audio {audio_id} via {self.engine.__class__.__name__}")

        raw = self.engine.transcribe(
            audio_path=wav_path,
            language=language,
            word_timestamps=word_timestamps,
        )

        language_code = _map_language_code(raw.get("language", "en"))
        duration = raw.get("duration", 0.0)
        full_text = raw.get("text", "")

        segments: List[TranscriptSegment] = []
        for seg_data in raw.get("segments", []):
            words: List[TranscriptWord] = [
                TranscriptWord(
                    word=w["word"],
                    start=w.get("start"),
                    end=w.get("end"),
                    confidence=w.get("probability"),
                )
                for w in seg_data.get("words", [])
            ]

            segment = TranscriptSegment(
                sequence_order=seg_data.get("id", 0),
                start=seg_data.get("start"),
                end=seg_data.get("end"),
                text=seg_data.get("text", ""),
                words=words,
            )
            segments.append(segment)

        # Flatten all words for backward-compatible Transcript.words field
        all_words: List[TranscriptWord] = []
        for seg in segments:
            seg.transcript_id = audio_id  # will be overwritten after transcript ID is assigned
            all_words.extend(seg.words)

        transcript = Transcript(
            audio_id=audio_id,
            text=full_text,
            language=language_code,
            duration=duration,
            segments=segments,
            words=all_words,
        )

        # Backfill transcript_id on segments now that transcript.id is assigned
        for seg in segments:
            seg.transcript_id = transcript.id

        logger.info(
            f"[Transcription] Audio {audio_id}: {len(segments)} segment(s), "
            f"lang={language_code.value}, duration={duration:.1f}s"
        )
        return transcript
