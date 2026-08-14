"""
Heuristic speaker-turn segmentation engine based on silence and energy boundaries.
"""

from pathlib import Path
from typing import List
import uuid

from config.settings import settings
from engines.diarization.base import DiarizationEngine
from schemas.models import SpeakerSegment
from utils.logger import logger


class HeuristicTurnSegmentationEngine(DiarizationEngine):
    """
    Detects candidate speaker-turn boundaries by identifying silence intervals.
    CRITICAL: This engine does NOT identify speaker identities.
    All segments are produced with speaker_id=None, speaker_label="Unknown Speaker", and confidence=0.0.
    """

    def __init__(
        self,
        min_silence_ms: int = None,
        silence_thresh_db: float = None,
    ):
        self.min_silence_ms = min_silence_ms or settings.DIARIZATION_MIN_SILENCE_MS
        self.silence_thresh_db = silence_thresh_db or settings.DIARIZATION_PAUSE_THRESHOLD_DB

    def segment(self, audio_path: str) -> List[SpeakerSegment]:
        if not audio_path or not Path(audio_path).exists():
            logger.warning(f"Audio file not found for segmentation: {audio_path}")
            return []

        try:
            from pydub import AudioSegment
            from pydub.silence import detect_silence

            audio = AudioSegment.from_file(audio_path)
            duration_s = len(audio) / 1000.0

            if duration_s <= 0.0:
                return []

            silence_ranges = detect_silence(
                audio,
                min_silence_len=self.min_silence_ms,
                silence_thresh=self.silence_thresh_db,
            )

            # Derive speech turns from non-silent intervals
            segments: List[SpeakerSegment] = []
            audio_id = Path(audio_path).stem

            if not silence_ranges:
                # Continuous speech or no silence detected
                segments.append(
                    SpeakerSegment(
                        id=str(uuid.uuid4()),
                        audio_id=audio_id,
                        speaker_id=None,
                        speaker_label="Unknown Speaker",
                        start_time=0.0,
                        end_time=duration_s,
                        confidence=0.0,
                    )
                )
                return segments

            # Build speech segments between silence intervals
            current_start = 0.0
            for sil_start_ms, sil_end_ms in silence_ranges:
                sil_start = sil_start_ms / 1000.0
                sil_end = sil_end_ms / 1000.0

                if sil_start > current_start + 0.3:  # minimum speech chunk 300ms
                    segments.append(
                        SpeakerSegment(
                            id=str(uuid.uuid4()),
                            audio_id=audio_id,
                            speaker_id=None,
                            speaker_label="Unknown Speaker",
                            start_time=round(current_start, 2),
                            end_time=round(sil_start, 2),
                            confidence=0.0,
                        )
                    )
                current_start = sil_end

            if current_start < duration_s - 0.3:
                segments.append(
                    SpeakerSegment(
                        id=str(uuid.uuid4()),
                        audio_id=audio_id,
                        speaker_id=None,
                        speaker_label="Unknown Speaker",
                        start_time=round(current_start, 2),
                        end_time=round(duration_s, 2),
                        confidence=0.0,
                    )
                )

            logger.info(f"HeuristicTurnSegmentation detected {len(segments)} candidate turns in {audio_path}.")
            return segments

        except Exception as exc:
            logger.warning(f"Heuristic turn segmentation degraded/failed for {audio_path}: {exc}")
            return []

    def is_available(self) -> bool:
        try:
            import pydub  # noqa: F401
            return True
        except ImportError:
            return False
