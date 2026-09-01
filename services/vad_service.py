"""
VAD Service — V3 Phase 1D
High-level speech segmentation service wrapping the configured VAD engine.
Handles interval merging and segment filtering.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from config.settings import settings
from engines.base import VADEngine
from engines.vad_engine import SileroVADEngine
from utils.exceptions import AudioProcessingError
from utils.logger import logger


class VADService:
    """
    Service providing high-level speech segmentation using the configured VAD engine.
    Returns speech intervals as (start_sec, end_sec, confidence) tuples.
    """

    def __init__(self, engine: Optional[VADEngine] = None):
        self.engine = engine or SileroVADEngine()

    def detect_segments(
        self,
        wav_path: Path,
        threshold: Optional[float] = None,
        min_speech_duration_ms: Optional[int] = None,
        min_silence_duration_ms: Optional[int] = None,
        speech_pad_ms: Optional[int] = None,
    ) -> List[Tuple[float, float, float]]:
        """
        Detect and return speech segments from a normalized WAV file.

        All parameters fall back to values from application settings if not provided.
        Returns: List of (start_sec, end_sec, confidence) tuples, sorted by start time.
        """
        _threshold = threshold if threshold is not None else settings.VAD_THRESHOLD
        _min_speech = min_speech_duration_ms if min_speech_duration_ms is not None else settings.VAD_MIN_SPEECH_DURATION_MS
        _min_silence = min_silence_duration_ms if min_silence_duration_ms is not None else settings.VAD_MIN_SILENCE_DURATION_MS
        _pad = speech_pad_ms if speech_pad_ms is not None else settings.VAD_SPEECH_PAD_MS

        if not self.engine.is_available():
            raise AudioProcessingError("VAD engine is not available or failed to load.")

        segments = self.engine.detect_speech_segments(
            audio_path=wav_path,
            threshold=_threshold,
            min_speech_duration_ms=_min_speech,
            min_silence_duration_ms=_min_silence,
            speech_pad_ms=_pad,
        )

        # Sort by start time (should already be sorted, but ensures robustness)
        segments = sorted(segments, key=lambda s: s[0])

        logger.debug(f"VAD produced {len(segments)} segment(s) for {wav_path.name}")
        return segments

    def merge_close_segments(
        self,
        segments: List[Tuple[float, float, float]],
        max_gap_sec: float = 0.5,
    ) -> List[Tuple[float, float, float]]:
        """
        Merge speech segments separated by gaps ≤ max_gap_sec.
        The confidence of merged segments is the max of constituent segments.

        This is useful when VAD produces many micro-segments in continuous speech.
        """
        if not segments:
            return []

        merged: List[Tuple[float, float, float]] = []
        current_start, current_end, current_conf = segments[0]

        for start, end, conf in segments[1:]:
            gap = start - current_end
            if gap <= max_gap_sec:
                # Extend current segment
                current_end = max(current_end, end)
                current_conf = max(current_conf, conf)
            else:
                merged.append((current_start, current_end, current_conf))
                current_start, current_end, current_conf = start, end, conf

        merged.append((current_start, current_end, current_conf))
        return merged

    def filter_short_segments(
        self,
        segments: List[Tuple[float, float, float]],
        min_duration_sec: float = 0.25,
    ) -> List[Tuple[float, float, float]]:
        """Remove speech segments shorter than min_duration_sec."""
        return [(s, e, c) for s, e, c in segments if (e - s) >= min_duration_sec]
