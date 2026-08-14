"""
Unit tests for Speaker Turn Segmentation Engines.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from engines.diarization.base import DiarizationEngine
from engines.diarization.factory import get_diarization_engine
from engines.diarization.heuristic_engine import HeuristicTurnSegmentationEngine
from engines.diarization.null_engine import NullDiarizationEngine
from schemas.models import SpeakerSegment


def test_null_diarization_engine():
    engine = NullDiarizationEngine()
    assert engine.is_available() is True
    segments = engine.segment("any_audio.wav")
    assert segments == []


def test_diarization_factory():
    engine_h = get_diarization_engine("heuristic")
    assert isinstance(engine_h, HeuristicTurnSegmentationEngine)

    engine_n = get_diarization_engine("none")
    assert isinstance(engine_n, NullDiarizationEngine)

    engine_unknown = get_diarization_engine("unknown_xyz")
    assert isinstance(engine_unknown, NullDiarizationEngine)


def test_heuristic_engine_nonexistent_file():
    engine = HeuristicTurnSegmentationEngine()
    # Must degrade gracefully and return empty list, never raise
    segments = engine.segment("nonexistent_path_to_audio_file.wav")
    assert segments == []


def test_heuristic_engine_never_fabricates_identities(tmp_path):
    # Mock pydub detect_silence
    fake_audio_file = tmp_path / "test.wav"
    fake_audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

    mock_audio = MagicMock()
    mock_audio.__len__.return_value = 5000  # 5 seconds in ms

    with patch("pydub.AudioSegment.from_file", return_value=mock_audio), \
         patch("pydub.silence.detect_silence", return_value=[(1000, 2000), (3000, 3500)]):
        engine = HeuristicTurnSegmentationEngine(min_silence_ms=500, silence_thresh_db=-30.0)
        segments = engine.segment(str(fake_audio_file))

        assert len(segments) >= 1
        for seg in segments:
            assert isinstance(seg, SpeakerSegment)
            # CRITICAL RULE: Never fabricate speaker identities
            assert seg.speaker_id is None
            assert seg.speaker_label == "Unknown Speaker"
            assert seg.confidence == 0.0
            assert seg.start_time >= 0.0
            assert seg.end_time <= 5.0
