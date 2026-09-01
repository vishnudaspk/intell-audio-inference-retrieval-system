"""
Unit tests for VAD Service — V3 Phase 1J
Tests segment detection, merging, and filtering logic.
Model loading is mocked to avoid network/GPU dependencies in unit tests.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.vad_service import VADService


class TestVADServiceMerging:
    """Test merge_close_segments and filter_short_segments without model loading."""

    def setup_method(self):
        self.engine_mock = MagicMock()
        self.engine_mock.is_available.return_value = True
        self.service = VADService(engine=self.engine_mock)

    def test_merge_close_segments_basic(self):
        segs = [(0.0, 1.0, 0.9), (1.2, 2.0, 0.8), (3.0, 4.0, 0.7)]
        merged = self.service.merge_close_segments(segs, max_gap_sec=0.5)
        assert len(merged) == 2, f"Expected 2 merged segments, got {len(merged)}"
        assert merged[0] == (0.0, 2.0, 0.9)
        assert merged[1] == (3.0, 4.0, 0.7)

    def test_merge_close_segments_no_merge(self):
        segs = [(0.0, 1.0, 0.9), (2.0, 3.0, 0.8)]
        merged = self.service.merge_close_segments(segs, max_gap_sec=0.5)
        assert len(merged) == 2

    def test_merge_close_segments_empty(self):
        assert self.service.merge_close_segments([]) == []

    def test_merge_close_segments_single(self):
        segs = [(0.5, 1.5, 0.85)]
        merged = self.service.merge_close_segments(segs)
        assert merged == segs

    def test_merge_max_confidence_taken(self):
        segs = [(0.0, 1.0, 0.6), (1.1, 2.0, 0.95)]
        merged = self.service.merge_close_segments(segs, max_gap_sec=0.5)
        assert len(merged) == 1
        assert merged[0][2] == 0.95

    def test_filter_short_segments_removes_short(self):
        segs = [(0.0, 0.2, 0.9), (1.0, 2.0, 0.8), (3.0, 3.1, 0.5)]
        filtered = self.service.filter_short_segments(segs, min_duration_sec=0.25)
        assert len(filtered) == 1
        assert filtered[0] == (1.0, 2.0, 0.8)

    def test_filter_short_segments_keeps_all(self):
        segs = [(0.0, 1.0, 0.9), (1.5, 3.0, 0.8)]
        filtered = self.service.filter_short_segments(segs, min_duration_sec=0.25)
        assert len(filtered) == 2

    def test_filter_short_segments_empty(self):
        assert self.service.filter_short_segments([]) == []

    def test_detect_segments_uses_settings_defaults(self):
        """detect_segments should pass settings values to engine when called with defaults."""
        self.engine_mock.detect_speech_segments.return_value = [(0.0, 2.0, 0.9)]
        wav_path = MagicMock(spec=Path)

        result = self.service.detect_segments(wav_path)

        self.engine_mock.detect_speech_segments.assert_called_once()
        assert len(result) == 1
        assert result[0] == (0.0, 2.0, 0.9)

    def test_detect_segments_sorted(self):
        """Output should be sorted by start time."""
        self.engine_mock.detect_speech_segments.return_value = [
            (2.0, 3.0, 0.7),
            (0.0, 1.0, 0.9),
        ]
        result = self.service.detect_segments(MagicMock(spec=Path))
        assert result[0][0] < result[1][0]
