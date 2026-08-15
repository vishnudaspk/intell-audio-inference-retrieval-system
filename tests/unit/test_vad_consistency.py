"""
Unit tests for VAD segment filtering, merging, and consistency math.
"""

import pytest
from services.vad_service import VADService


class DummyVADEngine:
    def __init__(self, segments):
        self.segments = segments

    def is_available(self) -> bool:
        return True

    def detect_speech_segments(self, *args, **kwargs):
        return self.segments


def test_vad_filter_short_segments():
    """Segments shorter than 0.25s must be filtered out."""
    service = VADService(engine=DummyVADEngine([]))
    raw = [
        (0.0, 0.15, 0.9),    # 0.15s - short (filter)
        (0.5, 1.2, 0.95),    # 0.70s - valid
        (2.0, 2.24, 0.8),    # 0.24s - short (filter)
        (3.0, 5.0, 0.99),    # 2.00s - valid
    ]
    filtered = service.filter_short_segments(raw, min_duration_sec=0.25)
    assert len(filtered) == 2
    assert filtered[0] == (0.5, 1.2, 0.95)
    assert filtered[1] == (3.0, 5.0, 0.99)


def test_vad_merge_close_segments():
    """Segments separated by <= max_gap_sec must be merged together."""
    service = VADService(engine=DummyVADEngine([]))
    # Gap between seg 1 and 2 is 15.900 - 15.652 = 0.248s <= 0.3s -> MERGED
    # Gap between seg 2 and 3 is 58.140 - 57.444 = 0.696s > 0.3s -> SEPARATE
    segments = [
        (0.188, 15.652, 0.9692),
        (15.900, 57.444, 0.9585),
        (58.140, 59.178, 0.7889),
    ]
    merged = service.merge_close_segments(segments, max_gap_sec=0.3)
    assert len(merged) == 2
    assert merged[0] == (0.188, 57.444, 0.9692)
    assert merged[1] == (58.140, 59.178, 0.7889)


def test_vad_merge_empty_and_single():
    """Empty and single-element segment lists handled cleanly."""
    service = VADService(engine=DummyVADEngine([]))
    assert service.merge_close_segments([]) == []
    single = [(1.0, 2.5, 0.9)]
    assert service.merge_close_segments(single) == [(1.0, 2.5, 0.9)]
