"""
Unit tests for Speaker Assignment Service.
"""

from schemas.models import SpeakerSegment, TranscriptChunk, TranscriptWord
from services.speaker_assignment import SpeakerAssignmentService


def test_speaker_assignment_no_segments():
    service = SpeakerAssignmentService()
    chunks = [
        TranscriptChunk(
            chunk_id="chk_0001",
            audio_id="a1",
            transcript_id="t1",
            text="Hello world",
            start_time=0.0,
            end_time=5.0,
        )
    ]
    assigned = service.assign(chunks, [])
    assert len(assigned) == 1
    assert assigned[0].speaker_id is None
    assert assigned[0].speaker_label == "Unknown Speaker"
    assert assigned[0].speaker_confidence == 0.0


def test_speaker_assignment_heuristic_segments_always_unknown():
    service = SpeakerAssignmentService()
    chunks = [
        TranscriptChunk(
            chunk_id="chk_0001",
            audio_id="a1",
            transcript_id="t1",
            text="Hello world",
            start_time=1.0,
            end_time=4.0,
        )
    ]
    # Heuristic segments have confidence 0.0
    segments = [
        SpeakerSegment(
            id="seg_1",
            audio_id="a1",
            speaker_id=None,
            speaker_label="Unknown Speaker",
            start_time=0.5,
            end_time=4.5,
            confidence=0.0,
        )
    ]
    assigned = service.assign(chunks, segments)
    assert assigned[0].speaker_id is None
    assert assigned[0].speaker_label == "Unknown Speaker"
    assert assigned[0].speaker_confidence == 0.0


def test_speaker_assignment_high_confidence_future_integration():
    service = SpeakerAssignmentService()
    chunks = [
        TranscriptChunk(
            chunk_id="chk_0001",
            audio_id="a1",
            transcript_id="t1",
            text="Hello world",
            start_time=1.0,
            end_time=4.0,
        ),
        TranscriptChunk(
            chunk_id="chk_0002",
            audio_id="a1",
            transcript_id="t1",
            text="How are you",
            start_time=6.0,
            end_time=9.0,
        ),
    ]
    segments = [
        SpeakerSegment(
            id="seg_1",
            audio_id="a1",
            speaker_id="spk_01",
            speaker_label="Speaker 1",
            start_time=0.0,
            end_time=5.0,
            confidence=0.95,
        ),
        SpeakerSegment(
            id="seg_2",
            audio_id="a1",
            speaker_id="spk_02",
            speaker_label="Speaker 2",
            start_time=5.5,
            end_time=10.0,
            confidence=0.90,
        ),
    ]
    assigned = service.assign(chunks, segments, confidence_threshold=0.7)
    assert assigned[0].speaker_id == "spk_01"
    assert assigned[0].speaker_label == "Speaker 1"
    assert assigned[0].speaker_confidence == 0.95

    assert assigned[1].speaker_id == "spk_02"
    assert assigned[1].speaker_label == "Speaker 2"
    assert assigned[1].speaker_confidence == 0.90
