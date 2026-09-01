"""
Unit tests for AnalysisResult schema validation and serialization.
"""

from datetime import datetime
from schemas.analysis import (
    AcousticFeatureSet,
    AnalysisMetadata,
    AnalysisResult,
    AudioInfo,
    AudioQuality,
    ConversationAnalytics,
    DiarizationResult,
    ProcessingInfo,
    SpeakerProfile,
    SpeakerStatistics,
    TranscriptionResult,
    VADResult,
)


def test_analysis_result_schema_roundtrip():
    res = AnalysisResult(
        metadata=AnalysisMetadata(job_id="job_001", audio_id="aud_001"),
        audio=AudioInfo(
            filename="meeting.wav",
            format="wav",
            duration_sec=60.0,
        ),
        audio_quality=AudioQuality(
            rms_energy=0.08,
            clipping_detected=False,
            dynamic_range_db=45.2,
        ),
        vad=VADResult(
            speech_duration_sec=45.0,
            silence_duration_sec=15.0,
            speech_ratio=0.75,
            total_segments=10,
        ),
        transcription=TranscriptionResult(
            full_text="Hello world test transcript.",
            duration_sec=60.0,
            processing_sec=4.2,
        ),
        diarization=DiarizationResult(
            num_speakers=2,
        ),
        speakers=[
            SpeakerProfile(
                speaker_id="spk_1",
                speaker_label="Speaker 1",
                statistics=SpeakerStatistics(
                    total_speaking_sec=25.0,
                    speaking_percentage=41.6,
                    num_turns=5,
                    avg_turn_sec=5.0,
                ),
            ),
            SpeakerProfile(
                speaker_id="spk_2",
                speaker_label="Speaker 2",
                statistics=SpeakerStatistics(
                    total_speaking_sec=20.0,
                    speaking_percentage=33.3,
                    num_turns=4,
                    avg_turn_sec=5.0,
                ),
            ),
        ],
        conversation=ConversationAnalytics(
            total_duration_sec=60.0,
            num_turns=9,
            num_speakers=2,
            dominant_speaker="Speaker 1",
        ),
        processing=ProcessingInfo(
            total_duration_sec=12.5,
            audio_duration_sec=60.0,
            realtime_factor=0.21,
        ),
    )

    json_str = res.model_dump_json()
    assert "meeting.wav" in json_str
    assert "Speaker 1" in json_str
    assert "0.21" in json_str

    # Deserialization check
    parsed = AnalysisResult.model_validate_json(json_str)
    assert parsed.metadata.job_id == "job_001"
    assert len(parsed.speakers) == 2
    assert parsed.speakers[0].statistics.total_speaking_sec == 25.0
    assert parsed.processing.realtime_factor == 0.21
