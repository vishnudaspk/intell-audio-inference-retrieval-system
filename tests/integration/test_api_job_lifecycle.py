"""
Integration test for the full developer API job lifecycle and export formats.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.api import app, repo
from schemas.analysis import (
    AnalysisMetadata,
    AnalysisResult,
    AudioInfo,
    AudioQuality,
    ConversationAnalytics,
    DiarizationResult,
    DiarizedSegment,
    ProcessingInfo,
    SpeakerProfile,
    SpeakerStatistics,
    TranscriptionResult,
    VADResult,
)
from schemas.enums import JobStatus, SourceType
from schemas.models import AudioAsset, ProcessingJob

client = TestClient(app)


def _setup_completed_job() -> str:
    job_id = "test_lifecycle_job_001"
    audio_id = "test_lifecycle_aud_001"

    asset = AudioAsset(
        id=audio_id,
        filename="test_file.wav",
        file_path="data/raw/test_file.wav",
        format="wav",
        duration=20.0,
        source_type=SourceType.UPLOAD,
    )
    job = ProcessingJob(
        id=job_id,
        audio_id=audio_id,
        status=JobStatus.COMPLETED,
        timings={"vad": 0.5, "whisper": 2.1, "speaker_embedding": 1.4},
    )
    repo.save_audio_asset(asset)
    repo.save_job(job)

    result = AnalysisResult(
        metadata=AnalysisMetadata(job_id=job_id, audio_id=audio_id),
        audio=AudioInfo(filename="test_file.wav", format="wav", duration_sec=20.0),
        audio_quality=AudioQuality(rms_energy=0.06, dynamic_range_db=34.0),
        vad=VADResult(speech_duration_sec=16.0, silence_duration_sec=4.0, speech_ratio=0.8),
        transcription=TranscriptionResult(full_text="Test transcription result."),
        diarization=DiarizationResult(
            num_speakers=2,
            segments=[
                DiarizedSegment(id="s1", sequence_order=0, start_sec=0.0, end_sec=8.0, duration_sec=8.0, text="First turn.", speaker_label="Speaker 1"),
                DiarizedSegment(id="s2", sequence_order=1, start_sec=8.5, end_sec=16.0, duration_sec=7.5, text="Second turn.", speaker_label="Speaker 2"),
            ],
        ),
        speakers=[
            SpeakerProfile(speaker_id="spk1", speaker_label="Speaker 1", statistics=SpeakerStatistics(total_speaking_sec=8.0, num_turns=1)),
            SpeakerProfile(speaker_id="spk2", speaker_label="Speaker 2", statistics=SpeakerStatistics(total_speaking_sec=7.5, num_turns=1)),
        ],
        conversation=ConversationAnalytics(total_duration_sec=20.0, num_turns=2, dominant_speaker="Speaker 1"),
        processing=ProcessingInfo(total_duration_sec=4.0, audio_duration_sec=20.0, realtime_factor=0.2),
    )
    repo.save_analysis_result(job_id, audio_id, result.model_dump_json())
    return job_id


def test_get_job_result_endpoint():
    job_id = _setup_completed_job()

    res = client.get(f"/api/v1/jobs/{job_id}")
    assert res.status_code == 200
    data = res.json()
    assert data["metadata"]["job_id"] == job_id
    assert len(data["speakers"]) == 2
    assert data["transcription"]["full_text"] == "Test transcription result."


def test_get_job_subresources():
    job_id = _setup_completed_job()

    # Speakers
    res = client.get(f"/api/v1/jobs/{job_id}/speakers")
    assert res.status_code == 200
    speakers = res.json()
    assert len(speakers) == 2
    assert speakers[0]["speaker_label"] == "Speaker 1"

    # Diarization
    res = client.get(f"/api/v1/jobs/{job_id}/diarization")
    assert res.status_code == 200
    assert res.json()["num_speakers"] == 2

    # Status
    res = client.get(f"/api/v1/jobs/{job_id}/status")
    assert res.status_code == 200
    status_data = res.json()
    assert status_data["status"] == "COMPLETED"
    assert status_data["overall_progress"] == 100


def test_get_job_exports():
    job_id = _setup_completed_job()

    # JSON export
    res = client.get(f"/api/v1/jobs/{job_id}/export?format=json")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("application/json")

    # SRT export
    res = client.get(f"/api/v1/jobs/{job_id}/export?format=srt")
    assert res.status_code == 200
    assert "00:00:00,000 --> 00:00:08,000" in res.text

    # VTT export
    res = client.get(f"/api/v1/jobs/{job_id}/export?format=vtt")
    assert res.status_code == 200
    assert "WEBVTT" in res.text

    # Markdown report
    res = client.get(f"/api/v1/jobs/{job_id}/export?format=markdown")
    assert res.status_code == 200
    assert "# Audio Intelligence & Speaker Analytics Report" in res.text
