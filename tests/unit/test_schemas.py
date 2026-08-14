"""
Unit tests for Pydantic domain models.
"""

from schemas.enums import JobStatus, SourceType
from schemas.models import AudioAsset, ProcessingJob, Transcript, TranscriptWord


def test_audio_asset_schema():
    asset = AudioAsset(
        filename="test.mp3",
        file_path="/tmp/test.mp3",
        format="mp3",
        source_type=SourceType.UPLOAD,
    )
    assert asset.id is not None
    assert asset.filename == "test.mp3"


def test_transcript_schema():
    word1 = TranscriptWord(word="hello", start=0.0, end=0.5)
    word2 = TranscriptWord(word="world", start=0.5, end=1.0)
    transcript = Transcript(
        audio_id="test-audio-id",
        text="hello world",
        words=[word1, word2],
    )
    assert len(transcript.words) == 2
    assert transcript.words[0].word == "hello"


def test_processing_job_schema():
    job = ProcessingJob(audio_id="test-id", status=JobStatus.TRANSCRIBING)
    assert job.status == JobStatus.TRANSCRIBING
    assert job.timings == {}
