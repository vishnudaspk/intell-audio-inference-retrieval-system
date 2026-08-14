"""
Integration tests requiring database setup or external Gentle server.
"""

from pathlib import Path

import pytest

from database.sqlite_db import SQLiteRepository
from schemas.models import AudioAsset, Transcript, TranscriptWord


@pytest.mark.integration
def test_sqlite_repository_lifecycle(tmp_path: Path):
    db_file = tmp_path / "test_system.db"
    repo = SQLiteRepository(db_path=db_file)

    asset = AudioAsset(filename="sample.mp3", file_path=str(tmp_path / "sample.mp3"), format="mp3")
    repo.save_audio_asset(asset)

    retrieved_asset = repo.get_audio_asset(asset.id)
    assert retrieved_asset is not None
    assert retrieved_asset.filename == "sample.mp3"

    words = [
        TranscriptWord(word="test", start=0.1, end=0.5),
        TranscriptWord(word="audio", start=0.6, end=1.0),
    ]
    transcript = Transcript(audio_id=asset.id, text="test audio", words=words)
    repo.save_transcript(transcript)
    repo.save_alignment_words(asset.id, words)

    retrieved_transcript = repo.get_transcript(asset.id)
    assert retrieved_transcript is not None
    assert len(retrieved_transcript.words) == 2
    assert retrieved_transcript.words[0].word == "test"
