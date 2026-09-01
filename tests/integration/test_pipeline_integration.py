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


@pytest.mark.integration
def test_speaker_segments_persistence_and_reprocess_overwrite(tmp_path: Path):
    """Test that saving segments replaces old stale segments completely on reprocess."""
    from schemas.models import AudioSegment

    db_file = tmp_path / "test_diarize_db.db"
    repo = SQLiteRepository(db_path=db_file)

    asset = AudioAsset(filename="multi.wav", file_path=str(tmp_path / "multi.wav"), format="wav")
    repo.save_audio_asset(asset)

    # Initial run: 2 stale segments
    old_segments = [
        AudioSegment(audio_id=asset.id, sequence_order=0, start_sec=0.0, end_sec=5.0, text="hello", speaker_label="Speaker 1"),
        AudioSegment(audio_id=asset.id, sequence_order=1, start_sec=5.0, end_sec=10.0, text="there", speaker_label="Speaker 2"),
    ]
    repo.save_audio_segments(asset.id, old_segments)
    assert len(repo.get_audio_segments(asset.id)) == 2

    # Reprocess run: 5 new diarized segments
    new_segments = [
        AudioSegment(audio_id=asset.id, sequence_order=0, start_sec=0.0, end_sec=2.0, text="s1", speaker_label="Speaker 1"),
        AudioSegment(audio_id=asset.id, sequence_order=1, start_sec=2.0, end_sec=4.0, text="s2", speaker_label="Speaker 2"),
        AudioSegment(audio_id=asset.id, sequence_order=2, start_sec=4.0, end_sec=6.0, text="s3", speaker_label="Speaker 3"),
        AudioSegment(audio_id=asset.id, sequence_order=3, start_sec=6.0, end_sec=8.0, text="s4", speaker_label="Speaker 4"),
        AudioSegment(audio_id=asset.id, sequence_order=4, start_sec=8.0, end_sec=10.0, text="s5", speaker_label="Speaker 5"),
    ]
    repo.save_audio_segments(asset.id, new_segments)

    retrieved = repo.get_audio_segments(asset.id)
    assert len(retrieved) == 5
    unique_speakers = sorted(list(set(s.speaker_label for s in retrieved)))
    assert unique_speakers == ["Speaker 1", "Speaker 2", "Speaker 3", "Speaker 4", "Speaker 5"]
