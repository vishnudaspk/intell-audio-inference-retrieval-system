"""
Unit tests for TranscriptChunker and SQLite chunk persistence.
"""

from schemas.models import Transcript, TranscriptWord
from services.chunker import TranscriptChunker


def test_transcript_chunking_with_timestamps():
    words = [
        TranscriptWord(word="Hello", start=0.0, end=0.5),
        TranscriptWord(word="world", start=0.6, end=1.0),
        TranscriptWord(word="this", start=1.1, end=1.4),
        TranscriptWord(word="is", start=1.5, end=1.7),
        TranscriptWord(word="a", start=1.8, end=1.9),
        TranscriptWord(word="test", start=2.0, end=2.5),
        TranscriptWord(word="audio", start=2.6, end=3.0),
        TranscriptWord(word="file", start=3.1, end=3.5),
    ]

    transcript = Transcript(
        id="tx_123",
        audio_id="audio_456",
        text="Hello world this is a test audio file",
        duration=3.5,
        words=words,
    )

    chunker = TranscriptChunker(chunk_size_words=4, overlap_words=1)
    chunks = chunker.chunk_transcript(transcript)

    assert len(chunks) == 3
    assert chunks[0].chunk_id == "audio_456_chk_0000"
    assert chunks[0].text == "Hello world this is"
    assert chunks[0].start_time == 0.0
    assert chunks[0].end_time == 1.7
    assert len(chunks[0].words) == 4

    assert chunks[1].chunk_id == "audio_456_chk_0001"
    assert chunks[1].text == "is a test audio"
    assert chunks[1].start_time == 1.5
    assert chunks[1].end_time == 3.0


def test_chunking_fallback_when_no_words():
    transcript = Transcript(
        id="tx_fallback",
        audio_id="audio_fallback",
        text="Sample transcript without word alignments.",
        duration=10.0,
        words=[],
    )

    chunker = TranscriptChunker()
    chunks = chunker.chunk_transcript(transcript)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "audio_fallback_chk_0000"
    assert chunks[0].text == "Sample transcript without word alignments."
    assert chunks[0].start_time == 0.0
    assert chunks[0].end_time == 10.0


def test_sqlite_chunk_persistence(tmp_path):
    from database.sqlite_db import SQLiteRepository

    db_file = tmp_path / "test_chunks.db"
    repo = SQLiteRepository(db_path=db_file)

    words = [
        TranscriptWord(word="testing", start=0.1, end=0.8),
        TranscriptWord(word="persistence", start=0.9, end=1.5),
    ]
    transcript = Transcript(
        id="tx_p",
        audio_id="audio_p",
        text="testing persistence",
        duration=1.5,
        words=words,
    )

    chunker = TranscriptChunker(chunk_size_words=10, overlap_words=0)
    chunks = chunker.chunk_transcript(transcript)

    repo.save_chunks("audio_p", chunks)
    loaded_chunks = repo.get_chunks("audio_p")

    assert len(loaded_chunks) == 1
    assert loaded_chunks[0].chunk_id == "audio_p_chk_0000"
    assert loaded_chunks[0].text == "testing persistence"
    assert loaded_chunks[0].start_time == 0.1
    assert loaded_chunks[0].end_time == 1.5
