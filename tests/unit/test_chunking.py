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


def test_sqlite_chunk_phase7a_metadata_roundtrip(tmp_path):
    """Verify all 22 Phase 7A/7B metadata fields survive SQLite save_chunks -> get_chunks roundtrip."""
    from database.sqlite_db import SQLiteRepository
    from schemas.models import TranscriptChunk

    db_file = tmp_path / "test_meta.db"
    repo = SQLiteRepository(db_path=db_file)

    chunk = TranscriptChunk(
        chunk_id="chk_full_meta",
        audio_id="audio_meta",
        transcript_id="t1",
        sequence_order=0,
        text="Unscrew the two 13mm bolts from the turbo housing with a ratchet.",
        start_time=10.0,
        end_time=25.0,
        speaker_id="spk_01",
        speaker_label="Mechanic",
        speaker_confidence=0.88,
        chapter_id="chap_01",
        topic="Turbo Removal",
        subtopic="Fastener Removal",
        intent="remove_component",
        content_type="instruction",
        actions=["unscrew", "remove"],
        objects=["bolts"],
        targets=["turbo", "housing"],
        entities=["Garrett"],
        tools=["ratchet"],
        parts=["13mm bolt"],
        locations=["turbo housing lower flange"],
        quantities=["two", "13mm"],
        conditions=["engine cool"],
        warnings=["avoid hot exhaust manifold"],
        outcomes=["turbo detached"],
        temporal_references=["first"],
        procedure_step=1,
        chunk_summary="Unscrew lower 13mm bolts to detach turbo.",
    )

    repo.save_chunks("audio_meta", [chunk])
    loaded = repo.get_chunks("audio_meta")

    assert len(loaded) == 1
    c = loaded[0]
    assert c.chunk_id == "chk_full_meta"
    assert c.speaker_id == "spk_01"
    assert c.speaker_label == "Mechanic"
    assert c.speaker_confidence == 0.88
    assert c.chapter_id == "chap_01"
    assert c.topic == "Turbo Removal"
    assert c.subtopic == "Fastener Removal"
    assert c.intent == "remove_component"
    assert c.content_type == "instruction"
    assert c.actions == ["unscrew", "remove"]
    assert c.objects == ["bolts"]
    assert c.targets == ["turbo", "housing"]
    assert c.entities == ["Garrett"]
    assert c.tools == ["ratchet"]
    assert c.parts == ["13mm bolt"]
    assert c.locations == ["turbo housing lower flange"]
    assert c.quantities == ["two", "13mm"]
    assert c.conditions == ["engine cool"]
    assert c.warnings == ["avoid hot exhaust manifold"]
    assert c.outcomes == ["turbo detached"]
    assert c.temporal_references == ["first"]
    assert c.procedure_step == 1
    assert c.chunk_summary == "Unscrew lower 13mm bolts to detach turbo."
