"""
Unit tests for Chapter Generation Service and Persistence.
"""

import json
from unittest.mock import MagicMock
import pytest

from database.sqlite_db import SQLiteRepository
from schemas.models import Chapter, TranscriptChunk
from services.chapter_generator import ChapterGenerator


def test_chapter_generation_deterministic_pauses(tmp_path):
    chunks = [
        TranscriptChunk(
            chunk_id="chk_0",
            audio_id="a1",
            transcript_id="t1",
            text="Welcome to the tutorial on car maintenance.",
            start_time=0.0,
            end_time=5.0,
        ),
        TranscriptChunk(
            chunk_id="chk_1",
            audio_id="a1",
            transcript_id="t1",
            text="Today we will look at replacing the turbo.",
            start_time=5.5,
            end_time=10.0,
        ),
        # 5 second pause here (> CHAPTER_MIN_PAUSE_SEC 3.0s)
        TranscriptChunk(
            chunk_id="chk_2",
            audio_id="a1",
            transcript_id="t1",
            text="Now let's begin removing the two 13mm bolts.",
            start_time=15.0,
            end_time=20.0,
        ),
        TranscriptChunk(
            chunk_id="chk_3",
            audio_id="a1",
            transcript_id="t1",
            text="Lift the turbo assembly carefully.",
            start_time=20.5,
            end_time=25.0,
        ),
    ]

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = False  # Test fallback titles

    generator = ChapterGenerator(llm_provider=mock_llm)
    chapters = generator.generate_chapters(chunks, "a1")

    assert len(chapters) == 2
    # Chapter 1: chunks 0, 1
    assert chapters[0].start_time == 0.0
    assert chapters[0].end_time == 10.0
    assert "chk_0" in chapters[0].chunk_ids
    assert "chk_1" in chapters[0].chunk_ids

    # Chapter 2: chunks 2, 3
    assert chapters[1].start_time == 15.0
    assert chapters[1].end_time == 25.0
    assert "chk_2" in chapters[1].chunk_ids
    assert "chk_3" in chapters[1].chunk_ids

    # Verify chunks received assigned chapter_id
    assert chunks[0].chapter_id == chapters[0].chapter_id
    assert chunks[2].chapter_id == chapters[1].chapter_id


def test_chapter_sqlite_persistence(tmp_path):
    db_file = tmp_path / "test.db"
    repo = SQLiteRepository(db_path=db_file)

    chapters = [
        Chapter(
            chapter_id="ch_1",
            audio_id="audio_test",
            title="Introduction to Turbo Removal",
            summary="Covers safety checks.",
            start_time=0.0,
            end_time=60.0,
            dominant_topic="Introduction",
            sequence_order=0,
            speaker_ids=[],
            chunk_ids=["chk_0", "chk_1"],
        ),
        Chapter(
            chapter_id="ch_2",
            audio_id="audio_test",
            title="Removing Mounting Bolts",
            summary="Shows socket wrench usage.",
            start_time=60.0,
            end_time=150.0,
            dominant_topic="Disassembly",
            sequence_order=1,
            speaker_ids=[],
            chunk_ids=["chk_2", "chk_3"],
        ),
    ]

    repo.save_chapters("audio_test", chapters)
    loaded = repo.get_chapters("audio_test")

    assert len(loaded) == 2
    assert loaded[0].title == "Introduction to Turbo Removal"
    assert loaded[0].start_time == 0.0
    assert loaded[0].end_time == 60.0
    assert loaded[1].title == "Removing Mounting Bolts"
    assert loaded[1].start_time == 60.0
    assert loaded[1].end_time == 150.0
