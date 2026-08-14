"""
Unit tests for Phase 7A Graceful Degradation in IndexingWorker.
"""

from unittest.mock import MagicMock
import pytest

from database.sqlite_db import SQLiteRepository
from retrieval.bm25 import BM25Index
from retrieval.vector_store import InMemoryVectorStore
from schemas.models import AudioAsset, Transcript, TranscriptWord
from services.chunker import TranscriptChunker
from services.embedding_service import EmbeddingProvider
from workers.indexing_worker import IndexingWorker


class DummyEmbeddingProvider(EmbeddingProvider):
    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, query):
        return [0.1, 0.2, 0.3]

    def is_available(self):
        return True

    def get_dimension(self):
        return 3



def test_indexing_worker_graceful_degradation_all_stage_failures(tmp_path):
    db_file = tmp_path / "test.db"
    repo = SQLiteRepository(db_path=db_file)

    audio_id = "test_audio_degrade"
    asset = AudioAsset(
        id=audio_id,
        filename="test.wav",
        file_path=str(tmp_path / "test.wav"),
        format="wav",
        duration=10.0,
    )
    repo.save_audio_asset(asset)

    words = [
        TranscriptWord(word="Hello", start=0.0, end=1.0),
        TranscriptWord(word="world", start=1.1, end=2.0),
        TranscriptWord(word="testing", start=2.1, end=3.0),
    ]
    transcript = Transcript(
        id="t1",
        audio_id=audio_id,
        text="Hello world testing",
        duration=3.0,
        words=words,
    )
    repo.save_transcript(transcript)

    # Mock failing diarization engine
    mock_diarization = MagicMock()
    mock_diarization.segment.side_effect = RuntimeError("Diarization audio decoder error")

    # Mock failing content analyzer
    mock_analyzer = MagicMock()
    mock_analyzer.analyze_chunks.side_effect = RuntimeError("LM Studio down")

    # Mock failing chapter generator
    mock_chapter_gen = MagicMock()
    mock_chapter_gen.generate_chapters.side_effect = RuntimeError("Chapter gen error")

    bm25 = BM25Index(index_file=tmp_path / "bm25_index.json")
    vector_store = InMemoryVectorStore()
    emb_provider = DummyEmbeddingProvider()

    worker = IndexingWorker(
        repository=repo,
        chunker=TranscriptChunker(),
        bm25_index=bm25,
        vector_store=vector_store,
        embedding_provider=emb_provider,
        diarization_engine=mock_diarization,
        content_analyzer=mock_analyzer,
        chapter_generator=mock_chapter_gen,
    )

    # CRITICAL ACCEPTANCE CRITERION: Indexing MUST succeed despite all 3 stage failures
    status = worker.index_audio(audio_id)
    assert status.status == "completed"
    assert status.total_chunks > 0
    assert status.indexed_chunks == status.total_chunks

    # Chunks are still in SQLite and vector store
    saved_chunks = repo.get_chunks(audio_id)
    assert len(saved_chunks) == status.total_chunks
    assert len(vector_store.chunks) == status.total_chunks
