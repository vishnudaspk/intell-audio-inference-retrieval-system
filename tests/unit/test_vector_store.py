"""
Unit tests for VectorStore abstraction, InMemoryVectorStore, and Qdrant error handling.
"""

from unittest.mock import MagicMock, patch
import pytest

from schemas.models import TranscriptChunk
from retrieval.vector_store import InMemoryVectorStore, QdrantVectorStore
from utils.exceptions import IntellAudioError


def test_in_memory_vector_store():
    store = InMemoryVectorStore()

    chunk1 = TranscriptChunk(
        chunk_id="chk_1",
        audio_id="audio_1",
        transcript_id="tx_1",
        text="Kubernetes cluster deployment and architecture",
        start_time=0.0,
        end_time=10.0,
    )
    chunk2 = TranscriptChunk(
        chunk_id="chk_2",
        audio_id="audio_1",
        transcript_id="tx_1",
        text="Financial revenue quarterly earnings report",
        start_time=10.0,
        end_time=20.0,
    )

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store.upsert_chunks([chunk1, chunk2], embeddings)

    # Search with query embedding close to chunk1
    results = store.search([0.9, 0.1, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chk_1"
    assert results[0].score > results[1].score

    # Filter by audio_id
    results_filtered = store.search([0.9, 0.1, 0.0], audio_id="audio_2")
    assert len(results_filtered) == 0

    # Delete audio
    store.delete_audio("audio_1")
    assert len(store.search([1.0, 0.0, 0.0])) == 0


@patch("qdrant_client.QdrantClient")
def test_qdrant_vector_store_explicit_error_on_unreachable(mock_qdrant):
    mock_qdrant.side_effect = Exception("Connection refused to Qdrant")

    store = QdrantVectorStore(url="http://localhost:6333")
    assert store.is_available() is False

    chunk = TranscriptChunk(
        chunk_id="chk_err",
        audio_id="audio_err",
        transcript_id="tx_err",
        text="Error test",
        start_time=0.0,
        end_time=1.0,
    )

    with pytest.raises(IntellAudioError):
        store.upsert_chunks([chunk], [[0.1, 0.2]])
