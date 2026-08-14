"""
Unit tests for RetrievalPipeline, Reciprocal Rank Fusion (RRF), and DeterministicLocalReranker.
"""

from unittest.mock import MagicMock

from retrieval.bm25 import BM25Index
from retrieval.hybrid import RetrievalPipeline
from retrieval.reranker import DeterministicLocalReranker
from retrieval.vector_store import InMemoryVectorStore
from schemas.models import TranscriptChunk, TranscriptWord


def test_hybrid_retrieval_pipeline(tmp_path):
    # Setup BM25 & InMemoryVectorStore
    bm25 = BM25Index(index_file=tmp_path / "bm25.json")
    vector_store = InMemoryVectorStore()

    mock_embedding_provider = MagicMock()
    mock_embedding_provider.embed_query.return_value = [0.9, 0.1, 0.0]

    chunk1 = TranscriptChunk(
        chunk_id="audio1_chk_0000",
        audio_id="audio1",
        transcript_id="tx1",
        sequence_order=0,
        text="The containerized workflow reduces manual deployment steps.",
        start_time=0.0,
        end_time=12.0,
        words=[TranscriptWord(word="containerized", start=1.0, end=2.0)],
    )

    chunk2 = TranscriptChunk(
        chunk_id="audio1_chk_0001",
        audio_id="audio1",
        transcript_id="tx1",
        sequence_order=1,
        text="Quarterly revenue increased by ten percent.",
        start_time=12.5,
        end_time=25.0,
    )

    # Index into BM25 and VectorStore
    bm25.index_chunks([chunk1, chunk2])
    vector_store.upsert_chunks([chunk1, chunk2], [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])

    pipeline = RetrievalPipeline(
        bm25_index=bm25,
        vector_store=vector_store,
        embedding_provider=mock_embedding_provider,
        reranker=DeterministicLocalReranker(),
    )

    results = pipeline.search("containerized deployment", top_k=5, final_k=2)

    assert len(results) >= 1
    assert results[0].chunk.chunk_id == "audio1_chk_0000"
    assert results[0].retrieval_source == "hybrid_rrf_reranked"
    assert results[0].start_time == 0.0
    assert results[0].end_time == 12.0


def test_deterministic_reranker():
    reranker = DeterministicLocalReranker()

    chunk1 = TranscriptChunk(
        chunk_id="chk_1",
        audio_id="audio_1",
        transcript_id="tx_1",
        text="Database migration completed successfully.",
        start_time=0.0,
        end_time=5.0,
    )

    chunk2 = TranscriptChunk(
        chunk_id="chk_2",
        audio_id="audio_1",
        transcript_id="tx_1",
        text="Unrelated text about weather.",
        start_time=5.5,
        end_time=10.0,
    )

    from schemas.models import RetrievalResult

    cand1 = RetrievalResult(chunk=chunk1, retrieval_source="bm25", score=2.5, rank=1, start_time=0.0, end_time=5.0)
    cand2 = RetrievalResult(chunk=chunk2, retrieval_source="vector", score=0.2, rank=2, start_time=5.5, end_time=10.0)

    reranked = reranker.rerank("Database migration", [cand1, cand2], top_k=2)

    assert len(reranked) == 2
    assert reranked[0].chunk.chunk_id == "chk_1"
    assert reranked[0].score > reranked[1].score
