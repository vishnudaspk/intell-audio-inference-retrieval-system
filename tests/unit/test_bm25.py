"""
Unit tests for BM25Index lexical search component.
"""

from schemas.models import TranscriptChunk, TranscriptWord
from retrieval.bm25 import BM25Index


def test_bm25_indexing_and_search(tmp_path):
    index_file = tmp_path / "bm25.json"
    bm25 = BM25Index(index_file=index_file)

    chunk1 = TranscriptChunk(
        chunk_id="audio1_chk_0000",
        audio_id="audio1",
        transcript_id="tx1",
        sequence_order=0,
        text="The containerized pipeline reduces deployment latency significantly.",
        start_time=0.0,
        end_time=10.0,
        words=[TranscriptWord(word="containerized", start=0.5, end=1.0)],
    )

    chunk2 = TranscriptChunk(
        chunk_id="audio1_chk_0001",
        audio_id="audio1",
        transcript_id="tx1",
        sequence_order=1,
        text="Revenue grew by fifteen percent during the last fiscal quarter.",
        start_time=10.5,
        end_time=20.0,
    )

    chunk3 = TranscriptChunk(
        chunk_id="audio2_chk_0000",
        audio_id="audio2",
        transcript_id="tx2",
        sequence_order=0,
        text="Database migration was completed without downtime.",
        start_time=0.0,
        end_time=5.0,
    )

    bm25.index_chunks([chunk1, chunk2, chunk3])

    # Search deployment latency
    results = bm25.search("deployment latency", top_k=5)
    assert len(results) >= 1
    assert results[0].chunk.chunk_id == "audio1_chk_0000"
    assert results[0].retrieval_source == "bm25"
    assert results[0].score > 0

    # Search with audio_id filter
    results_audio2 = bm25.search("revenue", audio_id="audio2")
    assert len(results_audio2) == 0

    results_audio1 = bm25.search("revenue", audio_id="audio1")
    assert len(results_audio1) == 1
    assert results_audio1[0].chunk.chunk_id == "audio1_chk_0001"


def test_bm25_persistence(tmp_path):
    index_file = tmp_path / "bm25.json"
    bm25 = BM25Index(index_file=index_file)

    chunk = TranscriptChunk(
        chunk_id="chk_p",
        audio_id="audio_p",
        transcript_id="tx_p",
        text="Persistent BM25 storage test.",
        start_time=0.0,
        end_time=2.0,
    )
    bm25.index_chunks([chunk])

    # Reload from disk
    bm25_reloaded = BM25Index(index_file=index_file)
    results = bm25_reloaded.search("persistent storage")
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chk_p"
