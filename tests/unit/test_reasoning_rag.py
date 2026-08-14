"""
Unit tests for ReasoningAgent RAG answer synthesis and abstention logic.
"""

from unittest.mock import MagicMock

from schemas.models import RetrievalResult, TranscriptChunk
from services.reasoning_agent import ReasoningAgent


def test_reasoning_agent_grounded_answer():
    chunk = TranscriptChunk(
        chunk_id="chk_01",
        audio_id="audio_1",
        transcript_id="tx_1",
        text="The database migration was completed with zero downtime.",
        start_time=5.0,
        end_time=15.0,
    )
    retrieved = [RetrievalResult(chunk=chunk, retrieval_source="bm25", score=0.9, rank=1, start_time=5.0, end_time=15.0)]

    mock_llm = MagicMock()
    mock_llm.generate.return_value = '{"answer": "Database migration succeeded with no downtime.", "evidence_ids": ["chk_01"], "grounded": true}'

    agent = ReasoningAgent(llm_provider=mock_llm)
    response = agent.answer_question("How was database migration done?", retrieved_chunks=retrieved)

    assert response.grounded is True
    assert "Database migration succeeded" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].start_time == 5.0


def test_reasoning_agent_low_confidence_abstention():
    chunk = TranscriptChunk(
        chunk_id="chk_low",
        audio_id="audio_1",
        transcript_id="tx_1",
        text="Irrelevant text.",
        start_time=0.0,
        end_time=1.0,
    )
    # Retrieval score 0.1 below default threshold 0.3
    retrieved = [RetrievalResult(chunk=chunk, retrieval_source="bm25", score=0.1, rank=1, start_time=0.0, end_time=1.0)]

    mock_llm = MagicMock()
    agent = ReasoningAgent(llm_provider=mock_llm)
    response = agent.answer_question("What were the financial statistics?", retrieved_chunks=retrieved)

    assert response.grounded is False
    assert "couldn't find enough evidence" in response.answer
    assert len(response.citations) == 0
    mock_llm.generate.assert_not_called()
