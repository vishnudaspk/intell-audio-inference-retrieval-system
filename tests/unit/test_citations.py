"""
CRITICAL INVARIANT TEST: Verify application-side timestamp citation resolution.
Citations MUST strictly resolve to actual retrieved TranscriptChunk timestamps.
Invalid LLM-hallucinated evidence IDs must be safely discarded without fabricating timestamps.
"""

from unittest.mock import MagicMock

from schemas.models import RetrievalResult, TranscriptChunk
from services.reasoning_agent import ReasoningAgent


def test_citation_resolution_invariant():
    chunk1 = TranscriptChunk(
        chunk_id="audio1_chk_0000",
        audio_id="audio1",
        transcript_id="tx1",
        sequence_order=0,
        text="The containerized workflow reduces manual deployment steps.",
        start_time=12.5,
        end_time=25.0,
    )

    retrieved = [
        RetrievalResult(
            chunk=chunk1,
            retrieval_source="hybrid_rrf",
            score=0.85,
            rank=1,
            start_time=12.5,
            end_time=25.0,
        )
    ]

    mock_llm = MagicMock()
    # LLM returns valid chunk1 AND a hallucinated chunk ID "audio1_chk_9999"
    mock_llm.generate.return_value = """{
        "answer": "The team moved to containerized deployment.",
        "evidence_ids": ["audio1_chk_0000", "audio1_chk_9999", "non_existent_id"],
        "grounded": true
    }"""

    agent = ReasoningAgent(llm_provider=mock_llm)
    response = agent.answer_question("What about deployment?", retrieved_chunks=retrieved)

    # AssertIONS FOR CITATION INVARIANT
    assert len(response.citations) == 1
    assert response.citations[0].chunk_id == "audio1_chk_0000"
    assert response.citations[0].start_time == 12.5
    assert response.citations[0].end_time == 25.0
    assert response.citations[0].text == "The containerized workflow reduces manual deployment steps."


def test_citation_resolution_all_invalid_ids():
    chunk1 = TranscriptChunk(
        chunk_id="audio1_chk_0000",
        audio_id="audio1",
        transcript_id="tx1",
        sequence_order=0,
        text="Sample transcript text.",
        start_time=1.0,
        end_time=5.0,
    )
    retrieved = [RetrievalResult(chunk=chunk1, retrieval_source="bm25", score=0.8, rank=1, start_time=1.0, end_time=5.0)]

    mock_llm = MagicMock()
    # LLM returns ONLY hallucinated IDs
    mock_llm.generate.return_value = """{
        "answer": "Some statement.",
        "evidence_ids": ["fabricated_id_100"],
        "grounded": true
    }"""

    agent = ReasoningAgent(llm_provider=mock_llm)
    response = agent.answer_question("Sample query?", retrieved_chunks=retrieved)

    # Citations list must be empty because the ID was invalid
    assert len(response.citations) == 0
