"""
Integration tests for Phase 7B - Intent-Aware Retrieval & API Enriched Responses.

Tests the complete acceptance criterion flow:
Query: "Which bolt do I need to unscrew to remove the turbo?"
-> QueryUnderstanding extracts intent (procedural/recommendation), actions (unscrew, remove), objects (bolt), targets (turbo)
-> Intent-aware reranker ranks instruction chunks with turbo-bolt relationships top
-> ReasoningAgent answers and populates primary_timestamp, citations, intent, grounded
-> API endpoints /api/v1/chapters/{id} and /api/v1/speakers/{id} return expected structures
"""

import json
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from backend.api import app, repo
from database.sqlite_db import SQLiteRepository
from retrieval.bm25 import BM25Index
from retrieval.hybrid import RetrievalPipeline
from retrieval.reranker import DeterministicLocalReranker
from schemas.models import (
    AudioAsset,
    Chapter,
    QueryIntent,
    RetrievalResult,
    SpeakerSegment,
    StructuredRAGOutput,
    TranscriptChunk,
)
from services.query_understanding import QueryUnderstanding
from services.reasoning_agent import ReasoningAgent
from services.temporal_context_expander import TemporalContextExpander
from services.temporal_span_resolver import TemporalSpanResolver


@pytest.fixture
def mock_llm_provider():
    llm = MagicMock()
    llm.model_name = "Qwen3-8B"
    llm.generate.return_value = json.dumps({
        "answer": "To remove the turbo, you need to unscrew the two 13mm lower mounting bolts.",
        "evidence_ids": ["audio_turbo_001_chk_0001"],
        "grounded": True,
    })
    return llm


@pytest.fixture
def sample_indexed_data():
    """Sets up sample chunks for the turbo bolt acceptance scenario."""
    audio_id = "audio_turbo_001"

    chunk_overview = TranscriptChunk(
        chunk_id=f"{audio_id}_chk_0000",
        audio_id=audio_id,
        transcript_id="t1",
        sequence_order=0,
        text="Welcome to the automotive workshop. Today we are discussing turbocharger maintenance.",
        start_time=0.0,
        end_time=15.0,
        content_type="introduction",
        topic="turbocharger overview",
        actions=[],
        objects=["turbocharger"],
        targets=[],
    )

    chunk_instruction = TranscriptChunk(
        chunk_id=f"{audio_id}_chk_0001",
        audio_id=audio_id,
        transcript_id="t1",
        sequence_order=1,
        text="To remove the turbo from the housing, unscrew the two 13mm lower mounting bolts with a socket wrench.",
        start_time=15.0,
        end_time=32.0,
        content_type="instruction",
        topic="turbo removal",
        intent="remove_component",
        actions=["unscrew", "remove"],
        objects=["bolts"],
        targets=["turbo", "housing"],
        tools=["socket wrench"],
        chunk_summary="Unscrew two 13mm lower mounting bolts to remove turbo.",
    )

    chunk_distractor = TranscriptChunk(
        chunk_id=f"{audio_id}_chk_0002",
        audio_id=audio_id,
        transcript_id="t1",
        sequence_order=2,
        text="The turbo housing is made of cast iron and operates under extreme exhaust heat.",
        start_time=32.0,
        end_time=48.0,
        content_type="explanation",
        topic="turbo materials",
        actions=[],
        objects=["turbo housing"],
        targets=[],
    )

    return audio_id, [chunk_overview, chunk_instruction, chunk_distractor]


class TestPhase7BAcceptanceCriterion:
    def test_query_understanding_extracts_turbo_bolt_intent(self):
        qu = QueryUnderstanding()
        query = "Which bolt do I need to unscrew to remove the turbo?"
        intent = qu.extract(query)

        assert isinstance(intent, QueryIntent)
        assert intent.intent in ("recommendation", "procedural_instruction", "procedure_query")
        assert "unscrew" in intent.actions or "remove" in intent.actions
        assert "bolt" in intent.objects
        assert any("turbo" in t for t in intent.targets)

    def test_intent_aware_reranking_ranks_instruction_chunk_first(self, sample_indexed_data):
        audio_id, chunks = sample_indexed_data
        reranker = DeterministicLocalReranker(
            weight_vector=0.2,
            weight_bm25=0.2,
            weight_overlap=0.2,
            weight_content=0.1,
            weight_action=0.1,
            weight_object=0.1,
            weight_target=0.1,
            weight_relation=0.1,
        )

        query = "Which bolt do I need to unscrew to remove the turbo?"
        qu = QueryUnderstanding()
        intent = qu.extract(query)

        candidates = [
            RetrievalResult(
                chunk=chunks[0],
                retrieval_source="hybrid_rrf",
                score=0.4,
                rank=3,
                start_time=chunks[0].start_time,
                end_time=chunks[0].end_time,
                metadata={"bm25_score": 0.3, "vector_score": 0.4},
            ),
            RetrievalResult(
                chunk=chunks[1],  # instruction chunk
                retrieval_source="hybrid_rrf",
                score=0.5,
                rank=1,
                start_time=chunks[1].start_time,
                end_time=chunks[1].end_time,
                metadata={"bm25_score": 0.5, "vector_score": 0.5},
            ),
            RetrievalResult(
                chunk=chunks[2],
                retrieval_source="hybrid_rrf",
                score=0.45,
                rank=2,
                start_time=chunks[2].start_time,
                end_time=chunks[2].end_time,
                metadata={"bm25_score": 0.4, "vector_score": 0.5},
            ),
        ]

        reranked = reranker.rerank(query, candidates, top_k=3, query_intent=intent)
        assert len(reranked) == 3
        # Chunk 1 (instruction chunk) must be ranked #1
        assert reranked[0].chunk.chunk_id == f"{audio_id}_chk_0001"

    def test_reasoning_agent_produces_enriched_response(self, mock_llm_provider, sample_indexed_data):
        audio_id, chunks = sample_indexed_data
        agent = ReasoningAgent(llm_provider=mock_llm_provider)

        retrieved = [
            RetrievalResult(
                chunk=chunks[1],
                retrieval_source="hybrid_rrf_reranked",
                score=0.85,
                rank=1,
                start_time=chunks[1].start_time,
                end_time=chunks[1].end_time,
                metadata={},
            )
        ]

        qu = QueryUnderstanding()
        query = "Which bolt do I need to unscrew to remove the turbo?"
        intent = qu.extract(query)

        response = agent.answer_question(
            query=query,
            retrieved_chunks=retrieved,
            audio_id=audio_id,
            query_intent=intent,
        )

        assert response.grounded is True
        assert len(response.citations) == 1
        assert response.citations[0].chunk_id == f"{audio_id}_chk_0001"
        assert response.primary_timestamp is not None
        assert response.primary_timestamp.start_time >= 15.0
        assert response.primary_timestamp.end_time <= 32.0
        assert response.intent == intent.intent
        assert response.abstained is False
        assert "13mm" in response.answer


class TestPhase7BAPIEndpoints:
    def test_get_chapters_endpoint(self, tmp_path):
        client = TestClient(app)
        audio_id = "test_audio_api_chap"

        asset = AudioAsset(
            id=audio_id,
            filename="test.wav",
            file_path=str(tmp_path / "test.wav"),
            format="wav",
            duration=60.0,
        )
        repo.save_audio_asset(asset)

        chapters = [
            Chapter(
                chapter_id="chap_01",
                audio_id=audio_id,
                title="Introduction & Overview",
                start_time=0.0,
                end_time=30.0,
                sequence_order=0,
            ),
            Chapter(
                chapter_id="chap_02",
                audio_id=audio_id,
                title="Turbo Removal Procedure",
                start_time=30.0,
                end_time=60.0,
                sequence_order=1,
            ),
        ]
        repo.save_chapters(audio_id, chapters)

        resp = client.get(f"/api/v1/chapters/{audio_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["audio_id"] == audio_id
        assert data["count"] == 2
        assert data["chapters"][0]["title"] == "Introduction & Overview"
        assert data["chapters"][1]["title"] == "Turbo Removal Procedure"

    def test_get_speakers_endpoint(self, tmp_path):
        client = TestClient(app)
        audio_id = "test_audio_api_spk"

        asset = AudioAsset(
            id=audio_id,
            filename="test_spk.wav",
            file_path=str(tmp_path / "test_spk.wav"),
            format="wav",
            duration=45.0,
        )
        repo.save_audio_asset(asset)

        segments = [
            SpeakerSegment(
                id="seg_01",
                audio_id=audio_id,
                speaker_label="Unknown Speaker",
                start_time=0.0,
                end_time=20.0,
                confidence=0.0,
            )
        ]
        repo.save_speaker_segments(audio_id, segments)

        resp = client.get(f"/api/v1/speakers/{audio_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["audio_id"] == audio_id
        assert data["count"] == 1
        assert data["segments"][0]["speaker_label"] == "Unknown Speaker"
        assert "heuristic" in data["note"].lower()
