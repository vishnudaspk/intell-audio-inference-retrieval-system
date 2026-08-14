"""
Unit tests for Phase 7B intent-aware DeterministicLocalReranker.
"""

import pytest

from retrieval.reranker import DeterministicLocalReranker
from schemas.models import QueryIntent, RetrievalResult, TranscriptChunk


def make_chunk(chunk_id, text, actions=None, objects=None, targets=None, content_type=None):
    return TranscriptChunk(
        chunk_id=chunk_id,
        audio_id="test_audio",
        transcript_id="t1",
        text=text,
        start_time=0.0,
        end_time=10.0,
        actions=actions or [],
        objects=objects or [],
        targets=targets or [],
        content_type=content_type,
    )


def make_result(chunk, score=0.5, bm25_score=0.5, vector_score=0.5):
    return RetrievalResult(
        chunk=chunk,
        retrieval_source="hybrid_rrf",
        score=score,
        rank=1,
        start_time=chunk.start_time,
        end_time=chunk.end_time,
        metadata={"bm25_score": bm25_score, "vector_score": vector_score},
    )


@pytest.fixture
def intent_reranker():
    """Reranker with all Phase 7B weights enabled."""
    return DeterministicLocalReranker(
        weight_vector=0.2,
        weight_bm25=0.2,
        weight_overlap=0.2,
        weight_content=0.1,
        weight_action=0.1,
        weight_object=0.1,
        weight_target=0.1,
        weight_relation=0.0,
    )


class TestIntentAwareReranking:
    def test_content_type_score_boosts_matching_chunk(self, intent_reranker):
        chunk_a = make_chunk("a", "remove the bolt", content_type="instruction")
        chunk_b = make_chunk("b", "turbo overview", content_type="explanation")

        query_intent = QueryIntent(
            query="how to remove bolt",
            normalized_query="how to remove bolt",
            intent="procedural_instruction",
            content_type_preferences=["instruction"],
        )

        result_a = make_result(chunk_a, vector_score=0.5, bm25_score=0.5)
        result_b = make_result(chunk_b, vector_score=0.5, bm25_score=0.5)

        ranked = intent_reranker.rerank("how to remove bolt", [result_a, result_b], top_k=2, query_intent=query_intent)
        # Chunk A has matching content_type, should score higher
        assert ranked[0].chunk.chunk_id == "a"

    def test_action_overlap_boosts_chunk_with_matching_action(self, intent_reranker):
        chunk_a = make_chunk("a", "unscrew the bolt", actions=["unscrew", "remove"])
        chunk_b = make_chunk("b", "the turbo overview", actions=[])

        query_intent = QueryIntent(
            query="unscrew the bolt",
            normalized_query="unscrew the bolt",
            intent="procedural_instruction",
            actions=["unscrew"],
        )

        result_a = make_result(chunk_a)
        result_b = make_result(chunk_b)

        ranked = intent_reranker.rerank("unscrew the bolt", [result_a, result_b], top_k=2, query_intent=query_intent)
        assert ranked[0].chunk.chunk_id == "a"

    def test_backward_compat_no_intent(self):
        """Without query_intent, reranker uses only base 3 signals (backward compat)."""
        reranker = DeterministicLocalReranker(
            weight_vector=0.4,
            weight_bm25=0.3,
            weight_overlap=0.3,
        )
        chunk_a = make_chunk("a", "remove the bolt from the turbo housing")
        chunk_b = make_chunk("b", "something unrelated")

        result_a = make_result(chunk_a, bm25_score=0.8, vector_score=0.9)
        result_b = make_result(chunk_b, bm25_score=0.1, vector_score=0.1)

        ranked = reranker.rerank("remove bolt turbo", [result_a, result_b], top_k=2)
        assert ranked[0].chunk.chunk_id == "a"


class TestRelationshipScore:
    def test_full_triple_match_scores_1(self):
        reranker = DeterministicLocalReranker()
        intent = QueryIntent(
            query="q", normalized_query="q",
            actions=["remove"], objects=["bolt"], targets=["turbo"]
        )
        chunk = make_chunk("x", "remove bolt from turbo",
                           actions=["remove"], objects=["bolt"], targets=["turbo"])
        score = reranker._relationship_score(intent, chunk)
        assert score == 1.0

    def test_two_of_three_match_scores_06(self):
        reranker = DeterministicLocalReranker()
        intent = QueryIntent(
            query="q", normalized_query="q",
            actions=["remove"], objects=["bolt"], targets=["turbo"]
        )
        chunk = make_chunk("x", "remove bolt",
                           actions=["remove"], objects=["bolt"], targets=[])
        score = reranker._relationship_score(intent, chunk)
        assert score == 0.6

    def test_one_match_scores_02(self):
        reranker = DeterministicLocalReranker()
        intent = QueryIntent(
            query="q", normalized_query="q",
            actions=["remove"], objects=["bolt"], targets=["turbo"]
        )
        chunk = make_chunk("x", "about turbo",
                           actions=[], objects=[], targets=["turbo"])
        score = reranker._relationship_score(intent, chunk)
        assert score == 0.2

    def test_no_match_scores_0(self):
        reranker = DeterministicLocalReranker()
        intent = QueryIntent(
            query="q", normalized_query="q",
            actions=["remove"], objects=["bolt"], targets=["turbo"]
        )
        chunk = make_chunk("x", "unrelated text")
        score = reranker._relationship_score(intent, chunk)
        assert score == 0.0

    def test_semantic_weights_zero_strict_invariance(self):
        """When all semantic weights are 0.0, output ranking and score are strictly identical with or without intent."""
        reranker = DeterministicLocalReranker(
            weight_vector=0.4,
            weight_bm25=0.3,
            weight_overlap=0.3,
            weight_content=0.0,
            weight_action=0.0,
            weight_object=0.0,
            weight_target=0.0,
            weight_relation=0.0,
        )

        chunk1 = make_chunk("c1", "unscrew the bolt from turbo", actions=["unscrew"], objects=["bolt"], targets=["turbo"], content_type="instruction")
        chunk2 = make_chunk("c2", "turbo materials discussion", actions=[], objects=[], targets=["turbo"], content_type="discussion")

        res1 = make_result(chunk1, bm25_score=0.7, vector_score=0.8)
        res2 = make_result(chunk2, bm25_score=0.6, vector_score=0.9)

        ranked_without = reranker.rerank("unscrew bolt", [res1, res2], top_k=2, query_intent=None)

        intent = QueryIntent(
            query="unscrew bolt",
            normalized_query="unscrew bolt",
            actions=["unscrew"],
            objects=["bolt"],
            targets=["turbo"],
            content_type_preferences=["instruction"],
        )
        ranked_with = reranker.rerank("unscrew bolt", [res1, res2], top_k=2, query_intent=intent)

        assert [r.chunk.chunk_id for r in ranked_without] == [r.chunk.chunk_id for r in ranked_with]
        assert [r.score for r in ranked_without] == [r.score for r in ranked_with]
