"""
Reranker interface and deterministic local scoring strategy.
"""

from abc import ABC, abstractmethod
import re
from typing import List, Optional

from config.settings import settings
from schemas.models import QueryIntent, RetrievalResult
from utils.logger import logger


class BaseReranker(ABC):
    """Abstract interface for candidate chunk rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = 5,
        query_intent: Optional[QueryIntent] = None,
    ) -> List[RetrievalResult]:
        """Rerank candidate chunks for a query and return top_k candidates."""
        pass


class DeterministicLocalReranker(BaseReranker):
    """
    Lightweight, deterministic local reranker combining vector score, BM25 score,
    lexical term overlap, and (Phase 7B) intent-aware semantic signals.

    All Phase 7B weights default to 0.0, so behavior is identical to Phase 5/6
    when RERANK_WEIGHT_* settings are left at their defaults (0.0).
    """

    def __init__(
        self,
        weight_vector: Optional[float] = None,
        weight_bm25: Optional[float] = None,
        weight_overlap: Optional[float] = None,
        weight_content: Optional[float] = None,
        weight_action: Optional[float] = None,
        weight_object: Optional[float] = None,
        weight_target: Optional[float] = None,
        weight_relation: Optional[float] = None,
    ):
        # Use settings values as defaults so tests can override by passing explicit weights
        self.weight_vector = weight_vector if weight_vector is not None else settings.RERANK_WEIGHT_VECTOR
        self.weight_bm25 = weight_bm25 if weight_bm25 is not None else settings.RERANK_WEIGHT_BM25
        self.weight_overlap = weight_overlap if weight_overlap is not None else settings.RERANK_WEIGHT_OVERLAP
        self.weight_content = weight_content if weight_content is not None else settings.RERANK_WEIGHT_CONTENT
        self.weight_action = weight_action if weight_action is not None else settings.RERANK_WEIGHT_ACTION
        self.weight_object = weight_object if weight_object is not None else settings.RERANK_WEIGHT_OBJECT
        self.weight_target = weight_target if weight_target is not None else settings.RERANK_WEIGHT_TARGET
        self.weight_relation = weight_relation if weight_relation is not None else settings.RERANK_WEIGHT_RELATION

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = 5,
        query_intent: Optional[QueryIntent] = None,
    ) -> List[RetrievalResult]:
        if not candidates or not query or not query.strip():
            return []

        query_clean = query.strip().lower()
        query_terms = set(re.sub(r"[^\w\s]", " ", query_clean).split())

        # Normalize existing scores across candidates
        max_v_score = max((c.metadata.get("vector_score", c.score) for c in candidates), default=1.0)
        max_bm25_score = max((c.score if c.retrieval_source == "bm25" else 0.0 for c in candidates), default=1.0)

        max_v_score = max(max_v_score, 1e-5)
        max_bm25_score = max(max_bm25_score, 1e-5)

        scored_list = []
        for cand in candidates:
            chunk_text = cand.chunk.text.lower()
            chunk_terms = set(re.sub(r"[^\w\s]", " ", chunk_text).split())

            # 1. Normalized vector score
            v_raw = cand.metadata.get("vector_score", cand.score if cand.retrieval_source == "vector" else 0.0)
            norm_v = max(0.0, min(1.0, float(v_raw) / max_v_score))

            # 2. Normalized BM25 score
            bm25_raw = cand.score if cand.retrieval_source == "bm25" else cand.metadata.get("bm25_score", 0.0)
            norm_bm25 = max(0.0, min(1.0, float(bm25_raw) / max_bm25_score))

            # 3. Term overlap ratio
            overlap_ratio = 0.0
            if query_terms:
                matched_terms = query_terms.intersection(chunk_terms)
                overlap_ratio = len(matched_terms) / len(query_terms)

            # Phase 7B intent-aware signals (all 0.0 when weights are 0.0)
            content_type_score = 0.0
            action_overlap = 0.0
            object_overlap = 0.0
            target_overlap = 0.0
            relationship_score = 0.0

            if query_intent is not None:
                chunk = cand.chunk

                # 4. Content type match
                if query_intent.content_type_preferences and chunk.content_type:
                    content_type_score = 1.0 if chunk.content_type in query_intent.content_type_preferences else 0.0

                # 5. Action overlap
                if query_intent.actions and chunk.actions:
                    qa = set(query_intent.actions)
                    ca = set(chunk.actions)
                    action_overlap = len(qa & ca) / max(len(qa), 1)

                # 6. Object overlap
                if query_intent.objects and chunk.objects:
                    qo = set(query_intent.objects)
                    co = set(chunk.objects)
                    object_overlap = len(qo & co) / max(len(qo), 1)

                # 7. Target overlap
                if query_intent.targets and chunk.targets:
                    qt = set(query_intent.targets)
                    ct = set(chunk.targets)
                    target_overlap = len(qt & ct) / max(len(qt), 1)

                # 8. Relationship triple score
                relationship_score = self._relationship_score(query_intent, chunk)

            # Composite rank score
            final_score = (
                self.weight_vector * norm_v
                + self.weight_bm25 * norm_bm25
                + self.weight_overlap * overlap_ratio
                + self.weight_content * content_type_score
                + self.weight_action * action_overlap
                + self.weight_object * object_overlap
                + self.weight_target * target_overlap
                + self.weight_relation * relationship_score
            )

            # Bonus for exact query string match in text
            if query_clean in chunk_text:
                final_score += 0.2

            scored_list.append((cand, final_score))

        # Sort descending by final score
        scored_list.sort(key=lambda x: x[1], reverse=True)

        results: List[RetrievalResult] = []
        for rank, (cand, final_score) in enumerate(scored_list[:top_k], start=1):
            res = RetrievalResult(
                chunk=cand.chunk,
                retrieval_source="hybrid_rrf_reranked",
                score=float(final_score),
                rank=rank,
                start_time=cand.chunk.start_time,
                end_time=cand.chunk.end_time,
                metadata={
                    **cand.metadata,
                    "rerank_score": final_score,
                    "original_source": cand.retrieval_source,
                },
            )
            results.append(res)

        logger.info(f"Reranked {len(candidates)} candidates down to {len(results)} top results.")
        return results

    @staticmethod
    def _relationship_score(query_intent: QueryIntent, chunk) -> float:
        """
        Measures whether chunk expresses the same (action, object, target) triple as the query.
        Full triple match = 1.0; two of three = 0.6; one = 0.2; none = 0.0.
        """
        a_match = bool(set(query_intent.actions) & set(chunk.actions or []))
        o_match = bool(set(query_intent.objects) & set(chunk.objects or []))
        t_match = bool(set(query_intent.targets) & set(chunk.targets or []))
        matches = int(a_match) + int(o_match) + int(t_match)
        return [0.0, 0.2, 0.6, 1.0][matches]
