"""
Reranker interface and deterministic local scoring strategy.
"""

from abc import ABC, abstractmethod
import re
from typing import List

from schemas.models import RetrievalResult
from utils.logger import logger


class BaseReranker(ABC):
    """Abstract interface for candidate chunk rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Rerank candidate chunks for a query and return top_k candidates."""
        pass


class DeterministicLocalReranker(BaseReranker):
    """
    Lightweight, deterministic local reranker combining vector score, BM25 score,
    lexical term overlap, and exact phrase matching bonus.
    """

    def __init__(
        self,
        weight_vector: float = 0.4,
        weight_bm25: float = 0.3,
        weight_overlap: float = 0.3,
    ):
        self.weight_vector = weight_vector
        self.weight_bm25 = weight_bm25
        self.weight_overlap = weight_overlap

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = 5,
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

            # Composite rank score
            final_score = (
                self.weight_vector * norm_v
                + self.weight_bm25 * norm_bm25
                + self.weight_overlap * overlap_ratio
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
