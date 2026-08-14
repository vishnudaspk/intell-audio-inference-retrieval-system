"""
Deterministic Hybrid Retrieval Pipeline combining BM25 and Vector Search with Reciprocal Rank Fusion.
"""

from typing import Dict, List, Optional

from config.settings import settings
from database.base import BaseRepository
from retrieval.bm25 import BM25Index
from retrieval.reranker import BaseReranker, DeterministicLocalReranker
from retrieval.vector_store import VectorStore
from schemas.models import QueryIntent, RetrievalResult, TranscriptChunk
from services.embedding_service import EmbeddingProvider
from utils.logger import logger


class RetrievalPipeline:
    """
    Deterministic hybrid retrieval pipeline.
    Combines BM25 lexical search and vector semantic search via Reciprocal Rank Fusion (RRF)
    and applies local reranking without relying on an LLM.

    Phase 7B additions:
    - Optional QueryUnderstanding service (intent extraction before retrieval)
    - Optional TemporalContextExpander service (configurable window context)
    Both are fully optional — omitting them preserves Phase 5/6 behavior exactly.
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        reranker: Optional[BaseReranker] = None,
        repository: Optional[BaseRepository] = None,
        query_understanding=None,        # Optional[QueryUnderstanding]
        context_expander=None,           # Optional[TemporalContextExpander]
    ):
        self.bm25_index = bm25_index
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.reranker = reranker or DeterministicLocalReranker()
        self.repository = repository
        self.query_understanding = query_understanding
        self.context_expander = context_expander

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        final_k: Optional[int] = None,
        audio_id: Optional[str] = None,
        expand_context: Optional[bool] = None,
        query_intent: Optional[QueryIntent] = None,
    ) -> List[RetrievalResult]:
        """
        Execute deterministic hybrid search for query string.

        If query_understanding is configured, extracts QueryIntent and uses it for
        intent-aware reranking. If context_expander is configured, expands each top
        result with surrounding chunks using a configurable window.
        """
        if not query or not query.strip():
            return []

        query = query.strip()
        top_k = top_k or settings.RAG_TOP_K
        final_k = final_k or settings.RAG_FINAL_K
        expand_context = expand_context if expand_context is not None else settings.EXPAND_ADJACENT_CONTEXT

        # Phase 7B: Extract query intent if service is available
        if query_intent is None and self.query_understanding is not None:
            try:
                query_intent = self.query_understanding.extract(query)
                search_query = query_intent.normalized_query
            except Exception as exc:
                logger.warning(f"Query understanding failed, using raw query: {exc}")
                search_query = query
        else:
            search_query = query

        # 1. Lexical retrieval via BM25
        bm25_results = self.bm25_index.search(search_query, top_k=top_k, audio_id=audio_id)

        # 2. Semantic retrieval via Vector Store
        vector_results: List[RetrievalResult] = []
        try:
            query_emb = self.embedding_provider.embed_query(search_query)
            vector_results = self.vector_store.search(query_emb, top_k=top_k, audio_id=audio_id)
        except Exception as exc:
            logger.warning(f"Vector search failed during hybrid retrieval: {exc}")
            if not settings.ALLOW_LEXICAL_FALLBACK and not isinstance(self.vector_store, type(None)):
                logger.error("Vector search unavailable and ALLOW_LEXICAL_FALLBACK is False.")

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        candidate_chunks: Dict[str, TranscriptChunk] = {}
        metadata_map: Dict[str, dict] = {}

        for rank, res in enumerate(bm25_results, start=1):
            c_id = res.chunk.chunk_id
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (60.0 + rank))
            candidate_chunks[c_id] = res.chunk
            metadata_map[c_id] = metadata_map.get(c_id, {})
            metadata_map[c_id]["bm25_score"] = res.score
            metadata_map[c_id]["bm25_rank"] = rank

        for rank, res in enumerate(vector_results, start=1):
            c_id = res.chunk.chunk_id
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + (1.0 / (60.0 + rank))
            candidate_chunks[c_id] = res.chunk
            metadata_map[c_id] = metadata_map.get(c_id, {})
            metadata_map[c_id]["vector_score"] = res.score
            metadata_map[c_id]["vector_rank"] = rank

        fused_candidates: List[RetrievalResult] = []
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        for rank, (c_id, rrf_score) in enumerate(sorted_rrf, start=1):
            chunk = candidate_chunks[c_id]
            fused_candidates.append(
                RetrievalResult(
                    chunk=chunk,
                    retrieval_source="hybrid_rrf",
                    score=float(rrf_score),
                    rank=rank,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    metadata=metadata_map[c_id],
                )
            )

        # 4. Reranking (Phase 7B: passes query_intent for intent-aware scoring)
        reranked_results = self.reranker.rerank(
            query, fused_candidates, top_k=final_k, query_intent=query_intent
        )

        # Attach query_intent to retrieval_metadata for downstream consumers
        if query_intent is not None:
            for res in reranked_results:
                res.metadata["query_intent"] = query_intent.model_dump()

        # 5. Context expansion
        if expand_context and self.repository:
            if self.context_expander is not None:
                # Phase 7B: configurable window expansion
                try:
                    all_chunks = self.repository.get_chunks(audio_id) if audio_id else []
                    if all_chunks:
                        self.context_expander.expand(
                            reranked_results,
                            all_chunks,
                            window_before=settings.RAG_CONTEXT_WINDOW_BEFORE,
                            window_after=settings.RAG_CONTEXT_WINDOW_AFTER,
                            max_window=settings.RAG_CONTEXT_MAX_WINDOW_CHUNKS,
                        )
                except Exception as exc:
                    logger.warning(f"Context expansion failed: {exc}")
            else:
                # Phase 5/6 fallback: single-neighbor expansion
                for res in reranked_results:
                    self._expand_chunk_context(res.chunk)

        logger.info(
            f"Hybrid retrieval produced {len(reranked_results)} results for query '{query}' "
            f"(BM25: {len(bm25_results)}, Vector: {len(vector_results)})."
        )
        return reranked_results

    def _expand_chunk_context(self, chunk: TranscriptChunk) -> None:
        """Fetch adjacent chunks (previous and next) from repository to populate expanded_context."""
        if not self.repository:
            return

        try:
            all_chunks = self.repository.get_chunks(chunk.audio_id)
            if not all_chunks:
                return

            seq = chunk.sequence_order
            prev_chunk = next((c for c in all_chunks if c.sequence_order == seq - 1), None)
            next_chunk = next((c for c in all_chunks if c.sequence_order == seq + 1), None)

            context_parts = []
            if prev_chunk:
                context_parts.append(f"[Previous Context]: {prev_chunk.text}")
            context_parts.append(f"[Current Chunk]: {chunk.text}")
            if next_chunk:
                context_parts.append(f"[Next Context]: {next_chunk.text}")

            chunk.metadata["expanded_context"] = "\n".join(context_parts)
        except Exception as exc:
            logger.debug(f"Could not expand adjacent context for chunk {chunk.chunk_id}: {exc}")

