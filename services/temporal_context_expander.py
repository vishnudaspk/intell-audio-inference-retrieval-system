"""
Temporal Context Expander - Phase 7B.

Replaces the single-neighbor hack in RetrievalPipeline._expand_chunk_context
with a configurable window that fetches N chunks before and M chunks after each
top result, capped at a maximum window size.
"""

from typing import List

from schemas.models import RetrievalResult, TranscriptChunk
from utils.logger import logger


class TemporalContextExpander:
    """
    Expands retrieved results with surrounding transcript chunks.

    For each top result at sequence_order N, collects chunks
    [N - window_before ... N ... N + window_after], capped at max_window total.
    The expanded text is stored in result.metadata["expanded_context"].
    The contributing chunk IDs are stored in result.metadata["context_chunk_ids"].
    """

    def expand(
        self,
        top_results: List[RetrievalResult],
        all_chunks: List[TranscriptChunk],
        window_before: int = 1,
        window_after: int = 2,
        max_window: int = 5,
    ) -> List[RetrievalResult]:
        """
        Expand each top result with surrounding chunks.

        Args:
            top_results: Reranked retrieval results to expand.
            all_chunks: All chunks for the same audio_id (loaded from repository).
            window_before: Number of chunks to include before the matched chunk.
            window_after: Number of chunks to include after the matched chunk.
            max_window: Maximum total chunks in the window (including the matched chunk).

        Returns:
            The same list of RetrievalResult objects with updated metadata.
        """
        if not all_chunks or not top_results:
            return top_results

        chunk_map: dict = {}
        for chunk in all_chunks:
            chunk_map[(chunk.audio_id, chunk.sequence_order)] = chunk

        for result in top_results:
            try:
                self._expand_single(result, chunk_map, window_before, window_after, max_window)
            except Exception as exc:
                logger.debug(f"Context expansion failed for chunk {result.chunk.chunk_id}: {exc}")

        return top_results

    def _expand_single(
        self,
        result: RetrievalResult,
        chunk_map: dict,
        window_before: int,
        window_after: int,
        max_window: int,
    ) -> None:
        """Expand a single result in-place."""
        chunk = result.chunk
        audio_id = chunk.audio_id
        seq = chunk.sequence_order

        actual_before = min(window_before, (max_window - 1) // 2)
        actual_after = min(window_after, max_window - 1 - actual_before)

        context_parts: List[str] = []
        context_chunk_ids: List[str] = []

        for offset in range(-actual_before, actual_after + 1):
            neighbor = chunk_map.get((audio_id, seq + offset))
            if neighbor is None:
                continue

            if offset < 0:
                context_parts.append(f"[Context Before]: {neighbor.text}")
            elif offset == 0:
                context_parts.append(f"[Current]: {chunk.text}")
            else:
                context_parts.append(f"[Context After]: {neighbor.text}")

            context_chunk_ids.append(neighbor.chunk_id)

        if context_parts:
            result.metadata["expanded_context"] = "\n".join(context_parts)
            result.metadata["context_chunk_ids"] = context_chunk_ids
