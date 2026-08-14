"""
Temporal Span Resolver - Phase 7B.

Resolves application-side timestamp spans from citation data.
The LLM never generates timestamps; this service reads them from chunk data only.

When word-level timestamps are available and the query has object/action terms,
it attempts word-level precision. Otherwise it falls back to chunk boundaries.
"""

import re
from typing import List, Optional

from schemas.models import Citation, QueryIntent, RelevantTemporalSpan, TranscriptChunk
from utils.logger import logger


class TemporalSpanResolver:
    """
    Converts citations into RelevantTemporalSpan objects.
    All timestamps come from chunk / word data - never from LLM output.
    """

    def resolve_primary(
        self,
        citations: List[Citation],
        query_intent: Optional[QueryIntent] = None,
        top_chunk: Optional[TranscriptChunk] = None,
    ) -> Optional[RelevantTemporalSpan]:
        """
        Resolve the primary (highest-ranked) citation into a RelevantTemporalSpan.
        If top_chunk has aligned TranscriptWord entries matching query intent terms,
        uses exact aligned word start/end times. Otherwise uses chunk boundaries or proportional approximation.
        """
        if not citations:
            return None

        primary = citations[0]

        try:
            start_time, end_time, match_reason = self._resolve_times(primary, query_intent, top_chunk)
            return RelevantTemporalSpan(
                start_time=start_time,
                end_time=end_time,
                source_chunk_ids=[primary.chunk_id],
                confidence=0.90 if "aligned" in match_reason else (0.80 if "word" in match_reason else 0.70),
                reason=match_reason,
            )
        except Exception as exc:
            logger.debug(f"Temporal span resolution failed for primary citation: {exc}")
            return RelevantTemporalSpan(
                start_time=primary.start_time,
                end_time=primary.end_time,
                source_chunk_ids=[primary.chunk_id],
                confidence=0.6,
                reason="chunk boundary (fallback)",
            )

    def resolve_related(
        self,
        citations: List[Citation],
    ) -> List[RelevantTemporalSpan]:
        """
        Resolve remaining citations (after the primary) into related temporal spans.
        Each citation becomes one RelevantTemporalSpan using its chunk boundaries.
        """
        spans: List[RelevantTemporalSpan] = []
        for cit in citations:
            try:
                spans.append(
                    RelevantTemporalSpan(
                        start_time=cit.start_time,
                        end_time=cit.end_time,
                        source_chunk_ids=[cit.chunk_id],
                        confidence=0.60,
                        reason="related citation chunk boundary",
                    )
                )
            except Exception as exc:
                logger.debug(f"Failed to resolve related citation {cit.chunk_id}: {exc}")
        return spans

    def _resolve_times(
        self,
        citation: Citation,
        query_intent: Optional[QueryIntent],
        top_chunk: Optional[TranscriptChunk] = None,
    ) -> tuple:
        """
        Try to narrow the timestamp to word-level precision using query terms.
        Returns (start_time, end_time, reason) tuple.
        """
        if not query_intent:
            return citation.start_time, citation.end_time, "chunk boundary"

        query_terms = set(query_intent.actions + query_intent.objects + query_intent.targets)
        if not query_terms:
            return citation.start_time, citation.end_time, "chunk boundary"

        # 1. First priority: Check for aligned TranscriptWord entries with real timestamps
        if top_chunk and top_chunk.words:
            aligned_matches = []
            for w in top_chunk.words:
                if w.start is not None and w.end is not None:
                    w_clean = re.sub(r"[^\w]", "", w.word.lower())
                    if w_clean in query_terms or any(term in w_clean for term in query_terms):
                        aligned_matches.append(w)

            if aligned_matches:
                # Use min start and max end of matched aligned words (with 0.2s padding bounded by chunk)
                w_start = max(citation.start_time, min(w.start for w in aligned_matches) - 0.2)
                w_end = min(citation.end_time, max(w.end for w in aligned_matches) + 0.2)
                return round(w_start, 2), round(w_end, 2), "aligned word timestamps"

        # 2. Second priority: Proportional estimation based on citation text
        text = citation.text.lower()
        text_words = re.sub(r"[^\w\s]", " ", text).split()
        total_words = max(len(text_words), 1)
        duration = citation.end_time - citation.start_time

        match_indices = []
        for i, word in enumerate(text_words):
            if word in query_terms or any(term in word for term in query_terms):
                match_indices.append(i)

        if not match_indices:
            return citation.start_time, citation.end_time, "chunk boundary"

        first_idx = max(0, match_indices[0] - 1)
        last_idx = min(total_words - 1, match_indices[-1] + 1)

        start_time = citation.start_time + (first_idx / total_words) * duration
        end_time = citation.start_time + ((last_idx + 1) / total_words) * duration

        return round(start_time, 2), round(end_time, 2), "word-level match"
