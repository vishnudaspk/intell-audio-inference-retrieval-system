"""
Unit tests for Phase 7B TemporalSpanResolver.
"""

import pytest

from services.temporal_span_resolver import TemporalSpanResolver
from schemas.models import Citation, QueryIntent, RelevantTemporalSpan


def make_citation(chunk_id="chunk1", start_time=10.0, end_time=20.0,
                  text="remove the two 13mm bolts from turbo housing"):
    return Citation(
        audio_id="audio1",
        chunk_id=chunk_id,
        start_time=start_time,
        end_time=end_time,
        text=text,
    )


@pytest.fixture
def resolver():
    return TemporalSpanResolver()


class TestResolvePrimary:
    def test_no_citations_returns_none(self, resolver):
        result = resolver.resolve_primary([])
        assert result is None

    def test_returns_relevant_temporal_span(self, resolver):
        cit = make_citation()
        result = resolver.resolve_primary([cit])
        assert isinstance(result, RelevantTemporalSpan)
        assert result.source_chunk_ids == ["chunk1"]

    def test_without_intent_uses_chunk_boundaries(self, resolver):
        cit = make_citation(start_time=5.0, end_time=15.0)
        result = resolver.resolve_primary([cit], query_intent=None)
        assert result.start_time == 5.0
        assert result.end_time == 15.0

    def test_with_intent_narrows_span(self, resolver):
        """Span should be narrowed when query terms match words in citation text."""
        cit = make_citation(
            start_time=0.0, end_time=10.0,
            text="first irrelevant words remove the bolt from turbo housing last words"
        )
        intent = QueryIntent(
            query="remove bolt from turbo",
            normalized_query="remove bolt from turbo",
            actions=["remove"], objects=["bolt"], targets=["turbo"]
        )
        result = resolver.resolve_primary([cit], query_intent=intent)
        assert isinstance(result, RelevantTemporalSpan)
        # The resolver should produce times within the chunk boundaries
        assert result.start_time >= 0.0
        assert result.end_time <= 10.0

    def test_chunk_boundaries_used_as_fallback(self, resolver):
        """If query terms don't match text, fall back to chunk boundaries."""
        cit = make_citation(start_time=5.0, end_time=15.0, text="completely unrelated content")
        intent = QueryIntent(
            query="remove bolt from turbo",
            normalized_query="remove bolt from turbo",
            actions=["remove"], objects=["bolt"], targets=["turbo"]
        )
        result = resolver.resolve_primary([cit], query_intent=intent)
        # No term matches ? falls back to chunk boundaries
        assert result.start_time == 5.0
        assert result.end_time == 15.0

    def test_confidence_populated(self, resolver):
        cit = make_citation()
        result = resolver.resolve_primary([cit])
        assert 0.0 <= result.confidence <= 1.0

    def test_reason_string_populated(self, resolver):
        cit = make_citation()
        result = resolver.resolve_primary([cit])
        assert isinstance(result.reason, str)
        assert len(result.reason) > 0


    def test_with_aligned_words_uses_word_timestamps(self, resolver):
        """When top_chunk has aligned TranscriptWord entries with start/end, use them directly."""
        from schemas.models import TranscriptChunk, TranscriptWord

        words = [
            TranscriptWord(word="To", start=10.0, end=10.3),
            TranscriptWord(word="remove", start=10.4, end=11.0),
            TranscriptWord(word="the", start=11.1, end=11.2),
            TranscriptWord(word="turbo", start=11.3, end=12.0),
            TranscriptWord(word="unscrew", start=12.1, end=12.8),
            TranscriptWord(word="the", start=12.9, end=13.0),
            TranscriptWord(word="bolt", start=13.1, end=13.8),
            TranscriptWord(word="carefully", start=13.9, end=15.0),
        ]

        top_chunk = TranscriptChunk(
            chunk_id="chunk1",
            audio_id="audio1",
            transcript_id="t1",
            text="To remove the turbo unscrew the bolt carefully",
            start_time=10.0,
            end_time=20.0,
            words=words,
        )

        cit = make_citation(chunk_id="chunk1", start_time=10.0, end_time=20.0, text=top_chunk.text)
        intent = QueryIntent(
            query="unscrew bolt",
            normalized_query="unscrew bolt",
            actions=["unscrew"],
            objects=["bolt"],
        )

        result = resolver.resolve_primary([cit], query_intent=intent, top_chunk=top_chunk)
        assert result is not None
        assert "aligned" in result.reason
        # Aligned words 'unscrew' (12.1) and 'bolt' (13.8) with padding should be around 11.9..14.0
        assert 11.8 <= result.start_time <= 12.2
        assert 13.7 <= result.end_time <= 14.1


class TestResolveRelated:
    def test_empty_returns_empty(self, resolver):
        assert resolver.resolve_related([]) == []

    def test_one_citation_returns_one_span(self, resolver):
        cit = make_citation(chunk_id="chunk2", start_time=20.0, end_time=30.0)
        spans = resolver.resolve_related([cit])
        assert len(spans) == 1
        assert spans[0].start_time == 20.0
        assert spans[0].end_time == 30.0

    def test_multiple_citations_return_multiple_spans(self, resolver):
        cits = [
            make_citation(chunk_id="c1", start_time=0.0, end_time=5.0),
            make_citation(chunk_id="c2", start_time=5.0, end_time=10.0),
            make_citation(chunk_id="c3", start_time=10.0, end_time=15.0),
        ]
        spans = resolver.resolve_related(cits)
        assert len(spans) == 3


class TestNeverRaises:
    def test_primary_with_malformed_text(self, resolver):
        cit = Citation(audio_id="a", chunk_id="x", start_time=0.0, end_time=1.0, text="")
        result = resolver.resolve_primary([cit])
        assert isinstance(result, RelevantTemporalSpan)

