"""
Unit tests for non-destructive lexical search retrieval engine.
"""

from retrieval.lexical import LexicalRetrievalEngine
from schemas.models import TranscriptWord


def test_lexical_search_single_word():
    engine = LexicalRetrievalEngine()
    words = [
        TranscriptWord(word="the", start=0.0, end=0.2),
        TranscriptWord(word="quick", start=0.3, end=0.6),
        TranscriptWord(word="brown", start=0.7, end=1.0),
        TranscriptWord(word="fox", start=1.1, end=1.4),
    ]

    results = engine.search(words, "quick")
    assert len(results) == 1
    assert results[0].matched_text == "quick"
    assert results[0].start == 0.3
    assert results[0].end == 0.6


def test_lexical_search_consecutive_phrase():
    engine = LexicalRetrievalEngine()
    words = [
        TranscriptWord(word="the", start=0.0, end=0.2),
        TranscriptWord(word="quick", start=0.3, end=0.6),
        TranscriptWord(word="brown", start=0.7, end=1.0),
        TranscriptWord(word="fox", start=1.1, end=1.4),
    ]

    results = engine.search(words, "brown fox")
    assert len(results) == 1
    assert results[0].matched_text == "brown fox"
    assert results[0].start == 0.7
    assert results[0].end == 1.4


def test_lexical_search_case_insensitive():
    engine = LexicalRetrievalEngine()
    words = [
        TranscriptWord(word="Hello", start=0.0, end=0.5),
        TranscriptWord(word="World", start=0.6, end=1.0),
    ]

    results = engine.search(words, "hello world")
    assert len(results) == 1
    assert results[0].start == 0.0
    assert results[0].end == 1.0


def test_lexical_search_no_match():
    engine = LexicalRetrievalEngine()
    words = [TranscriptWord(word="hello", start=0.0, end=0.5)]

    results = engine.search(words, "missing")
    assert len(results) == 0
