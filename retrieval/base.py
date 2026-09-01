"""
Abstract base class for transcript search and retrieval engines.
"""

from abc import ABC, abstractmethod
from typing import List

from schemas.models import SearchResult, TranscriptWord


class BaseRetrievalEngine(ABC):
    """Abstract interface for transcript search engines."""

    @abstractmethod
    def search(self, words: List[TranscriptWord], query: str) -> List[SearchResult]:
        """Search aligned words for a target word or multi-word phrase query."""
        pass
