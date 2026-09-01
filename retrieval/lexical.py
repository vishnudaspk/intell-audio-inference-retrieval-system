import re
from typing import List

from retrieval.base import BaseRetrievalEngine
from schemas.models import SearchResult, TranscriptWord
from utils.logger import logger


def _normalize_token(text: str) -> str:
    """Strip leading/trailing punctuation and lowercase."""
    return re.sub(r"^[^\w]+|[^\w]+$", "", text.strip()).lower()


class LexicalRetrievalEngine(BaseRetrievalEngine):
    """
    Read-only exact word and consecutive-phrase lexical search.
    Never mutates underlying word alignment databases or files.
    """

    def search(self, words: List[TranscriptWord], query: str) -> List[SearchResult]:
        if not query or not query.strip():
            return []

        # Split query by whitespace and clean each token
        raw_tokens = query.strip().split()
        search_tokens = [_normalize_token(t) for t in raw_tokens if _normalize_token(t)]

        if not search_tokens:
            return []

        # Filter out empty/invalid word records while retaining full list indexing
        valid_words = [w for w in words if w.word and w.word.strip()]
        norm_words = [_normalize_token(w.word) for w in valid_words]
        results: List[SearchResult] = []

        query_len = len(search_tokens)

        for idx in range(len(valid_words)):
            if idx + query_len > len(valid_words):
                break

            candidate_tokens = norm_words[idx : idx + query_len]

            if candidate_tokens == search_tokens:
                start_word = valid_words[idx]
                end_word = valid_words[idx + query_len - 1]

                start_t = start_word.start if start_word.start is not None else 0.0
                end_t = end_word.end if end_word.end is not None else (end_word.start or start_t)

                matched_phrase = " ".join([valid_words[idx + offset].word for offset in range(query_len)])

                results.append(
                    SearchResult(
                        matched_text=matched_phrase,
                        start=start_t,
                        end=end_t,
                        confidence=start_word.confidence,
                        word_index=idx,
                    )
                )

        logger.info(f"Lexical search query '{query}' produced {len(results)} matches")
        return results
