"""
Lexical (exact word & phrase) search engine implementation.
"""

from typing import List

from nltk.tokenize import word_tokenize

from retrieval.base import BaseRetrievalEngine
from schemas.models import SearchResult, TranscriptWord
from utils.logger import logger


class LexicalRetrievalEngine(BaseRetrievalEngine):
    """
    Read-only exact word and consecutive-phrase lexical search.
    Never mutates underlying word alignment databases or files.
    """

    def search(self, words: List[TranscriptWord], query: str) -> List[SearchResult]:
        if not query or not query.strip():
            return []

        search_tokens = [
            t.lower()
            for t in word_tokenize(query.strip())
            if t.strip()
        ]

        if not search_tokens:
            return []

        # Filter out empty/invalid word records while retaining full list indexing
        valid_words = [w for w in words if w.word and w.word.strip()]
        results: List[SearchResult] = []

        query_len = len(search_tokens)

        for idx, word_obj in enumerate(valid_words):
            current_clean = word_obj.word.strip().lower()

            if query_len == 1:
                if current_clean == search_tokens[0]:
                    start_t = word_obj.start if word_obj.start is not None else 0.0
                    end_t = word_obj.end if word_obj.end is not None else start_t
                    results.append(
                        SearchResult(
                            matched_text=word_obj.word,
                            start=start_t,
                            end=end_t,
                            confidence=word_obj.confidence,
                            word_index=idx,
                        )
                    )
                continue

            # Multi-word consecutive phrase search
            candidate_tokens = [
                valid_words[idx + offset].word.strip().lower()
                for offset in range(query_len)
                if idx + offset < len(valid_words)
            ]

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
