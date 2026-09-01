"""
Persistent BM25 lexical search engine over temporal transcript chunks.
"""

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from config.settings import settings
from schemas.models import RetrievalResult, TranscriptChunk
from utils.logger import logger


def _tokenize(text: str) -> List[str]:
    """Normalize and tokenize text for BM25 indexing."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = [t.strip() for t in cleaned.split() if t.strip()]
    return tokens


class BM25Index:
    """
    Persistent Okapi BM25 index operating over TranscriptChunk domain models.
    Supports incremental indexing, multi-audio filtering, and persistence to disk.
    """

    def __init__(self, index_file: Optional[Path] = None, k1: float = 1.5, b: float = 0.75):
        self.index_file = index_file or (settings.bm25_dir / "bm25_index.json")
        self.k1 = k1
        self.b = b

        # Internal state
        self.chunks: Dict[str, TranscriptChunk] = {}  # chunk_id -> chunk
        self.doc_tokens: Dict[str, List[str]] = {}    # chunk_id -> tokens
        self.doc_len: Dict[str, int] = {}             # chunk_id -> token count
        self.avg_dl: float = 0.0
        self.df: Dict[str, int] = {}                  # term -> document frequency
        self.doc_count: int = 0

        self.load()

    def index_chunks(self, chunks: List[TranscriptChunk]) -> None:
        """Index or update a list of transcript chunks."""
        if not chunks:
            return

        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
            tokens = _tokenize(chunk.text)
            self.doc_tokens[chunk.chunk_id] = tokens
            self.doc_len[chunk.chunk_id] = len(tokens)

        self._recalculate_stats()
        self.save()
        logger.info(f"Indexed {len(chunks)} chunks into BM25 index. Total corpus: {self.doc_count} chunks.")

    def _recalculate_stats(self) -> None:
        """Recompute corpus document frequencies and average document length."""
        self.doc_count = len(self.chunks)
        if self.doc_count == 0:
            self.avg_dl = 0.0
            self.df = {}
            return

        total_len = sum(self.doc_len.values())
        self.avg_dl = total_len / self.doc_count

        new_df: Dict[str, int] = {}
        for tokens in self.doc_tokens.values():
            unique_terms: Set[str] = set(tokens)
            for term in unique_terms:
                new_df[term] = new_df.get(term, 0) + 1

        self.df = new_df

    def search(
        self,
        query: str,
        top_k: int = 10,
        audio_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Search indexed transcript chunks using BM25 scoring.
        """
        if not query or not query.strip() or self.doc_count == 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores: Dict[str, float] = {}

        for chunk_id, chunk in self.chunks.items():
            if audio_id and chunk.audio_id != audio_id:
                continue

            doc_tokens = self.doc_tokens.get(chunk_id, [])
            d_len = self.doc_len.get(chunk_id, 0)

            if d_len == 0:
                continue

            score = 0.0
            # Term frequency map for document
            tf_map: Dict[str, int] = {}
            for t in doc_tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for q_term in query_tokens:
                if q_term not in tf_map:
                    continue

                freq = tf_map[q_term]
                doc_freq = self.df.get(q_term, 0)

                # IDF formula with smoothing
                idf = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
                
                # BM25 term score
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (d_len / max(1.0, self.avg_dl)))
                score += idf * (numerator / denominator)

            # Bonus for exact phrase match
            clean_query = query.strip().lower()
            if clean_query in chunk.text.lower():
                score += 1.5

            if score > 0.0:
                scores[chunk_id] = score

        # Sort candidate chunks by score descending
        sorted_candidates = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results: List[RetrievalResult] = []
        for rank, (c_id, score) in enumerate(sorted_candidates, start=1):
            chunk = self.chunks[c_id]
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    retrieval_source="bm25",
                    score=float(score),
                    rank=rank,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    metadata={"query_tokens": query_tokens},
                )
            )

        return results

    def delete_audio(self, audio_id: str) -> None:
        """Remove all chunks associated with audio_id from index."""
        to_delete = [c_id for c_id, chunk in self.chunks.items() if chunk.audio_id == audio_id]
        for c_id in to_delete:
            self.chunks.pop(c_id, None)
            self.doc_tokens.pop(c_id, None)
            self.doc_len.pop(c_id, None)

        self._recalculate_stats()
        self.save()

    def clear(self) -> None:
        """Clear all indexed data."""
        self.chunks.clear()
        self.doc_tokens.clear()
        self.doc_len.clear()
        self.df.clear()
        self.doc_count = 0
        self.avg_dl = 0.0
        self.save()

    def save(self) -> None:
        """Persist BM25 state to disk."""
        try:
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "chunks": [c.model_dump() for c in self.chunks.values()],
                "k1": self.k1,
                "b": self.b,
            }
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.error(f"Failed to save BM25 index to {self.index_file}: {exc}")

    def load(self) -> None:
        """Load BM25 state from disk if exists."""
        if not self.index_file.exists():
            return
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_chunks = data.get("chunks", [])
            chunks = [TranscriptChunk(**c) for c in raw_chunks]
            self.k1 = data.get("k1", 1.5)
            self.b = data.get("b", 0.75)

            for chunk in chunks:
                self.chunks[chunk.chunk_id] = chunk
                tokens = _tokenize(chunk.text)
                self.doc_tokens[chunk.chunk_id] = tokens
                self.doc_len[chunk.chunk_id] = len(tokens)

            self._recalculate_stats()
            logger.info(f"Loaded BM25 index from {self.index_file} with {self.doc_count} chunks.")
        except Exception as exc:
            logger.error(f"Failed to load BM25 index from {self.index_file}: {exc}")
