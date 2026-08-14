"""
VectorStore abstract interface and Qdrant / InMemory implementations.
"""

from abc import ABC, abstractmethod
import math
import uuid
from typing import Dict, List, Optional

from config.settings import settings
from schemas.models import RetrievalResult, TranscriptChunk
from utils.exceptions import IntellAudioError
from utils.logger import logger


class VectorStore(ABC):
    """Abstract interface for vector database indices."""

    @abstractmethod
    def upsert_chunks(self, chunks: List[TranscriptChunk], embeddings: List[List[float]]) -> None:
        """Upsert embedded transcript chunks into vector storage."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        audio_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Search vector database using query embedding."""
        pass

    @abstractmethod
    def delete_audio(self, audio_id: str) -> None:
        """Delete all vectors for a specific audio asset."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if vector store service is reachable."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored vectors."""
        pass


class InMemoryVectorStore(VectorStore):
    """
    In-memory vector store computing exact cosine similarity.
    Used for isolated unit tests without requiring a running Qdrant instance.
    """

    def __init__(self):
        self.vectors: Dict[str, List[float]] = {}
        self.chunks: Dict[str, TranscriptChunk] = {}

    def upsert_chunks(self, chunks: List[TranscriptChunk], embeddings: List[List[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise IntellAudioError("Chunks and embeddings length mismatch.")

        for chunk, emb in zip(chunks, embeddings):
            self.chunks[chunk.chunk_id] = chunk
            self.vectors[chunk.chunk_id] = emb

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        audio_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        if not query_embedding or not self.vectors:
            return []

        scores: Dict[str, float] = {}
        norm_q = math.sqrt(sum(q * q for q in query_embedding))
        if norm_q == 0.0:
            return []

        for c_id, emb in self.vectors.items():
            chunk = self.chunks[c_id]
            if audio_id and chunk.audio_id != audio_id:
                continue

            dot = sum(q * e for q, e in zip(query_embedding, emb))
            norm_e = math.sqrt(sum(e * e for e in emb))
            sim = dot / (norm_q * norm_e) if norm_e > 0.0 else 0.0

            scores[c_id] = sim

        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: List[RetrievalResult] = []
        for rank, (c_id, score) in enumerate(sorted_candidates, start=1):
            chunk = self.chunks[c_id]
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    retrieval_source="vector",
                    score=float(score),
                    rank=rank,
                    start_time=chunk.start_time,
                    end_time=chunk.end_time,
                    metadata={"vector_score": score},
                )
            )

        return results

    def delete_audio(self, audio_id: str) -> None:
        to_delete = [c_id for c_id, chunk in self.chunks.items() if chunk.audio_id == audio_id]
        for c_id in to_delete:
            self.chunks.pop(c_id, None)
            self.vectors.pop(c_id, None)

    def is_available(self) -> bool:
        return True

    def clear(self) -> None:
        self.vectors.clear()
        self.chunks.clear()


class QdrantVectorStore(VectorStore):
    """
    Production vector store communicating with a local Qdrant server.
    Fails explicitly if Qdrant is unreachable.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        collection_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.url = url or settings.QDRANT_URL
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self.api_key = api_key or settings.QDRANT_API_KEY
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=self.url, api_key=self.api_key, timeout=10.0)
            return self._client
        except Exception as exc:
            logger.error(f"Failed to instantiate QdrantClient at {self.url}: {exc}")
            raise IntellAudioError(f"Qdrant client initialization failed: {exc}") from exc

    def _ensure_collection(self, vector_size: int) -> None:
        try:
            client = self._get_client()
            from qdrant_client.http import models as rest_models

            collections = client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest_models.VectorParams(
                        size=vector_size,
                        distance=rest_models.Distance.COSINE,
                    ),
                )
                logger.info(f"Created Qdrant collection '{self.collection_name}' with vector size {vector_size}.")
        except Exception as exc:
            logger.error(f"Failed to verify/create Qdrant collection '{self.collection_name}': {exc}")
            raise IntellAudioError(f"Qdrant collection error: {exc}") from exc

    def upsert_chunks(self, chunks: List[TranscriptChunk], embeddings: List[List[float]]) -> None:
        if not chunks or not embeddings:
            return

        if len(chunks) != len(embeddings):
            raise IntellAudioError("Mismatch between chunk count and embedding count.")

        vector_size = len(embeddings[0])
        self._ensure_collection(vector_size)

        try:
            client = self._get_client()
            from qdrant_client.http import models as rest_models

            # Qdrant point IDs must be unsigned integers or UUIDs.
            # Derive a deterministic UUID5 from the string chunk_id so the ID is
            # stable across re-indexing. The original chunk_id is kept in payload.
            _NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # uuid.NAMESPACE_URL

            points = []
            for chunk, emb in zip(chunks, embeddings):
                point_id = str(uuid.uuid5(_NS, chunk.chunk_id))
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "audio_id": chunk.audio_id,
                    "transcript_id": chunk.transcript_id,
                    "sequence_order": chunk.sequence_order,
                    "text": chunk.text,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "language": chunk.language,
                    "metadata": chunk.metadata,
                    "speaker_id": getattr(chunk, "speaker_id", None),
                    "speaker_label": getattr(chunk, "speaker_label", None),
                    "speaker_confidence": getattr(chunk, "speaker_confidence", 0.0),
                    "chapter_id": getattr(chunk, "chapter_id", None),
                    "topic": getattr(chunk, "topic", None),
                    "subtopic": getattr(chunk, "subtopic", None),
                    "intent": getattr(chunk, "intent", None),
                    "content_type": getattr(chunk, "content_type", None),
                    "actions": getattr(chunk, "actions", []),
                    "objects": getattr(chunk, "objects", []),
                    "targets": getattr(chunk, "targets", []),
                    "entities": getattr(chunk, "entities", []),
                    "tools": getattr(chunk, "tools", []),
                    "parts": getattr(chunk, "parts", []),
                    "locations": getattr(chunk, "locations", []),
                    "quantities": getattr(chunk, "quantities", []),
                    "conditions": getattr(chunk, "conditions", []),
                    "warnings": getattr(chunk, "warnings", []),
                    "outcomes": getattr(chunk, "outcomes", []),
                    "temporal_references": getattr(chunk, "temporal_references", []),
                    "procedure_step": getattr(chunk, "procedure_step", None),
                    "chunk_summary": getattr(chunk, "chunk_summary", None),
                }
                points.append(
                    rest_models.PointStruct(
                        id=point_id,
                        vector=emb,
                        payload=payload,
                    )
                )

            client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Upserted {len(points)} vectors to Qdrant collection '{self.collection_name}'.")
        except Exception as exc:
            logger.error(f"Failed to upsert chunks into Qdrant: {exc}")
            raise IntellAudioError(f"Qdrant upsert error: {exc}") from exc

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        audio_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        if not query_embedding:
            return []

        try:
            client = self._get_client()
            from qdrant_client.http import models as rest_models

            query_filter = None
            if audio_id:
                query_filter = rest_models.Filter(
                    must=[
                        rest_models.FieldCondition(
                            key="audio_id",
                            match=rest_models.MatchValue(value=audio_id),
                        )
                    ]
                )

            # Support both qdrant-client modern API (query_points) and legacy API (search)
            if hasattr(client, "query_points"):
                response = client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    query_filter=query_filter,
                    limit=top_k,
                )
                hits = response.points
            else:
                hits = client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    query_filter=query_filter,
                    limit=top_k,
                )

            results: List[RetrievalResult] = []
            for rank, hit in enumerate(hits, start=1):
                p = hit.payload or {}
                meta = p.get("metadata", {})
                chunk = TranscriptChunk(
                    chunk_id=p.get("chunk_id", str(hit.id)),
                    audio_id=p.get("audio_id", ""),
                    transcript_id=p.get("transcript_id", ""),
                    sequence_order=p.get("sequence_order", 0),
                    text=p.get("text", ""),
                    start_time=p.get("start_time", 0.0),
                    end_time=p.get("end_time", 0.0),
                    language=p.get("language", "en"),
                    metadata=meta,
                    speaker_id=p.get("speaker_id", meta.get("speaker_id")),
                    speaker_label=p.get("speaker_label", meta.get("speaker_label")),
                    speaker_confidence=p.get("speaker_confidence", meta.get("speaker_confidence", 0.0)),
                    chapter_id=p.get("chapter_id", meta.get("chapter_id")),
                    topic=p.get("topic", meta.get("topic")),
                    subtopic=p.get("subtopic", meta.get("subtopic")),
                    intent=p.get("intent", meta.get("intent")),
                    content_type=p.get("content_type", meta.get("content_type")),
                    actions=p.get("actions", meta.get("actions", [])),
                    objects=p.get("objects", meta.get("objects", [])),
                    targets=p.get("targets", meta.get("targets", [])),
                    entities=p.get("entities", meta.get("entities", [])),
                    tools=p.get("tools", meta.get("tools", [])),
                    parts=p.get("parts", meta.get("parts", [])),
                    locations=p.get("locations", meta.get("locations", [])),
                    quantities=p.get("quantities", meta.get("quantities", [])),
                    conditions=p.get("conditions", meta.get("conditions", [])),
                    warnings=p.get("warnings", meta.get("warnings", [])),
                    outcomes=p.get("outcomes", meta.get("outcomes", [])),
                    temporal_references=p.get("temporal_references", meta.get("temporal_references", [])),
                    procedure_step=p.get("procedure_step", meta.get("procedure_step")),
                    chunk_summary=p.get("chunk_summary", meta.get("chunk_summary")),
                )

                results.append(
                    RetrievalResult(
                        chunk=chunk,
                        retrieval_source="vector",
                        score=float(hit.score),
                        rank=rank,
                        start_time=chunk.start_time,
                        end_time=chunk.end_time,
                        metadata={"vector_score": hit.score},
                    )
                )

            return results
        except Exception as exc:
            logger.error(f"Qdrant vector search failed: {exc}")
            raise IntellAudioError(f"Qdrant vector search failed: {exc}") from exc

    def delete_audio(self, audio_id: str) -> None:
        try:
            client = self._get_client()
            from qdrant_client.http import models as rest_models

            client.delete(
                collection_name=self.collection_name,
                points_selector=rest_models.FilterSelector(
                    filter=rest_models.Filter(
                        must=[
                            rest_models.FieldCondition(
                                key="audio_id",
                                match=rest_models.MatchValue(value=audio_id),
                            )
                        ]
                    )
                ),
            )
            logger.info(f"Deleted Qdrant vectors for audio {audio_id}.")
        except Exception as exc:
            logger.error(f"Failed to delete Qdrant vectors for audio {audio_id}: {exc}")
            raise IntellAudioError(f"Qdrant delete error: {exc}") from exc

    def is_available(self) -> bool:
        try:
            client = self._get_client()
            client.get_collections()
            return True
        except Exception as exc:
            logger.debug(f"Qdrant health check failed: {exc}")
            return False

    def clear(self) -> None:
        try:
            client = self._get_client()
            client.delete_collection(collection_name=self.collection_name)
        except Exception as exc:
            logger.error(f"Failed to clear Qdrant collection: {exc}")
