"""
Post-alignment indexing worker orchestrating transcript chunking, BM25 indexing,
vector embedding generation, and Qdrant storage.
"""

from typing import Optional

from config.settings import settings
from database.base import BaseRepository
from retrieval.bm25 import BM25Index
from retrieval.vector_store import VectorStore
from schemas.models import IndexingStatus
from services.chunker import TranscriptChunker
from services.embedding_service import EmbeddingProvider
from utils.exceptions import IntellAudioError
from utils.logger import logger


class IndexingWorker:
    """
    Orchestrates indexing pipeline for an audio asset:
    Transcript → Temporal Chunks → SQLite Chunk Store → BM25 Index → Qwen3 Embeddings → Qdrant Vector Store
    """

    def __init__(
        self,
        repository: BaseRepository,
        chunker: Optional[TranscriptChunker] = None,
        bm25_index: Optional[BM25Index] = None,
        vector_store: Optional[VectorStore] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        self.repository = repository
        self.chunker = chunker or TranscriptChunker()
        self.bm25_index = bm25_index or BM25Index()

        if vector_store:
            self.vector_store = vector_store
        else:
            from retrieval.vector_store import QdrantVectorStore
            self.vector_store = QdrantVectorStore()

        if embedding_provider:
            self.embedding_provider = embedding_provider
        else:
            from services.embedding_service import LMStudioEmbeddingProvider
            self.embedding_provider = LMStudioEmbeddingProvider()

    def index_audio(self, audio_id: str) -> IndexingStatus:
        """Execute full temporal chunking, BM25, embedding, and vector store indexing."""
        logger.info(f"Starting indexing pipeline for audio asset {audio_id}...")

        transcript = self.repository.get_transcript(audio_id)
        if not transcript:
            raise IntellAudioError(f"Cannot index audio {audio_id}: Transcript not found.")

        emb_model_name = getattr(self.embedding_provider, "model_name", settings.LM_STUDIO_EMBEDDING_MODEL)

        # 1. Save initial pending status
        status = IndexingStatus(
            audio_id=audio_id,
            status="indexing",
            total_chunks=0,
            indexed_chunks=0,
            embedding_model=emb_model_name,
        )
        self.repository.save_indexing_status(status)

        try:
            # 2. Temporal transcript chunking
            chunks = self.chunker.chunk_transcript(transcript)
            logger.info(f"Created {len(chunks)} transcript chunks for audio {audio_id}.")

            if not chunks:
                status.status = "completed"
                status.total_chunks = 0
                status.indexed_chunks = 0
                return self.repository.save_indexing_status(status)

            # 3. Store chunks in SQLite primary source of truth
            self.repository.save_chunks(audio_id, chunks)

            # 4. Index into BM25
            self.bm25_index.index_chunks(chunks)
            logger.info(f"Indexed {len(chunks)} chunks into BM25 index.")

            # 5. Generate Qwen3 Embeddings
            chunk_texts = [c.text for c in chunks]
            embeddings = self.embedding_provider.embed_texts(chunk_texts)
            logger.info(f"Generated {len(embeddings)} embeddings via LM Studio.")

            # 6. Upsert into Vector Store (Qdrant)
            self.vector_store.upsert_chunks(chunks, embeddings)
            logger.info(f"Upserted {len(chunks)} vectors to vector store.")

            emb_dim = len(embeddings[0]) if embeddings else 0

            # 7. Update status to completed
            status.status = "completed"
            status.total_chunks = len(chunks)
            status.indexed_chunks = len(chunks)
            status.embedding_dimension = emb_dim
            status.error_message = None

            saved_status = self.repository.save_indexing_status(status)
            logger.info(f"Indexing completed successfully for audio asset {audio_id}.")
            return saved_status

        except Exception as exc:
            logger.error(f"Indexing pipeline failed for audio {audio_id}: {exc}")
            status.status = "failed"
            status.error_message = str(exc)
            self.repository.save_indexing_status(status)
            raise IntellAudioError(f"Indexing failed for audio {audio_id}: {exc}") from exc
