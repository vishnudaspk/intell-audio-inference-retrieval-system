"""
Health Diagnostics Service checking external dependencies and environment readiness.
"""

from typing import Any, Dict

from config.settings import settings
from database.sqlite_db import SQLiteRepository
from engines.factory import EngineFactory
from utils.logger import logger


class HealthService:
    """Provides application diagnostics and health status checks."""

    @staticmethod
    def check_health() -> Dict[str, Any]:
        gentle_status = "unavailable"
        db_status = "unavailable"
        dirs_status = "ok"
        lm_studio_status = "unavailable"
        chat_model_status = "unavailable"
        embedding_model_status = "unavailable"
        qdrant_status = "unavailable"
        bm25_status = "available"

        # 1. Gentle check
        try:
            alignment_engine = EngineFactory.get_alignment_engine()
            if alignment_engine.is_available():
                gentle_status = "available"
        except Exception as exc:
            logger.debug(f"Gentle health check failed: {exc}")

        # 2. Database check
        try:
            repo = SQLiteRepository()
            with repo._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                if cursor.fetchone():
                    db_status = "available"
        except Exception as exc:
            logger.debug(f"Database health check failed: {exc}")

        # 3. Data directory writable check
        try:
            settings.ensure_directories()
        except Exception:
            dirs_status = "error"

        # 4. LM Studio check
        try:
            from services.embedding_service import LMStudioEmbeddingProvider
            from services.llm_service import LMStudioLLMProvider

            emb_provider = LMStudioEmbeddingProvider()
            llm_provider = LMStudioLLMProvider()

            if emb_provider.is_available():
                lm_studio_status = "available"
                embedding_model_status = "available"

            if llm_provider.is_available():
                lm_studio_status = "available"
                chat_model_status = "available"
        except Exception as exc:
            logger.debug(f"LM Studio health check failed: {exc}")

        # 5. Qdrant check
        try:
            from retrieval.vector_store import QdrantVectorStore

            v_store = QdrantVectorStore()
            if v_store.is_available():
                qdrant_status = "available"
        except Exception as exc:
            logger.debug(f"Qdrant health check failed: {exc}")

        overall_status = "ok" if (db_status == "available" and dirs_status == "ok") else "degraded"

        return {
            "status": overall_status,
            "app_name": settings.APP_NAME,
            "environment": settings.APP_ENV,
            "asr_engine": settings.ASR_ENGINE,
            "alignment_engine": settings.ALIGNMENT_ENGINE,
            "retrieval_engine": settings.RETRIEVAL_ENGINE,
            "lm_studio": lm_studio_status,
            "chat_model": chat_model_status,
            "embedding_model": embedding_model_status,
            "qdrant": qdrant_status,
            "bm25_index": bm25_status,
            "services": {
                "gentle": gentle_status,
                "database": db_status,
                "data_directories": dirs_status,
                "lm_studio": lm_studio_status,
                "qdrant": qdrant_status,
            },
        }

