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

            # Execute a simple query
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

        overall_status = "ok" if (db_status == "available" and dirs_status == "ok") else "degraded"

        return {
            "status": overall_status,
            "app_name": settings.APP_NAME,
            "environment": settings.APP_ENV,
            "asr_engine": settings.ASR_ENGINE,
            "alignment_engine": settings.ALIGNMENT_ENGINE,
            "retrieval_engine": settings.RETRIEVAL_ENGINE,
            "services": {
                "gentle": gentle_status,
                "database": db_status,
                "data_directories": dirs_status,
            },
        }
