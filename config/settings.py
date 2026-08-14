"""
Configuration module for Intell Audio Inference & Retrieval System.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings using environment variables and defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "IntellAudioInferenceRetrieval"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # Base Data Directory
    DATA_DIR: Path = Path("data")

    # Gentle forced alignment server URL
    GENTLE_URL: str = "http://localhost:8888/transcriptions?async=false"

    # Database URL
    DATABASE_URL: Optional[str] = None

    # Engine Selections
    ASR_ENGINE: str = "pocketsphinx"
    ASR_MODEL: str = "default"
    ALIGNMENT_ENGINE: str = "gentle"
    RETRIEVAL_ENGINE: str = "lexical"

    # LM Studio Configuration
    LM_STUDIO_BASE_URL: str = "http://localhost:1234"
    LM_STUDIO_CHAT_MODEL: str = "qwen3-8b"
    LM_STUDIO_EMBEDDING_MODEL: str = "text-embedding-qwen3-embedding-0.6b"
    LM_STUDIO_TIMEOUT: float = 120.0

    # Qdrant Vector Store Configuration
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "intell_audio_chunks"
    QDRANT_API_KEY: Optional[str] = None
    VECTOR_STORE: str = "qdrant"
    ALLOW_LEXICAL_FALLBACK: bool = False

    # Chunking Configuration
    CHUNK_SIZE_WORDS: int = 60
    CHUNK_OVERLAP_WORDS: int = 10

    # RAG & Retrieval Parameters
    RAG_TOP_K: int = 10
    RAG_FINAL_K: int = 5
    RAG_MIN_RELEVANCE_SCORE: float = 0.3
    RAG_REQUIRE_EVIDENCE: bool = True
    RAG_ALLOW_UNGROUNDED_ANSWER: bool = False
    EXPAND_ADJACENT_CONTEXT: bool = True

    # Phase 7B Context Window Expansion (replaces binary EXPAND_ADJACENT_CONTEXT)
    RAG_CONTEXT_WINDOW_BEFORE: int = 1
    RAG_CONTEXT_WINDOW_AFTER: int = 2
    RAG_CONTEXT_MAX_WINDOW_CHUNKS: int = 5

    # Phase 7B Intent-Aware Reranker Weights (0.0 = disabled; backward-compatible)
    RERANK_WEIGHT_VECTOR: float = 0.40
    RERANK_WEIGHT_BM25: float = 0.30
    RERANK_WEIGHT_OVERLAP: float = 0.30
    RERANK_WEIGHT_CONTENT: float = 0.0
    RERANK_WEIGHT_ACTION: float = 0.0
    RERANK_WEIGHT_OBJECT: float = 0.0
    RERANK_WEIGHT_TARGET: float = 0.0
    RERANK_WEIGHT_RELATION: float = 0.0

    # Phase 7B Query Understanding
    ENABLE_QUERY_UNDERSTANDING: bool = True
    QUERY_UNDERSTANDING_LLM_FALLBACK: bool = False


    # Phase 7A Diarization Configuration
    DIARIZATION_ENGINE: str = "heuristic"
    DIARIZATION_MIN_SILENCE_MS: int = 800
    DIARIZATION_PAUSE_THRESHOLD_DB: float = -40.0

    # Phase 7A Content Analysis Configuration
    ENABLE_CONTENT_ANALYSIS: bool = False
    CONTENT_ANALYSIS_BATCH_SIZE: int = 10
    MAX_CONTENT_ANALYSIS_CHARS: int = 8000
    CONTENT_ANALYSIS_MAX_RETRIES: int = 2

    # Phase 7A Chapter Generation Configuration
    ENABLE_CHAPTER_GENERATION: bool = True
    CHAPTER_MIN_PAUSE_SEC: float = 3.0
    CHAPTER_DISCONTINUITY_THRESHOLD: float = 0.3
    CHAPTER_MAX_COUNT: int = 20


    # API Server Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Streamlit Settings
    STREAMLIT_PORT: int = 8501

    @property
    def audio_dir(self) -> Path:
        return self.DATA_DIR / "audio"

    @property
    def transcript_dir(self) -> Path:
        return self.DATA_DIR / "transcripts"

    @property
    def alignment_dir(self) -> Path:
        return self.DATA_DIR / "alignments"

    @property
    def export_dir(self) -> Path:
        return self.DATA_DIR / "exports"

    @property
    def temp_dir(self) -> Path:
        return self.DATA_DIR / "temp"

    @property
    def db_dir(self) -> Path:
        return self.DATA_DIR / "db"

    @property
    def db_path(self) -> Path:
        return self.db_dir / "system.db"

    @property
    def bm25_dir(self) -> Path:
        return self.DATA_DIR / "bm25"

    @property
    def qdrant_dir(self) -> Path:
        return self.DATA_DIR / "qdrant"

    def ensure_directories(self) -> None:
        """Ensure all required runtime data directories exist."""
        for directory in [
            self.DATA_DIR,
            self.audio_dir,
            self.transcript_dir,
            self.alignment_dir,
            self.export_dir,
            self.temp_dir,
            self.db_dir,
            self.bm25_dir,
            self.qdrant_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_directories()

