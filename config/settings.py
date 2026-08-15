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

    # Database URL
    DATABASE_URL: Optional[str] = None

    # V3 Audio Processing Settings
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1

    # V3 ASR (Whisper) Configuration
    ASR_ENGINE: str = "whisper"
    WHISPER_MODEL_SIZE: str = "base.en"
    WHISPER_DEVICE: str = "auto"  # "cuda", "cpu", or "auto"
    WHISPER_COMPUTE_TYPE: str = "auto"  # "float16", "int8", "float32", or "auto"
    WHISPER_BEAM_SIZE: int = 5
    WHISPER_VAD_FILTER: bool = False  # Set False since we have our own dedicated VAD layer

    # V3 VAD Configuration
    VAD_ENGINE: str = "silero"
    VAD_THRESHOLD: float = 0.5
    VAD_MIN_SPEECH_DURATION_MS: int = 250
    VAD_MIN_SILENCE_DURATION_MS: int = 300
    VAD_SPEECH_PAD_MS: int = 100

    # V3 Speaker Representation Configuration
    SPEAKER_EMBEDDING_ENGINE: str = "speechbrain"
    SPEAKER_EMBEDDING_MODEL: str = "speechbrain/spkrec-ecapa-voxceleb"
    SPEAKER_EMBEDDING_DEVICE: str = "auto"  # "cuda", "cpu", or "auto"
    SPEAKER_MIN_SEGMENT_DURATION_SEC: float = 0.5

    # V3 Acoustic Analysis Configuration
    EXTRACT_ACOUSTICS: bool = True
    ACOUSTIC_SAMPLE_RATE: int = 16000

    # Retrieval Engine Selection
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

    @property
    def models_dir(self) -> Path:
        return self.DATA_DIR / "models"

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
            self.models_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_directories()
