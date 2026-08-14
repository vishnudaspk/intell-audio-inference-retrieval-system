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
        ]:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_directories()
