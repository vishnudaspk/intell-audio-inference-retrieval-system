"""
Unit tests for configuration loading and directory structure initialization.
"""

from pathlib import Path

from config.settings import Settings


def test_settings_default_initialization(tmp_path: Path):
    custom_data_dir = tmp_path / "custom_data"
    settings = Settings(DATA_DIR=custom_data_dir)
    settings.ensure_directories()

    assert settings.audio_dir == custom_data_dir / "audio"
    assert settings.db_dir == custom_data_dir / "db"
    assert settings.audio_dir.exists()
    assert settings.db_dir.exists()
    assert settings.ASR_ENGINE == "pocketsphinx"
