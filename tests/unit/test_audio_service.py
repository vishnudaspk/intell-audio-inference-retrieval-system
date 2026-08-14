"""
Unit tests for AudioService validation and saving.
"""

from pathlib import Path

import pytest

from services.audio_service import AudioService
from utils.exceptions import AudioProcessingError


def test_audio_service_validation(tmp_path: Path):
    service = AudioService(data_dir=tmp_path)

    # Valid file
    service.validate_file("sample.mp3", 1024)

    # Invalid extension
    with pytest.raises(AudioProcessingError):
        service.validate_file("document.pdf", 1024)

    # File size too large
    with pytest.raises(AudioProcessingError):
        service.validate_file("huge.mp3", 200 * 1024 * 1024)


def test_audio_service_save_upload(tmp_path: Path):
    service = AudioService(data_dir=tmp_path)
    dummy_bytes = b"ID3\x03\x00\x00\x00"
    asset = service.save_uploaded_file(dummy_bytes, "test.mp3")

    assert asset.filename == "test.mp3"
    assert Path(asset.file_path).exists()
