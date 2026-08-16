"""
Unit tests for AudioService validation and saving.
"""

from pathlib import Path

import pytest

from services.audio_service import AudioService
from utils.exceptions import AudioProcessingError


def test_audio_service_validation(tmp_path: Path):
    service = AudioService(data_dir=tmp_path)

    # Valid files (audio + video audio extraction)
    service.validate_file("sample.mp3", 1024)
    service.validate_file("video.mp4", 1024)
    service.validate_file("speech.wav", 1024)
    service.validate_file("track.flac", 1024)

    # Invalid extension
    with pytest.raises(AudioProcessingError):
        service.validate_file("document.pdf", 1024)

    # File size too large (exceeds 500 MB limit)
    with pytest.raises(AudioProcessingError):
        service.validate_file("huge.mp4", 600 * 1024 * 1024)


def test_audio_service_save_upload(tmp_path: Path):
    service = AudioService(data_dir=tmp_path)
    dummy_bytes = b"ID3\x03\x00\x00\x00"
    asset = service.save_uploaded_file(dummy_bytes, "test.mp3")

    assert asset.filename == "test.mp3"
    assert Path(asset.file_path).exists()


def test_audio_service_youtube_validation(tmp_path: Path):
    service = AudioService(data_dir=tmp_path)

    # Empty URL raises
    with pytest.raises(AudioProcessingError):
        service.download_youtube_audio("")

    with pytest.raises(AudioProcessingError):
        service.download_youtube_audio("   ")


def test_audio_service_youtube_download_mock(tmp_path: Path, monkeypatch):
    from unittest.mock import MagicMock
    import yt_dlp
    from schemas.enums import SourceType

    service = AudioService(data_dir=tmp_path)

    # Create dummy downloaded raw file
    class MockYoutubeDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def extract_info(self, url, download=True):
            # Create the file yt-dlp would have outputted
            outtmpl = self.opts["outtmpl"]
            out_prefix = outtmpl.replace("%(ext)s", "")
            raw_path = Path(out_prefix + "mp3")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(b"dummy mp3 audio data")
            return {"title": "Test YouTube Speech Video"}

    monkeypatch.setattr(yt_dlp, "YoutubeDL", MockYoutubeDL)

    asset = service.download_youtube_audio("https://www.youtube.com/watch?v=mocktest123")
    assert asset.filename == "Test YouTube Speech Video.mp3"
    assert asset.source_type == SourceType.YOUTUBE
    assert asset.format == "mp3"
    assert Path(asset.file_path).exists()
