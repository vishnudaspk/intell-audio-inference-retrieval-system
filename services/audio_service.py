"""
Audio Service managing file ingestion, validation, WAV conversion, and segment seeking.
"""

import io
import uuid
from pathlib import Path
from typing import Optional

from moviepy.editor import AudioFileClip
from pydub import AudioSegment
from pytube import YouTube

from config.settings import settings
from schemas.enums import SourceType
from schemas.models import AudioAsset
from utils.exceptions import AudioProcessingError
from utils.logger import logger

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB limit


class AudioService:
    """Service handling audio file validation, YouTube download, format conversion, and timestamp seeking."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.audio_dir = (data_dir or settings.DATA_DIR) / "audio"
        self.temp_dir = (data_dir or settings.DATA_DIR) / "temp"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def validate_file(self, filename: str, file_size: int) -> None:
        """Validate filename extension and file size."""
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise AudioProcessingError(
                f"Unsupported audio format '{ext}'. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        if file_size > MAX_FILE_SIZE_BYTES:
            raise AudioProcessingError(f"Audio file size ({file_size} bytes) exceeds maximum limit (100 MB)")

    def save_uploaded_file(self, file_bytes: bytes, original_filename: str) -> AudioAsset:
        """Save uploaded audio bytes under a unique asset ID."""
        self.validate_file(original_filename, len(file_bytes))
        asset_id = str(uuid.uuid4())
        ext = Path(original_filename).suffix.lower() or ".mp3"
        target_filename = f"{asset_id}{ext}"
        target_path = self.audio_dir / target_filename

        try:
            with open(target_path, "wb") as f:
                f.write(file_bytes)

            logger.info(f"Saved uploaded audio asset {asset_id} ({original_filename}) to {target_path}")

            return AudioAsset(
                id=asset_id,
                filename=original_filename,
                file_path=str(target_path),
                format=ext.lstrip("."),
                source_type=SourceType.UPLOAD,
            )
        except Exception as exc:
            logger.error(f"Failed to save uploaded audio file: {exc}")
            raise AudioProcessingError(f"Failed to save audio file: {exc}") from exc

    def download_youtube_audio(self, youtube_url: str) -> AudioAsset:
        """Download lowest bitrate audio stream from a YouTube video URL."""
        if not youtube_url or not youtube_url.strip():
            raise AudioProcessingError("YouTube URL must not be empty")

        asset_id = str(uuid.uuid4())
        target_filename = f"{asset_id}.mp3"
        target_path = self.audio_dir / target_filename

        try:
            logger.info(f"Downloading YouTube audio from {youtube_url} for asset {asset_id}")
            yt = YouTube(youtube_url.strip())
            audio_streams = yt.streams.filter(only_audio=True)
            sorted_streams = sorted(audio_streams, key=lambda s: s.bitrate)

            if not sorted_streams:
                raise AudioProcessingError("No suitable audio stream found in YouTube link")

            best_audio = sorted_streams[0]
            best_audio.download(output_path=str(self.audio_dir), filename=target_filename)

            logger.info(f"Successfully downloaded YouTube audio for asset {asset_id}")

            return AudioAsset(
                id=asset_id,
                filename=f"youtube_{yt.video_id or asset_id}.mp3",
                file_path=str(target_path),
                format="mp3",
                source_type=SourceType.YOUTUBE,
            )
        except Exception as exc:
            logger.error(f"YouTube audio download failed for {youtube_url}: {exc}")
            raise AudioProcessingError(f"YouTube download failed: {exc}") from exc

    def convert_to_wav(self, audio_asset: AudioAsset) -> Path:
        """Convert input audio asset to 16kHz WAV format required by PocketSphinx & Gentle."""
        input_path = Path(audio_asset.file_path)
        if not input_path.exists():
            raise AudioProcessingError(f"Source audio file not found: {input_path}")

        wav_filename = f"{audio_asset.id}.wav"
        wav_path = self.audio_dir / wav_filename

        logger.info(f"Converting audio {audio_asset.id} to WAV standard at {wav_path}")

        clip = None
        try:
            clip = AudioFileClip(str(input_path))
            clip.write_audiofile(str(wav_path), verbose=False, logger=None)

            # Update duration on asset
            audio_asset.duration = clip.duration
            return wav_path

        except Exception as exc:
            logger.error(f"Failed to convert audio {audio_asset.id} to WAV: {exc}")
            raise AudioProcessingError(f"Audio conversion failed: {exc}") from exc
        finally:
            if clip is not None:
                try:
                    clip.close()
                except Exception:
                    pass

    def extract_audio_preview(self, audio_path: Path, start_time_sec: float) -> bytes:
        """Slice audio from start_time_sec to end of file as WAV bytes for UI seeking."""
        if not audio_path.exists():
            raise AudioProcessingError(f"Audio file for preview seek not found: {audio_path}")

        try:
            audio_seg = AudioSegment.from_file(str(audio_path))
            start_ms = int(float(start_time_sec) * 1000)
            preview_seg = audio_seg[start_ms:]

            output_buffer = io.BytesIO()
            preview_seg.export(output_buffer, format="wav")
            return output_buffer.getvalue()

        except Exception as exc:
            logger.error(f"Failed to seek audio preview at timestamp {start_time_sec}s: {exc}")
            raise AudioProcessingError(f"Audio seek error: {exc}") from exc
