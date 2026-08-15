"""
Audio Service — V3 Phase 1C
Handles media ingestion, validation, and normalization to 16kHz mono PCM WAV.
Supports: MP3, WAV, M4A, FLAC, OGG, MP4 (audio track extraction).
Uses soundfile + torchaudio/librosa for reliable, dependency-light normalization.
"""

import io
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torchaudio
import torchaudio.transforms as T

from config.settings import settings
from schemas.enums import SourceType
from schemas.models import AudioAsset
from utils.exceptions import AudioProcessingError
from utils.logger import logger

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4"}
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500 MB limit (accommodates video files)
TARGET_SAMPLE_RATE = 16_000
TARGET_CHANNELS = 1


class AudioService:
    """Service handling audio/video ingestion, validation, format normalization, and preview seeking."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.audio_dir = (data_dir or settings.DATA_DIR) / "audio"
        self.temp_dir = (data_dir or settings.DATA_DIR) / "temp"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_file(self, filename: str, file_size: int) -> None:
        """Validate filename extension and file size."""
        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise AudioProcessingError(
                f"Unsupported media format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            )
        if file_size > MAX_FILE_SIZE_BYTES:
            raise AudioProcessingError(
                f"File size ({file_size / 1024 / 1024:.1f} MB) exceeds maximum limit (500 MB)"
            )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def save_uploaded_file(self, file_bytes: bytes, original_filename: str) -> AudioAsset:
        """Persist raw upload bytes and return an AudioAsset record."""
        self.validate_file(original_filename, len(file_bytes))
        asset_id = str(uuid.uuid4())
        ext = Path(original_filename).suffix.lower() or ".mp3"
        raw_filename = f"{asset_id}_raw{ext}"
        raw_path = self.audio_dir / raw_filename

        try:
            with open(raw_path, "wb") as f:
                f.write(file_bytes)

            logger.info(f"Saved uploaded asset {asset_id} ({original_filename}) -> {raw_path}")

            return AudioAsset(
                id=asset_id,
                filename=original_filename,
                file_path=str(raw_path),
                format=ext.lstrip("."),
                source_type=SourceType.UPLOAD,
            )
        except Exception as exc:
            logger.error(f"Failed to save uploaded file: {exc}")
            raise AudioProcessingError(f"Failed to save audio file: {exc}") from exc

    # ------------------------------------------------------------------
    # Normalization — Core V3 Stage
    # ------------------------------------------------------------------

    def normalize_to_wav(self, asset: AudioAsset) -> Path:
        """
        Normalize any supported audio/video file to 16 kHz mono PCM WAV.

        Handles:
        - Resampling to TARGET_SAMPLE_RATE (16 kHz)
        - Downmixing to mono (averaging channels)
        - Float32 → int16 PCM conversion for broad compatibility
        - MP4 video files: extracts audio track via torchaudio's ffmpeg backend

        Returns the path to the normalized WAV file.
        """
        input_path = Path(asset.file_path)
        if not input_path.exists():
            raise AudioProcessingError(f"Source media file not found: {input_path}")

        wav_path = self.audio_dir / f"{asset.id}.wav"

        logger.info(f"Normalizing asset {asset.id} to 16kHz mono WAV -> {wav_path}")

        try:
            # torchaudio handles MP3, WAV, FLAC, OGG, M4A, MP4 (via ffmpeg backend)
            waveform, sample_rate = torchaudio.load(str(input_path))
            # waveform shape: (channels, num_samples)

            # Downmix to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if needed
            if sample_rate != TARGET_SAMPLE_RATE:
                resampler = T.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
                waveform = resampler(waveform)

            # Compute duration
            num_samples = waveform.shape[-1]
            duration_sec = num_samples / TARGET_SAMPLE_RATE
            asset.duration = round(duration_sec, 3)

            # Save as 16-bit PCM WAV via soundfile (avoids torchcodec dependency)
            audio_np = waveform.squeeze(0).numpy().astype(np.float32)
            sf.write(str(wav_path), audio_np, TARGET_SAMPLE_RATE, subtype="PCM_16")

            logger.info(
                f"Normalized asset {asset.id}: {duration_sec:.1f}s | "
                f"sample_rate={TARGET_SAMPLE_RATE} | mono | {wav_path}"
            )
            return wav_path

        except Exception as exc:
            logger.error(f"Failed to normalize asset {asset.id}: {exc}")
            raise AudioProcessingError(f"Audio normalization failed: {exc}") from exc

    def load_waveform(self, wav_path: Path) -> tuple:
        """
        Load a normalized WAV file and return (numpy_array, sample_rate).
        The array is 1-D float32 in [-1.0, 1.0] range.
        """
        if not wav_path.exists():
            raise AudioProcessingError(f"WAV file not found: {wav_path}")

        audio_np, sr = sf.read(str(wav_path), dtype="float32")
        if audio_np.ndim == 2:
            audio_np = audio_np.mean(axis=1)
        return audio_np, sr

    # ------------------------------------------------------------------
    # Preview seeking (retained for Streamlit UI)
    # ------------------------------------------------------------------

    def extract_audio_preview(self, audio_path: Path, start_time_sec: float) -> bytes:
        """Slice audio from start_time_sec to end of file as WAV bytes for UI seeking."""
        if not audio_path.exists():
            raise AudioProcessingError(f"Audio file for preview seek not found: {audio_path}")

        try:
            audio_np, sr = sf.read(str(audio_path), dtype="float32")
            if audio_np.ndim == 2:
                audio_np = audio_np.mean(axis=1)
            start_sample = int(start_time_sec * sr)
            preview_np = audio_np[start_sample:]

            buffer = io.BytesIO()
            sf.write(buffer, preview_np, sr, subtype="PCM_16", format="WAV")
            buffer.seek(0)
            return buffer.read()

        except Exception as exc:
            logger.error(f"Failed to seek audio preview at {start_time_sec}s: {exc}")
            raise AudioProcessingError(f"Audio seek error: {exc}") from exc

    # ------------------------------------------------------------------
    # Backward-compat shim (old callers that used convert_to_wav)
    # ------------------------------------------------------------------

    def convert_to_wav(self, audio_asset: AudioAsset) -> Path:
        """Backward-compatible alias for normalize_to_wav."""
        return self.normalize_to_wav(audio_asset)
