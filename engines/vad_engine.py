"""
Silero VAD Engine — V3 Phase 1D
Implements VADEngine using Silero VAD (PyTorch-based, offline, no cloud dependencies).
Supports 8kHz and 16kHz audio. Outputs speech intervals with confidence scores.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from engines.base import VADEngine
from utils.exceptions import AudioProcessingError
from utils.logger import logger

# Silero VAD target sample rate (16kHz preferred)
SILERO_SAMPLE_RATE = 16_000
SILERO_CHUNK_SIZE = 512  # samples per chunk at 16kHz (32ms windows)


class SileroVADEngine(VADEngine):
    """
    Voice Activity Detection using Silero VAD.
    Model is loaded lazily on first use and cached for the lifetime of the instance.
    """

    def __init__(self):
        self._model = None
        self._get_speech_timestamps_fn = None
        self._utils = None

    def _load_model(self) -> None:
        """Lazily load Silero VAD model from torch hub."""
        if self._model is not None:
            return

        try:
            logger.info("Loading Silero VAD model from torch hub (silero-vad)...")
            from silero_vad import load_silero_vad, get_speech_timestamps

            self._model = load_silero_vad()
            self._model.eval()
            self._get_speech_timestamps_fn = get_speech_timestamps

            logger.info("Silero VAD model loaded successfully.")
        except Exception as exc:
            logger.error(f"Failed to load Silero VAD model: {exc}")
            raise AudioProcessingError(f"SileroVAD model load error: {exc}") from exc

    def is_available(self) -> bool:
        """Check if VAD model is loadable."""
        try:
            self._load_model()
            return self._model is not None
        except Exception:
            return False

    def detect_speech_segments(
        self,
        audio_path: Path,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 100,
    ) -> List[Tuple[float, float, float]]:
        """
        Detect speech segments in a 16kHz mono WAV file using Silero VAD.

        Args:
            audio_path: Path to a normalized 16kHz mono WAV file.
            threshold: Speech probability threshold (0.0 – 1.0). Default 0.5.
            min_speech_duration_ms: Minimum speech segment duration to keep.
            min_silence_duration_ms: Minimum silence gap to split segments.
            speech_pad_ms: Padding added around each detected speech segment.

        Returns:
            List of (start_sec, end_sec, confidence) tuples. Confidence is
            estimated as the mean Silero probability over the speech interval.
        """
        self._load_model()

        if not audio_path.exists():
            raise AudioProcessingError(f"VAD input WAV not found: {audio_path}")

        try:
            import soundfile as sf

            audio_np, sr = sf.read(str(audio_path), dtype="float32")
            # soundfile returns (samples,) mono or (samples, channels)
            if audio_np.ndim == 2:
                audio_np = audio_np.mean(axis=1)  # downmix to mono

            waveform = torch.from_numpy(audio_np).unsqueeze(0)  # (1, samples)

            if sr != SILERO_SAMPLE_RATE:
                import torchaudio.transforms as T

                resampler = T.Resample(orig_freq=sr, new_freq=SILERO_SAMPLE_RATE)
                waveform = resampler(waveform)
                sr = SILERO_SAMPLE_RATE

            # Silero expects 1-D float32 tensor
            audio_tensor = waveform.squeeze(0)

            timestamps = self._get_speech_timestamps_fn(
                audio_tensor,
                self._model,
                threshold=threshold,
                sampling_rate=sr,
                min_speech_duration_ms=min_speech_duration_ms,
                min_silence_duration_ms=min_silence_duration_ms,
                speech_pad_ms=speech_pad_ms,
                return_seconds=False,  # returns sample indices
            )

            segments: List[Tuple[float, float, float]] = []
            total_samples = audio_tensor.shape[0]

            for ts in timestamps:
                start_sample = ts["start"]
                end_sample = ts["end"]

                # Clamp to valid range
                start_sample = max(0, start_sample)
                end_sample = min(total_samples, end_sample)

                start_sec = start_sample / sr
                end_sec = end_sample / sr

                # Estimate confidence: mean VAD probability over the interval
                # (run a second pass over the window at chunk level)
                confidence = self._estimate_confidence(
                    audio_tensor, start_sample, end_sample, sr
                )

                segments.append((round(start_sec, 4), round(end_sec, 4), round(confidence, 4)))

            logger.info(
                f"Silero VAD detected {len(segments)} speech segment(s) in {audio_path.name}"
            )
            return segments

        except AudioProcessingError:
            raise
        except Exception as exc:
            logger.error(f"SileroVAD inference failed for {audio_path}: {exc}")
            raise AudioProcessingError(f"VAD detection error: {exc}") from exc

    def _estimate_confidence(
        self,
        audio_tensor: "torch.Tensor",
        start_sample: int,
        end_sample: int,
        sr: int,
    ) -> float:
        """Estimate mean speech confidence over a slice by running Silero chunk-by-chunk."""
        try:
            segment = audio_tensor[start_sample:end_sample]
            probs = []
            chunk_size = SILERO_CHUNK_SIZE

            # Reset model hidden state
            self._model.reset_states()

            for i in range(0, len(segment), chunk_size):
                chunk = segment[i : i + chunk_size]
                if len(chunk) < chunk_size:
                    # Pad last chunk
                    padding = torch.zeros(chunk_size - len(chunk))
                    chunk = torch.cat([chunk, padding])
                with torch.no_grad():
                    prob = self._model(chunk, sr).item()
                probs.append(prob)

            return float(np.mean(probs)) if probs else 0.5
        except Exception:
            return 0.5  # Fallback confidence if estimation fails
