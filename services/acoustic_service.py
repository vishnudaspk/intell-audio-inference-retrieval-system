"""
Acoustic Feature Service — V3 Phase 1G
Extracts pitch/F0, RMS energy, and spectral features per audio segment using librosa.
All extraction operates on numpy float32 arrays at 16kHz.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from utils.exceptions import AudioProcessingError
from utils.logger import logger

# Feature extraction constants
N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048
FMIN_HZ = 50.0
FMAX_HZ = 400.0  # Typical human voice F0 range


@dataclass
class AcousticFeatures:
    """Container for all acoustic features extracted from an audio segment."""

    # Pitch / F0
    f0_mean: Optional[float] = None
    f0_median: Optional[float] = None
    f0_min: Optional[float] = None
    f0_max: Optional[float] = None
    f0_std: Optional[float] = None
    f0_voiced_fraction: Optional[float] = None  # Fraction of frames that are voiced

    # Energy (RMS)
    rms_mean: Optional[float] = None
    rms_std: Optional[float] = None
    rms_max: Optional[float] = None

    # Spectral features
    spectral_centroid_mean: Optional[float] = None
    spectral_bandwidth_mean: Optional[float] = None
    spectral_rolloff_mean: Optional[float] = None
    zero_crossing_rate_mean: Optional[float] = None

    # MFCCs (means of first N_MFCC coefficients)
    mfcc_means: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON storage."""
        return {
            "f0_mean": self.f0_mean,
            "f0_median": self.f0_median,
            "f0_min": self.f0_min,
            "f0_max": self.f0_max,
            "f0_std": self.f0_std,
            "f0_voiced_fraction": self.f0_voiced_fraction,
            "rms_mean": self.rms_mean,
            "rms_std": self.rms_std,
            "rms_max": self.rms_max,
            "spectral_centroid_mean": self.spectral_centroid_mean,
            "spectral_bandwidth_mean": self.spectral_bandwidth_mean,
            "spectral_rolloff_mean": self.spectral_rolloff_mean,
            "zero_crossing_rate_mean": self.zero_crossing_rate_mean,
            "mfcc_means": self.mfcc_means,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AcousticFeatures":
        """Deserialize from a plain dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class AcousticFeatureService:
    """
    Service extracting pitch, energy, and spectral features from audio segments using librosa.

    All methods are stateless — no model loading required.
    """

    def __init__(self, sample_rate: int = 16_000):
        self.sr = sample_rate

    def extract_for_segment(
        self,
        wav_path: Path,
        start_sec: float,
        end_sec: float,
    ) -> AcousticFeatures:
        """
        Extract acoustic features for a single time-bounded segment.

        Args:
            wav_path: Normalized 16kHz mono WAV file.
            start_sec: Segment start in seconds.
            end_sec: Segment end in seconds.

        Returns:
            AcousticFeatures dataclass populated with feature values.
        """
        if not wav_path.exists():
            raise AudioProcessingError(f"WAV file not found for acoustic extraction: {wav_path}")

        duration = end_sec - start_sec
        if duration <= 0:
            logger.warning(f"Zero-length segment [{start_sec}–{end_sec}]s — returning empty features.")
            return AcousticFeatures()

        try:
            import soundfile as sf

            audio_np, sr = sf.read(str(wav_path), dtype="float32")
            if audio_np.ndim == 2:
                audio_np = audio_np.mean(axis=1)  # downmix to mono

            # Slice to segment
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            segment_np = audio_np[start_sample:end_sample]

            if len(segment_np) == 0:
                return AcousticFeatures()

            return self._extract_features(segment_np, sr)

        except AudioProcessingError:
            raise
        except Exception as exc:
            logger.error(f"Acoustic extraction failed for [{start_sec:.2f}–{end_sec:.2f}]s: {exc}")
            raise AudioProcessingError(f"Acoustic feature extraction error: {exc}") from exc

    def extract_for_array(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> AcousticFeatures:
        """
        Extract acoustic features from a raw numpy audio array.

        Args:
            audio: 1-D float32 audio array.
            sr: Sample rate of the audio.

        Returns:
            AcousticFeatures dataclass.
        """
        if len(audio) == 0:
            return AcousticFeatures()
        return self._extract_features(audio, sr)

    def _extract_features(self, audio: np.ndarray, sr: int) -> AcousticFeatures:
        """Core extraction logic using librosa."""
        import librosa

        features = AcousticFeatures()

        # ── Pitch / F0 via pyin ──────────────────────────────────────────
        try:
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=FMIN_HZ,
                fmax=FMAX_HZ,
                sr=sr,
                hop_length=HOP_LENGTH,
            )
            voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])

            if len(voiced_f0) > 0:
                features.f0_mean = float(np.mean(voiced_f0))
                features.f0_median = float(np.median(voiced_f0))
                features.f0_min = float(np.min(voiced_f0))
                features.f0_max = float(np.max(voiced_f0))
                features.f0_std = float(np.std(voiced_f0))

            if voiced_flag is not None and len(voiced_flag) > 0:
                features.f0_voiced_fraction = float(np.mean(voiced_flag.astype(float)))
        except Exception as exc:
            logger.debug(f"F0 extraction failed (likely segment too short): {exc}")

        # ── RMS Energy ──────────────────────────────────────────────────
        try:
            rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
            features.rms_mean = float(np.mean(rms))
            features.rms_std = float(np.std(rms))
            features.rms_max = float(np.max(rms))
        except Exception as exc:
            logger.debug(f"RMS extraction failed: {exc}")

        # ── Spectral Features ───────────────────────────────────────────
        try:
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
            features.spectral_centroid_mean = float(np.mean(centroid))
        except Exception as exc:
            logger.debug(f"Spectral centroid failed: {exc}")

        try:
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
            features.spectral_bandwidth_mean = float(np.mean(bandwidth))
        except Exception as exc:
            logger.debug(f"Spectral bandwidth failed: {exc}")

        try:
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
            features.spectral_rolloff_mean = float(np.mean(rolloff))
        except Exception as exc:
            logger.debug(f"Spectral rolloff failed: {exc}")

        try:
            zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)[0]
            features.zero_crossing_rate_mean = float(np.mean(zcr))
        except Exception as exc:
            logger.debug(f"ZCR failed: {exc}")

        # ── MFCCs ───────────────────────────────────────────────────────
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
            features.mfcc_means = [round(float(np.mean(mfccs[i])), 4) for i in range(N_MFCC)]
        except Exception as exc:
            logger.debug(f"MFCC extraction failed: {exc}")

        return features

    def extract_batch(
        self,
        wav_path: Path,
        segments: List[Tuple[float, float]],
    ) -> List[AcousticFeatures]:
        """
        Extract acoustic features for a list of (start_sec, end_sec) segments.
        Returns a list of AcousticFeatures in the same order.
        Short/failed segments return empty AcousticFeatures().
        """
        results: List[AcousticFeatures] = []
        for i, (start, end) in enumerate(segments):
            try:
                feat = self.extract_for_segment(wav_path, start, end)
                results.append(feat)
            except AudioProcessingError as exc:
                logger.warning(f"Acoustic extraction failed for segment {i} [{start:.2f}–{end:.2f}]s: {exc}")
                results.append(AcousticFeatures())
        return results
