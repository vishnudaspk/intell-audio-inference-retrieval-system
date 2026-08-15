"""
Unit tests for Acoustic Feature Service — V3 Phase 1J
Tests feature extraction on synthetic audio arrays without real files.
"""

import numpy as np
import pytest

from services.acoustic_service import AcousticFeatureService, AcousticFeatures, N_MFCC


class TestAcousticFeatures:
    def test_to_dict_from_dict_roundtrip(self):
        features = AcousticFeatures(
            f0_mean=220.0,
            rms_mean=0.05,
            mfcc_means=[1.0] * N_MFCC,
        )
        d = features.to_dict()
        restored = AcousticFeatures.from_dict(d)
        assert restored.f0_mean == 220.0
        assert restored.rms_mean == 0.05
        assert len(restored.mfcc_means) == N_MFCC

    def test_empty_features_to_dict(self):
        features = AcousticFeatures()
        d = features.to_dict()
        assert d["f0_mean"] is None
        assert d["rms_mean"] is None
        assert d["mfcc_means"] == []


class TestAcousticFeatureService:
    """Tests using synthetic audio arrays to avoid real file I/O."""

    def setup_method(self):
        self.sr = 16000
        self.service = AcousticFeatureService(sample_rate=self.sr)

    def _sine_wave(self, freq_hz: float, duration_sec: float) -> np.ndarray:
        """Generate a pure sine wave for testing."""
        t = np.linspace(0, duration_sec, int(self.sr * duration_sec), endpoint=False)
        return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)

    def test_rms_extracted_for_sine_wave(self):
        audio = self._sine_wave(440, 2.0)
        features = self.service.extract_for_array(audio, self.sr)
        # RMS of sin(x) ≈ 1/sqrt(2) ≈ 0.707
        assert features.rms_mean is not None
        assert 0.6 < features.rms_mean < 0.8, f"Unexpected RMS: {features.rms_mean}"

    def test_mfccs_extracted(self):
        audio = self._sine_wave(440, 2.0)
        features = self.service.extract_for_array(audio, self.sr)
        assert len(features.mfcc_means) == N_MFCC

    def test_spectral_centroid_nonzero(self):
        audio = self._sine_wave(1000, 1.0)  # 1kHz sine
        features = self.service.extract_for_array(audio, self.sr)
        assert features.spectral_centroid_mean is not None
        assert features.spectral_centroid_mean > 0

    def test_empty_audio_returns_empty_features(self):
        features = self.service.extract_for_array(np.array([], dtype=np.float32), self.sr)
        assert features.f0_mean is None
        assert features.rms_mean is None

    def test_zero_length_segment_returns_empty(self, tmp_path):
        import numpy as np
        import soundfile as sf

        silence = np.zeros(self.sr * 3, dtype=np.float32)
        wav_path = tmp_path / "silent.wav"
        sf.write(str(wav_path), silence, self.sr)

        features = self.service.extract_for_segment(wav_path, start_sec=1.0, end_sec=1.0)
        assert features.f0_mean is None

    def test_rms_silent_audio_near_zero(self):
        silence = np.zeros(self.sr * 2, dtype=np.float32)
        features = self.service.extract_for_array(silence, self.sr)
        if features.rms_mean is not None:
            assert features.rms_mean < 1e-4

    def test_batch_extract_correct_count(self, tmp_path):
        import soundfile as sf

        sr = self.sr
        audio = self._sine_wave(440, 5.0)
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), audio, sr)

        segments = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
        results = self.service.extract_batch(wav_path, segments)

        assert len(results) == 3
        for feat in results:
            assert isinstance(feat, AcousticFeatures)
