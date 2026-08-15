"""
Unit tests for Speaker Embedding Service — V3 Phase 1J
Tests zero embedding for short segments, output shape, normalization behavior.
All model loading is mocked.
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from services.speaker_embedding_service import (
    SpeakerEmbeddingService,
    ECAPA_EMBEDDING_DIM,
    MIN_SEGMENT_DURATION_SEC,
)


class TestSpeakerEmbeddingService:
    def _make_mock_model(self):
        """Create a mock SpeechBrain model that returns a synthetic embedding."""
        import torch

        mock_model = MagicMock()
        # Simulate encode_batch output: (1, 1, 192)
        fake_embedding = torch.rand(1, 1, ECAPA_EMBEDDING_DIM)
        mock_model.encode_batch.return_value = fake_embedding
        mock_model.eval.return_value = None
        return mock_model

    def test_short_segment_returns_zero_embedding(self, tmp_path):
        service = SpeakerEmbeddingService()
        service._model = MagicMock()  # Mark as loaded
        service._resolved_device = "cpu"

        # Segment shorter than threshold
        short_dur = MIN_SEGMENT_DURATION_SEC * 0.5
        wav = tmp_path / "dummy.wav"
        wav.touch()

        emb = service.embed_segment(wav, start_sec=0.0, end_sec=short_dur)

        assert emb.shape == (ECAPA_EMBEDDING_DIM,)
        assert np.all(emb == 0), "Short segment should return zero embedding"

    def test_embedding_is_l2_normalized(self, tmp_path):
        """Verify returned embedding has unit L2 norm (within tolerance)."""
        import numpy as np
        import soundfile as sf

        service = SpeakerEmbeddingService()
        service._resolved_device = "cpu"

        mock_model = self._make_mock_model()
        service._model = mock_model

        # Create a real 1-second sine wav for slicing
        sr = 16000
        duration_samples = sr * 2
        t = np.linspace(0, 2 * np.pi * 440, duration_samples)
        audio_np = np.sin(t).astype(np.float32)
        wav_path = tmp_path / "sine.wav"
        sf.write(str(wav_path), audio_np, sr)

        emb = service.embed_segment(wav_path, start_sec=0.0, end_sec=1.0)

        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.01, f"Embedding norm should be ≈1.0, got {norm:.4f}"

    def test_embed_segments_returns_correct_count(self, tmp_path):
        """embed_segments should return one embedding per input segment."""
        import numpy as np
        import soundfile as sf

        service = SpeakerEmbeddingService()
        service._resolved_device = "cpu"
        service._model = self._make_mock_model()

        sr = 16000
        silence = np.zeros(sr * 5, dtype=np.float32)
        wav_path = tmp_path / "silent.wav"
        sf.write(str(wav_path), silence, sr)

        segments = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
        embeddings = service.embed_segments(wav_path, segments)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert emb.shape == (ECAPA_EMBEDDING_DIM,)

    def test_speaker_embedding_service_is_available(self):
        """SpeechBrain ECAPA-TDNN should be available and loadable without lazy-module errors."""
        service = SpeakerEmbeddingService()
        assert service.is_available() is True

    def test_cluster_segments_speaker_labeling(self):
        """Verify deterministic clustering assigns Speaker 1, Speaker 2, etc. correctly."""
        service = SpeakerEmbeddingService()
        # Create synthetic normalized embeddings
        e1 = np.zeros(ECAPA_EMBEDDING_DIM, dtype=np.float32)
        e1[0] = 1.0  # Speaker 1 base
        e2 = np.zeros(ECAPA_EMBEDDING_DIM, dtype=np.float32)
        e2[0] = 0.95
        e2[1] = 0.31  # Cosine sim with e1 ≈ 0.95 (same speaker)
        e2 /= np.linalg.norm(e2)

        e3 = np.zeros(ECAPA_EMBEDDING_DIM, dtype=np.float32)
        e3[50] = 1.0  # Speaker 2 base (orthogonal, sim ≈ 0.0)

        e_zero = np.zeros(ECAPA_EMBEDDING_DIM, dtype=np.float32)

        labels = service.cluster_segments([e1, e2, e3, e_zero, None])
        assert labels == ["Speaker 1", "Speaker 1", "Speaker 2", None, None]
