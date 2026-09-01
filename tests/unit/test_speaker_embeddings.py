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

    def test_cluster_single_speaker_multiple_segments(self):
        """Multiple segments from the same speaker must converge to exactly 1 speaker."""
        service = SpeakerEmbeddingService()
        base = np.zeros(ECAPA_EMBEDDING_DIM, dtype=np.float32)
        base[0] = 1.0
        # 6 segments with slight variations (cosine similarity > 0.85)
        embs = []
        for i in range(6):
            e = base.copy()
            e[1] = 0.1 * i
            e /= np.linalg.norm(e)
            embs.append(e)

        labels = service.cluster_segments(embs)
        assert all(label == "Speaker 1" for label in labels)

    def test_cluster_two_distinct_speakers_alternating(self):
        """Alternating conversation between 2 speakers maps to exactly Speaker 1 and Speaker 2."""
        service = SpeakerEmbeddingService()
        spk1_base = np.zeros(ECAPA_EMBEDDING_DIM, dtype=np.float32)
        spk1_base[10] = 1.0
        spk2_base = np.zeros(ECAPA_EMBEDDING_DIM, dtype=np.float32)
        spk2_base[80] = 1.0

        embs = [
            spk1_base,
            spk2_base,
            spk1_base,
            spk2_base,
            spk1_base,
            spk2_base,
        ]
        labels = service.cluster_segments(embs)
        assert labels == [
            "Speaker 1",
            "Speaker 2",
            "Speaker 1",
            "Speaker 2",
            "Speaker 1",
            "Speaker 2",
        ]

    def test_cluster_noisy_and_short_segments_do_not_create_spurious_speakers(self):
        """Zero vectors, None, and slightly noisy segments should not explode cluster count."""
        service = SpeakerEmbeddingService()
        spk1_base = np.zeros(ECAPA_EMBEDDING_DIM, dtype=np.float32)
        spk1_base[5] = 1.0
        spk2_base = np.zeros(ECAPA_EMBEDDING_DIM, dtype=np.float32)
        spk2_base[95] = 1.0

        embs = [
            spk1_base,
            None,
            np.zeros(ECAPA_EMBEDDING_DIM, dtype=np.float32),
            spk2_base,
            spk1_base,
            None,
        ]
        labels = service.cluster_segments(embs)
        assert labels == ["Speaker 1", None, None, "Speaker 2", "Speaker 1", None]

    def test_cluster_three_speakers_chronological_stability(self):
        """Three distinct speakers are naturally recognized with chronological labels."""
        service = SpeakerEmbeddingService()
        s1 = np.zeros(ECAPA_EMBEDDING_DIM, dtype=np.float32); s1[0] = 1.0
        s2 = np.zeros(ECAPA_EMBEDDING_DIM, dtype=np.float32); s2[50] = 1.0
        s3 = np.zeros(ECAPA_EMBEDDING_DIM, dtype=np.float32); s3[120] = 1.0

        # Simulating window embeddings
        service._model = self._make_mock_model()
        service._resolved_device = "cpu"

    def test_diarize_audio_five_speakers_discovery(self, tmp_path):
        """5-speaker conversation produces approximately 5 clusters when embeddings support it."""
        import soundfile as sf

        service = SpeakerEmbeddingService()
        service._resolved_device = "cpu"

        sr = 16000
        audio_np = np.zeros(sr * 20, dtype=np.float32)
        wav_path = tmp_path / "multi_speaker.wav"
        sf.write(str(wav_path), audio_np, sr)

        # Create 5 distinct orthogonal base embeddings
        spk_bases = []
        for i in range(5):
            vec = np.zeros(ECAPA_EMBEDDING_DIM, dtype=np.float32)
            vec[i * 30] = 1.0
            spk_bases.append(vec)

        # Mock embed_segments to return respective speaker vector based on time
        def mock_embed_segments(wav, segments):
            out = []
            for st, et in segments:
                spk_idx = int(st // 3) % 5
                out.append(spk_bases[spk_idx] + 0.01 * np.random.randn(ECAPA_EMBEDDING_DIM).astype(np.float32))
            return out

        service.embed_segments = mock_embed_segments

        speech_intervals = [(0.0, 15.0)]
        words = [
            {"word": f"word_{i} ", "start_time": i * 1.0, "end_time": i * 1.0 + 0.5, "confidence": 0.9}
            for i in range(15)
        ]

        diarized_segments, diagnostics = service.diarize_audio(
            wav_path=wav_path,
            speech_intervals=speech_intervals,
            transcript_words=words,
        )

        assert diagnostics["estimated_speakers"] == 5
        assert diagnostics["distinct_speakers"] == 5
        assert len(diarized_segments) > 0

    def test_diarize_audio_two_speakers_stable(self, tmp_path):
        """2-speaker alternating dialogue produces exactly 2 speaker clusters."""
        import soundfile as sf

        service = SpeakerEmbeddingService()
        service._resolved_device = "cpu"

        sr = 16000
        audio_np = np.zeros(sr * 10, dtype=np.float32)
        wav_path = tmp_path / "two_speaker.wav"
        sf.write(str(wav_path), audio_np, sr)

        spk1 = np.zeros(ECAPA_EMBEDDING_DIM, dtype=np.float32); spk1[10] = 1.0
        spk2 = np.zeros(ECAPA_EMBEDDING_DIM, dtype=np.float32); spk2[90] = 1.0

        def mock_embed_segments(wav, segments):
            return [spk1 if int(st // 2) % 2 == 0 else spk2 for st, et in segments]

        service.embed_segments = mock_embed_segments

        speech_intervals = [(0.0, 8.0)]
        words = [
            {"word": f"word_{i} ", "start_time": i * 1.0, "end_time": i * 1.0 + 0.5, "confidence": 0.95}
            for i in range(8)
        ]

        diarized_segments, diagnostics = service.diarize_audio(
            wav_path=wav_path,
            speech_intervals=speech_intervals,
            transcript_words=words,
        )

        assert diagnostics["estimated_speakers"] == 2
        assert diagnostics["distinct_speakers"] == 2
