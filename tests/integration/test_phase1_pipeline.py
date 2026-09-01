"""
Integration test for V3 Phase 1 pipeline — Phase 1J
Tests the full AudioWorker pipeline end-to-end using a synthetic WAV file.
All heavy ML models (Whisper, SpeechBrain, Silero) are mocked to avoid
requiring GPU/network access during CI runs.
"""

import json
import numpy as np
import pytest
import torch
import torchaudio
from pathlib import Path
from unittest.mock import MagicMock, patch

from schemas.enums import JobStatus, LanguageCode
from schemas.models import AudioAsset, AudioSegment
from services.acoustic_service import AcousticFeatures
from workers.audio_worker import AudioWorker


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_wav(tmp_path) -> Path:
    """Create a 3-second 440Hz sine wave WAV file at 16kHz."""
    import numpy as np
    import soundfile as sf

    sr = 16000
    t = np.linspace(0, 3.0, int(sr * 3), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    wav_path = tmp_path / "test_audio.wav"
    sf.write(str(wav_path), audio, sr)
    return wav_path



@pytest.fixture
def audio_asset(tmp_path, synthetic_wav) -> AudioAsset:
    return AudioAsset(
        id="test-asset-001",
        filename="test_audio.wav",
        file_path=str(synthetic_wav),
        format="wav",
        duration=3.0,
    )


# ── Mock helpers ──────────────────────────────────────────────────────────────

def _mock_vad_service(segments=None):
    """Returns a VADService mock that yields synthetic speech intervals."""
    svc = MagicMock()
    _segs = segments or [(0.0, 1.5, 0.9), (1.8, 3.0, 0.85)]
    svc.detect_segments.return_value = _segs
    svc.filter_short_segments.return_value = _segs
    svc.merge_close_segments.return_value = _segs
    return svc


def _mock_transcription_service():
    """Returns a TranscriptionService mock that yields a synthetic transcript."""
    from schemas.models import Transcript, TranscriptSegment, TranscriptWord

    svc = MagicMock()
    words1 = [TranscriptWord(word="Hello", start=0.1, end=0.5, confidence=0.95)]
    words2 = [TranscriptWord(word="world", start=1.9, end=2.3, confidence=0.92)]
    seg1 = TranscriptSegment(sequence_order=0, start=0.0, end=1.5, text="Hello", words=words1)
    seg2 = TranscriptSegment(sequence_order=1, start=1.8, end=3.0, text="world", words=words2)
    transcript = Transcript(
        id="t-001",
        audio_id="test-asset-001",
        text="Hello world",
        language=LanguageCode.ENGLISH,
        duration=3.0,
        segments=[seg1, seg2],
        words=words1 + words2,
    )
    svc.transcribe_audio.return_value = transcript
    return svc


def _mock_embedding_service():
    """Returns a SpeakerEmbeddingService mock yielding random 192-dim embeddings."""
    svc = MagicMock()
    svc.embed_segments.return_value = [
        np.random.rand(192).astype(np.float32),
        np.random.rand(192).astype(np.float32),
    ]
    svc.cluster_segments.return_value = ["Speaker 1", "Speaker 1"]
    svc.diarize_audio.return_value = (
        [
            {
                "start_sec": 0.0,
                "end_sec": 1.5,
                "duration_sec": 1.5,
                "vad_confidence": 0.9,
                "text": "Hello",
                "words": [{"word": "Hello", "start_time": 0.1, "end_time": 0.5, "confidence": 0.95}],
                "speaker_label": "Speaker 1",
            },
            {
                "start_sec": 1.8,
                "end_sec": 3.0,
                "duration_sec": 1.2,
                "vad_confidence": 0.85,
                "text": "world",
                "words": [{"word": "world", "start_time": 1.9, "end_time": 2.3, "confidence": 0.92}],
                "speaker_label": "Speaker 1",
            },
        ],
        {
            "num_windows": 2,
            "num_embeddings": 2,
            "embedding_dim": 192,
            "estimated_speakers": 1,
            "distinct_speakers": 1,
            "cluster_sizes": {"Speaker 1": 2},
            "mean_cosine_sim": 0.9,
        },
    )
    return svc


def _mock_acoustic_service():
    """Returns an AcousticFeatureService mock yielding synthetic features."""
    svc = MagicMock()
    feat = AcousticFeatures(f0_mean=220.0, rms_mean=0.05, mfcc_means=[0.1] * 13)
    svc.extract_batch.return_value = [feat, feat]
    return svc


def _mock_repository():
    """Returns a SQLiteRepository mock with no-op persistence methods."""
    repo = MagicMock()
    repo.save_job.return_value = None
    repo.save_audio_asset.return_value = None
    repo.save_transcript.return_value = None
    repo.save_alignment_words.return_value = None
    repo.save_audio_segments.return_value = None
    return repo


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPhase1Pipeline:
    def _make_worker(self, audio_service_path="services.audio_service.AudioService.normalize_to_wav"):
        """Build a fully-mocked AudioWorker."""
        return AudioWorker(
            repository=_mock_repository(),
            vad_service=_mock_vad_service(),
            transcription_service=_mock_transcription_service(),
            speaker_embedding_service=_mock_embedding_service(),
            acoustic_service=_mock_acoustic_service(),
            extract_acoustics=True,
            extract_embeddings=True,
        )

    def test_pipeline_completes_successfully(self, audio_asset, synthetic_wav):
        worker = self._make_worker()
        # Patch audio_service to skip actual normalization (file already correct)
        worker.audio_service = MagicMock()
        worker.audio_service.normalize_to_wav.return_value = synthetic_wav

        job = worker.process_asset(audio_asset)

        assert job.status == JobStatus.COMPLETED
        assert job.audio_id == audio_asset.id

    def test_pipeline_produces_segments(self, audio_asset, synthetic_wav):
        worker = self._make_worker()
        worker.audio_service = MagicMock()
        worker.audio_service.normalize_to_wav.return_value = synthetic_wav

        job = worker.process_asset(audio_asset)

        # Verify repo was called with 2 segments (from mock VAD)
        repo = worker.repo
        repo.save_audio_segments.assert_called_once()
        segments_arg = repo.save_audio_segments.call_args[0][1]
        assert len(segments_arg) == 2

    def test_segment_has_correct_structure(self, audio_asset, synthetic_wav):
        worker = self._make_worker()
        worker.audio_service = MagicMock()
        worker.audio_service.normalize_to_wav.return_value = synthetic_wav

        worker.process_asset(audio_asset)

        segments = worker.repo.save_audio_segments.call_args[0][1]
        seg: AudioSegment = segments[0]

        assert seg.audio_id == audio_asset.id
        assert seg.start_sec == 0.0
        assert seg.end_sec == 1.5
        assert seg.duration_sec == 1.5
        assert seg.vad_confidence == 0.9
        assert seg.speaker_embedding is not None
        assert len(seg.speaker_embedding) == 192
        assert seg.acoustic_features is not None
        assert seg.acoustic_features["f0_mean"] == pytest.approx(220.0)

    def test_pipeline_records_timings(self, audio_asset, synthetic_wav):
        worker = self._make_worker()
        worker.audio_service = MagicMock()
        worker.audio_service.normalize_to_wav.return_value = synthetic_wav

        job = worker.process_asset(audio_asset)

        assert "total_sec" in job.timings
        assert "vad_sec" in job.timings
        assert "asr_sec" in job.timings
        assert job.timings["segments_produced"] == 2

    def test_pipeline_fails_gracefully_on_normalization_error(self, audio_asset):
        from utils.exceptions import IntellAudioError, AudioProcessingError

        worker = AudioWorker(
            repository=_mock_repository(),
            vad_service=_mock_vad_service(),
            transcription_service=_mock_transcription_service(),
            speaker_embedding_service=_mock_embedding_service(),
            acoustic_service=_mock_acoustic_service(),
        )
        worker.audio_service = MagicMock()
        worker.audio_service.normalize_to_wav.side_effect = AudioProcessingError("Normalization failed")

        with pytest.raises(IntellAudioError):
            worker.process_asset(audio_asset)

        # Job should be saved as FAILED
        repo = worker.repo
        failed_calls = [
            call for call in repo.save_job.call_args_list
            if call[0][0].status == JobStatus.FAILED
        ]
        assert len(failed_calls) >= 1

    def test_pipeline_no_speech_completes_empty(self, audio_asset, synthetic_wav):
        """Pipeline should complete gracefully when VAD detects no speech."""
        worker = AudioWorker(
            repository=_mock_repository(),
            vad_service=_mock_vad_service(segments=[]),
            transcription_service=_mock_transcription_service(),
            speaker_embedding_service=_mock_embedding_service(),
            acoustic_service=_mock_acoustic_service(),
        )
        worker.audio_service = MagicMock()
        worker.audio_service.normalize_to_wav.return_value = synthetic_wav

        job = worker.process_asset(audio_asset)

        assert job.status == JobStatus.COMPLETED

    def test_pipeline_with_embeddings_disabled(self, audio_asset, synthetic_wav):
        worker = AudioWorker(
            repository=_mock_repository(),
            vad_service=_mock_vad_service(),
            transcription_service=_mock_transcription_service(),
            speaker_embedding_service=_mock_embedding_service(),
            acoustic_service=_mock_acoustic_service(),
            extract_embeddings=False,
            extract_acoustics=False,
        )
        worker.audio_service = MagicMock()
        worker.audio_service.normalize_to_wav.return_value = synthetic_wav

        job = worker.process_asset(audio_asset)
        assert job.status == JobStatus.COMPLETED

        segments = worker.repo.save_audio_segments.call_args[0][1]
        for seg in segments:
            assert seg.speaker_embedding is None
            assert seg.acoustic_features is None
