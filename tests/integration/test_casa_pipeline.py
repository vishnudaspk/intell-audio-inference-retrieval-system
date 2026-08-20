"""
Integration tests for the V3.2 CASA pipeline.
Uses the same mocked AudioWorker pattern established in test_phase1_pipeline.py.

Key verifications:
- AudioSegment objects produced by the pipeline have speaker_confidence,
  attribution_decision, attribution_evidence, and provisional fields populated.
- Disabling CASA (apply_casa=False) leaves those fields as None.
- The V3.1 baseline fields (speaker_label, speaker_embedding, acoustic_features)
  are completely unaffected.
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from schemas.enums import JobStatus, LanguageCode
from schemas.models import AudioAsset, AudioSegment
from services.acoustic_service import AcousticFeatures
from workers.audio_worker import AudioWorker


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_wav(tmp_path) -> Path:
    import soundfile as sf
    sr = 16000
    t = np.linspace(0, 3.0, int(sr * 3), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    wav_path = tmp_path / "test_casa.wav"
    sf.write(str(wav_path), audio, sr)
    return wav_path


@pytest.fixture
def audio_asset(tmp_path, synthetic_wav) -> AudioAsset:
    return AudioAsset(
        id="casa-test-001",
        filename="test_casa.wav",
        file_path=str(synthetic_wav),
        format="wav",
        duration=3.0,
    )


# ── Mock helpers ──────────────────────────────────────────────────────────────

def _make_vad():
    svc = MagicMock()
    segs = [(0.0, 1.5, 0.9), (1.8, 3.0, 0.85)]
    svc.detect_segments.return_value = segs
    svc.filter_short_segments.return_value = segs
    svc.merge_close_segments.return_value = segs
    return svc


def _make_transcription():
    from schemas.models import Transcript, TranscriptSegment, TranscriptWord
    svc = MagicMock()
    words1 = [TranscriptWord(word="What", start=0.1, end=0.4, confidence=0.95),
              TranscriptWord(word=" is", start=0.4, end=0.6, confidence=0.93),
              TranscriptWord(word=" your", start=0.6, end=0.9, confidence=0.94),
              TranscriptWord(word=" name?", start=0.9, end=1.4, confidence=0.96)]
    words2 = [TranscriptWord(word="My", start=1.9, end=2.1, confidence=0.92),
              TranscriptWord(word=" name", start=2.1, end=2.5, confidence=0.90),
              TranscriptWord(word=" is", start=2.5, end=2.7, confidence=0.91),
              TranscriptWord(word=" Jeff.", start=2.7, end=3.0, confidence=0.95)]
    seg1 = TranscriptSegment(sequence_order=0, start=0.0, end=1.5,
                             text="What is your name?", words=words1)
    seg2 = TranscriptSegment(sequence_order=1, start=1.8, end=3.0,
                             text="My name is Jeff.", words=words2)
    transcript = Transcript(
        id="t-casa-001", audio_id="casa-test-001",
        text="What is your name? My name is Jeff.",
        language=LanguageCode.ENGLISH, duration=3.0,
        segments=[seg1, seg2], words=words1 + words2,
    )
    svc.transcribe_audio.return_value = transcript
    return svc


def _make_speaker_embedding_with_centroids():
    """Mock that includes the V3.2 CASA data in diagnostics."""
    DIM = 192
    svc = MagicMock()

    c1 = np.zeros(DIM, dtype=np.float32); c1[10] = 1.0
    c2 = np.zeros(DIM, dtype=np.float32); c2[90] = 1.0

    svc.embed_segments.return_value = [
        c1.copy() + 0.01 * np.random.randn(DIM).astype(np.float32),
        c2.copy() + 0.01 * np.random.randn(DIM).astype(np.float32),
    ]
    svc.cluster_segments.return_value = ["Speaker 1", "Speaker 2"]
    svc.diarize_audio.return_value = (
        [
            {
                "start_sec": 0.0, "end_sec": 1.5, "duration_sec": 1.5,
                "text": "What is your name?",
                "words": [{"word": "What", "start_time": 0.1, "end_time": 0.4, "confidence": 0.95}],
                "speaker_label": "Speaker 1",
            },
            {
                "start_sec": 1.8, "end_sec": 3.0, "duration_sec": 1.2,
                "text": "My name is Jeff.",
                "words": [{"word": "My", "start_time": 1.9, "end_time": 2.1, "confidence": 0.92}],
                "speaker_label": "Speaker 2",
            },
        ],
        {
            "num_windows": 2, "num_embeddings": 2, "embedding_dim": DIM,
            "estimated_speakers": 2, "distinct_speakers": 2,
            "cluster_sizes": {"Speaker 1": 1, "Speaker 2": 1},
            "mean_cosine_sim": 0.10,
            # V3.2 data
            "phrase_embeddings": [
                c1 + 0.01 * np.random.randn(DIM).astype(np.float32),
                c2 + 0.01 * np.random.randn(DIM).astype(np.float32),
            ],
            "speaker_centroids": {"Speaker 1": c1, "Speaker 2": c2},
        },
    )
    return svc


def _make_acoustic():
    svc = MagicMock()
    feat = AcousticFeatures(f0_mean=220.0, rms_mean=0.05, mfcc_means=[0.1] * 13)
    svc.extract_batch.return_value = [feat, feat]
    return svc


def _make_repo():
    repo = MagicMock()
    repo.save_job.return_value = None
    repo.save_audio_asset.return_value = None
    repo.save_transcript.return_value = None
    repo.save_alignment_words.return_value = None
    repo.save_audio_segments.return_value = None
    return repo


def _make_worker(apply_casa: bool = True) -> AudioWorker:
    return AudioWorker(
        repository=_make_repo(),
        vad_service=_make_vad(),
        transcription_service=_make_transcription(),
        speaker_embedding_service=_make_speaker_embedding_with_centroids(),
        acoustic_service=_make_acoustic(),
        extract_acoustics=True,
        extract_embeddings=True,
        apply_casa=apply_casa,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCASAPipeline:

    def _run(self, apply_casa: bool, synthetic_wav, audio_asset):
        worker = _make_worker(apply_casa=apply_casa)
        worker.audio_service = MagicMock()
        worker.audio_service.normalize_to_wav.return_value = synthetic_wav
        worker.process_asset(audio_asset)
        segments = worker.repo.save_audio_segments.call_args[0][1]
        return segments

    def test_pipeline_completes_with_casa(self, audio_asset, synthetic_wav):
        worker = _make_worker(apply_casa=True)
        worker.audio_service = MagicMock()
        worker.audio_service.normalize_to_wav.return_value = synthetic_wav
        job = worker.process_asset(audio_asset)
        assert job.status == JobStatus.COMPLETED

    def test_casa_segments_have_confidence(self, audio_asset, synthetic_wav):
        segments = self._run(True, synthetic_wav, audio_asset)
        for seg in segments:
            assert seg.speaker_confidence is not None, (
                f"segment {seg.sequence_order} missing speaker_confidence"
            )
            assert 0.0 <= seg.speaker_confidence <= 1.0

    def test_casa_segments_have_decision(self, audio_asset, synthetic_wav):
        segments = self._run(True, synthetic_wav, audio_asset)
        for seg in segments:
            assert seg.attribution_decision is not None
            assert seg.attribution_decision in {"CONFIRM", "CORRECT", "UNCERTAIN"}

    def test_casa_segments_have_evidence(self, audio_asset, synthetic_wav):
        segments = self._run(True, synthetic_wav, audio_asset)
        for seg in segments:
            assert seg.attribution_evidence is not None
            assert isinstance(seg.attribution_evidence, list)
            assert len(seg.attribution_evidence) > 0

    def test_casa_early_phrases_marked_provisional(self, audio_asset, synthetic_wav):
        """Phrases starting before early_dialogue_window_sec are provisional=True."""
        segments = self._run(True, synthetic_wav, audio_asset)
        # All segments in our 3-second test are within the default 5-second window
        for seg in segments:
            assert seg.provisional is True

    def test_casa_disabled_confidence_is_none(self, audio_asset, synthetic_wav):
        """When CASA is off, all CASA fields should remain None."""
        segments = self._run(False, synthetic_wav, audio_asset)
        for seg in segments:
            assert seg.speaker_confidence is None, (
                f"CASA disabled but segment {seg.sequence_order} has speaker_confidence={seg.speaker_confidence}"
            )
            assert seg.attribution_decision is None
            assert seg.attribution_evidence is None

    def test_v31_fields_unaffected_when_casa_enabled(self, audio_asset, synthetic_wav):
        """V3.1 fields must be present regardless of CASA state."""
        segments = self._run(True, synthetic_wav, audio_asset)
        for seg in segments:
            assert seg.speaker_label is not None
            assert seg.speaker_embedding is not None
            assert len(seg.speaker_embedding) == 192
            assert seg.acoustic_features is not None

    def test_v31_fields_unaffected_when_casa_disabled(self, audio_asset, synthetic_wav):
        """V3.1 fields are unaffected when CASA is disabled."""
        segments = self._run(False, synthetic_wav, audio_asset)
        for seg in segments:
            assert seg.speaker_label is not None
            assert seg.speaker_embedding is not None
            assert seg.acoustic_features is not None

    def test_casa_timing_recorded(self, audio_asset, synthetic_wav):
        """When CASA is enabled, timings must include casa_sec."""
        worker = _make_worker(apply_casa=True)
        worker.audio_service = MagicMock()
        worker.audio_service.normalize_to_wav.return_value = synthetic_wav
        job = worker.process_asset(audio_asset)
        assert "casa_sec" in job.timings

    def test_no_casa_timing_when_disabled(self, audio_asset, synthetic_wav):
        """When CASA is disabled, casa_sec should NOT appear in timings."""
        worker = _make_worker(apply_casa=False)
        worker.audio_service = MagicMock()
        worker.audio_service.normalize_to_wav.return_value = synthetic_wav
        job = worker.process_asset(audio_asset)
        assert "casa_sec" not in job.timings

    def test_speaker_labels_are_grounded(self, audio_asset, synthetic_wav):
        """CASA must never introduce labels other than Speaker N."""
        import re
        segments = self._run(True, synthetic_wav, audio_asset)
        for seg in segments:
            if seg.speaker_label:
                assert re.match(r"^Speaker \d+$", seg.speaker_label), (
                    f"Non-grounded speaker label: {seg.speaker_label}"
                )
