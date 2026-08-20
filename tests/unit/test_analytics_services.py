"""
Unit tests for AudioQualityService, SpeakerAnalyticsService, ConversationAnalyzer, and HMMSmoother.
"""

from pathlib import Path
import numpy as np
import pytest

from schemas.analysis import AcousticFeatureSet, DiarizedSegment
from services.audio_quality_service import AudioQualityService
from services.conversation_analyzer import ConversationAnalyzer
from services.hmm_smoother import HMMSmoother
from services.speaker_analytics_service import SpeakerAnalyticsService


def test_audio_quality_service_synthetic_data(tmp_path: Path):
    import soundfile as sf

    sr = 16000
    # Create 1s synthetic sine wave (clean, no clipping)
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav_file = tmp_path / "clean_sine.wav"
    sf.write(str(wav_file), audio, sr)

    service = AudioQualityService()
    quality = service.analyze(wav_file)

    assert not quality.clipping_detected
    assert quality.rms_energy > 0.0
    assert quality.dynamic_range_db >= 0.0


def test_speaker_analytics_service():
    service = SpeakerAnalyticsService()
    segments = [
        DiarizedSegment(
            id="1", sequence_order=0, start_sec=0.0, end_sec=5.0, duration_sec=5.0,
            text="Hello welcome to the meeting.", speaker_label="Speaker 1",
            acoustic_features=AcousticFeatureSet(f0_mean=120.0, rms_mean=0.05),
        ),
        DiarizedSegment(
            id="2", sequence_order=1, start_sec=5.5, end_sec=9.5, duration_sec=4.0,
            text="Thank you glad to be here.", speaker_label="Speaker 2",
            acoustic_features=AcousticFeatureSet(f0_mean=210.0, rms_mean=0.04),
        ),
        DiarizedSegment(
            id="3", sequence_order=2, start_sec=10.0, end_sec=15.0, duration_sec=5.0,
            text="Let us discuss the roadmap.", speaker_label="Speaker 1",
            acoustic_features=AcousticFeatureSet(f0_mean=125.0, rms_mean=0.06),
        ),
    ]

    profiles = service.compute_profiles(segments, total_audio_duration_sec=15.0)
    assert len(profiles) == 2

    spk1 = next(p for p in profiles if p.speaker_label == "Speaker 1")
    assert spk1.statistics.num_turns == 2
    assert spk1.statistics.total_speaking_sec == 10.0
    assert spk1.statistics.longest_turn_sec == 5.0
    assert spk1.features.f0_mean == 122.5


def test_conversation_analyzer():
    analyzer = ConversationAnalyzer()
    segments = [
        DiarizedSegment(id="1", sequence_order=0, start_sec=0.0, end_sec=4.0, duration_sec=4.0, text="First topic is ready.", speaker_label="Speaker 1"),
        DiarizedSegment(id="2", sequence_order=1, start_sec=4.2, end_sec=5.0, duration_sec=0.8, text="Yes.", speaker_label="Speaker 2"),
        DiarizedSegment(id="3", sequence_order=2, start_sec=6.0, end_sec=10.0, duration_sec=4.0, text="Let us review results.", speaker_label="Speaker 1"),
    ]

    conv = analyzer.analyze(segments, total_audio_duration_sec=10.0, silence_gap_threshold_sec=0.5)

    assert conv.num_turns == 3
    assert conv.num_speakers == 2
    assert conv.dominant_speaker == "Speaker 1"
    assert len(conv.short_responses) == 1
    assert conv.short_responses[0].text == "Yes."
    assert len(conv.silence_gaps) == 1
    assert conv.silence_gaps[0].duration_sec == 1.0


def test_hmm_smoother_jitter_reduction():
    smoother = HMMSmoother()
    # Sequence with isolated 1-frame jitter: A -> A -> B -> A -> A
    raw_labels = ["Speaker 1", "Speaker 1", "Speaker 2", "Speaker 1", "Speaker 1"]
    res = smoother.smooth_sequence(raw_labels)

    assert res is not None
    assert res.num_states == 2
    assert len(res.transition_matrix) == 2
    # The middle jitter should be smoothed back to Speaker 1
    assert res.speaker_sequence[2] == "Speaker 1"
