"""
Unit test for the ExportService.
"""

from schemas.analysis import (
    AcousticFeatureSet,
    AnalysisMetadata,
    AnalysisResult,
    AudioInfo,
    AudioQuality,
    ConversationAnalytics,
    DiarizationResult,
    DiarizedSegment,
    ProcessingInfo,
    SpeakerProfile,
    SpeakerStatistics,
    TranscriptionResult,
    VADResult,
)
from services.export_service import ExportService


def _get_mock_result() -> AnalysisResult:
    return AnalysisResult(
        metadata=AnalysisMetadata(job_id="test_exp_job", audio_id="test_exp_aud"),
        audio=AudioInfo(filename="podcast.wav", format="wav", duration_sec=12.0),
        audio_quality=AudioQuality(rms_energy=0.05, dynamic_range_db=32.0),
        vad=VADResult(speech_duration_sec=10.0, silence_duration_sec=2.0, speech_ratio=0.83, total_segments=2),
        transcription=TranscriptionResult(full_text="Hello and welcome. Thank you."),
        diarization=DiarizationResult(
            num_speakers=2,
            segments=[
                DiarizedSegment(
                    id="s1", sequence_order=0, start_sec=0.0, end_sec=5.0, duration_sec=5.0,
                    text="Hello and welcome.", speaker_label="Speaker 1",
                    acoustic_features=AcousticFeatureSet(f0_mean=130.0, mfcc_means=[1.0]*13),
                ),
                DiarizedSegment(
                    id="s2", sequence_order=1, start_sec=6.0, end_sec=11.0, duration_sec=5.0,
                    text="Thank you.", speaker_label="Speaker 2",
                    acoustic_features=AcousticFeatureSet(f0_mean=220.0, mfcc_means=[2.0]*13),
                ),
            ],
        ),
        speakers=[
            SpeakerProfile(
                speaker_id="spk1", speaker_label="Speaker 1",
                statistics=SpeakerStatistics(total_speaking_sec=5.0, speaking_percentage=50.0, num_turns=1, avg_turn_sec=5.0),
            ),
            SpeakerProfile(
                speaker_id="spk2", speaker_label="Speaker 2",
                statistics=SpeakerStatistics(total_speaking_sec=5.0, speaking_percentage=50.0, num_turns=1, avg_turn_sec=5.0),
            ),
        ],
        conversation=ConversationAnalytics(total_duration_sec=12.0, num_turns=2, dominant_speaker="Speaker 1"),
        processing=ProcessingInfo(total_duration_sec=3.0, audio_duration_sec=12.0, realtime_factor=0.25),
    )


def test_export_srt():
    service = ExportService()
    res = _get_mock_result()
    srt = service.to_srt(res)
    assert "00:00:00,000 --> 00:00:05,000" in srt
    assert "[Speaker 1] Hello and welcome." in srt
    assert "00:00:06,000 --> 00:00:11,000" in srt
    assert "[Speaker 2] Thank you." in srt


def test_export_vtt():
    service = ExportService()
    res = _get_mock_result()
    vtt = service.to_vtt(res)
    assert "WEBVTT" in vtt
    assert "00:00:00.000 --> 00:00:05.000" in vtt
    assert "<v Speaker 1>Hello and welcome." in vtt


def test_export_csv_and_feature_matrix():
    service = ExportService()
    res = _get_mock_result()
    csv_out = service.to_csv_segments(res)
    assert "Speaker 1" in csv_out
    assert "Hello and welcome." in csv_out

    feat_out = service.to_feature_matrix_csv(res)
    assert "mfcc_1" in feat_out
    assert "mfcc_13" in feat_out
    assert "130.0" in feat_out


def test_export_markdown_report():
    service = ExportService()
    res = _get_mock_result()
    md = service.to_speaker_report_markdown(res)
    assert "# Audio Intelligence & Speaker Analytics Report" in md
    assert "Speakers Detected:" in md
    assert "Speaker 1" in md
    assert "Speaker 2" in md
