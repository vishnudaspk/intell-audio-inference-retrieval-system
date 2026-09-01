"""
Export Service for formatting AnalysisResult into CSV, SRT, VTT, and Markdown reports.
"""

import csv
import io
from typing import Dict, List
from schemas.analysis import AnalysisResult


class ExportService:
    """Generates structured exports directly from an AnalysisResult."""

    @staticmethod
    def _format_timestamp_srt(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_timestamp_vtt(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def to_srt(self, result: AnalysisResult) -> str:
        """Export diarized speech segments to SubRip (.srt) format with speaker labels."""
        lines: List[str] = []
        segments = result.diarization.segments or []
        for idx, seg in enumerate(segments, start=1):
            st = self._format_timestamp_srt(seg.start_sec)
            et = self._format_timestamp_srt(seg.end_sec)
            text = seg.text or ""
            spk = seg.speaker_label
            lines.append(f"{idx}\n{st} --> {et}\n[{spk}] {text}\n")
        return "\n".join(lines)

    def to_vtt(self, result: AnalysisResult) -> str:
        """Export diarized speech segments to WebVTT (.vtt) format."""
        lines: List[str] = ["WEBVTT", ""]
        segments = result.diarization.segments or []
        for idx, seg in enumerate(segments, start=1):
            st = self._format_timestamp_vtt(seg.start_sec)
            et = self._format_timestamp_vtt(seg.end_sec)
            text = seg.text or ""
            spk = seg.speaker_label
            lines.append(f"{idx}\n{st} --> {et}\n<v {spk}>{text}\n")
        return "\n".join(lines)

    def to_csv_segments(self, result: AnalysisResult) -> str:
        """Export segments with speaker attribution and timestamps to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["index", "speaker", "start_sec", "end_sec", "duration_sec", "confidence", "text"])
        for seg in (result.diarization.segments or []):
            writer.writerow([
                seg.sequence_order,
                seg.speaker_label,
                seg.start_sec,
                seg.end_sec,
                seg.duration_sec,
                seg.confidence,
                seg.text,
            ])
        return output.getvalue()

    def to_feature_matrix_csv(self, result: AnalysisResult) -> str:
        """Export acoustic features per segment to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        header = ["index", "speaker", "start_sec", "end_sec", "f0_mean", "f0_std", "rms_mean", "spectral_centroid", "zcr"]
        # Add MFCC column headers
        header.extend([f"mfcc_{i+1}" for i in range(13)])
        writer.writerow(header)

        for seg in (result.diarization.segments or []):
            feat = seg.acoustic_features
            row = [
                seg.sequence_order,
                seg.speaker_label,
                seg.start_sec,
                seg.end_sec,
                feat.f0_mean if feat else None,
                feat.f0_std if feat else None,
                feat.rms_mean if feat else None,
                feat.spectral_centroid_mean if feat else None,
                feat.zcr_mean if feat else None,
            ]
            mfccs = (feat.mfcc_means if feat and feat.mfcc_means else [None] * 13)
            row.extend(mfccs[:13])
            writer.writerow(row)

        return output.getvalue()

    def to_speaker_report_markdown(self, result: AnalysisResult) -> str:
        """Export a comprehensive Markdown executive summary report."""
        lines = [
            f"# Audio Intelligence & Speaker Analytics Report",
            f"**File:** `{result.audio.filename}` | **Duration:** {result.audio.duration_sec}s",
            f"**Job ID:** `{result.metadata.job_id}` | **Generated:** {result.metadata.created_at}",
            "",
            "## 1. Executive Summary",
            f"- **Speakers Detected:** {len(result.speakers)}",
            f"- **Dominant Speaker:** {result.conversation.dominant_speaker}",
            f"- **Total Turns:** {result.conversation.num_turns}",
            f"- **Speech Ratio:** {result.vad.speech_ratio * 100:.1f}% ({result.vad.speech_duration_sec}s speech / {result.vad.silence_duration_sec}s silence)",
            "",
            "## 2. Speaker Breakdown",
            "| Speaker | Speaking Time (s) | Share (%) | Turns | Avg Turn (s) | F0 Mean (Hz) |",
            "|---|---|---|---|---|---|",
        ]

        for spk in result.speakers:
            lines.append(
                f"| {spk.speaker_label} | {spk.statistics.total_speaking_sec} | "
                f"{spk.statistics.speaking_percentage}% | {spk.statistics.num_turns} | "
                f"{spk.statistics.avg_turn_sec} | {spk.features.f0_mean or 'N/A'} |"
            )

        lines.extend([
            "",
            "## 3. Signal Quality",
            f"- **RMS Energy:** {result.audio_quality.rms_energy}",
            f"- **Dynamic Range:** {result.audio_quality.dynamic_range_db} dB",
            f"- **Clipping Detected:** {'Yes' if result.audio_quality.clipping_detected else 'No'}",
        ])

        return "\n".join(lines)
