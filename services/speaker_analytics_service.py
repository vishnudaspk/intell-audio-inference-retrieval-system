"""
Speaker Analytics Service.
Computes per-speaker turn counts, durations, pause times, response latencies, and speaking rates.
"""

from typing import Dict, List, Optional
import numpy as np

from schemas.analysis import AcousticFeatureSet, DiarizedSegment, SpeakerProfile, SpeakerStatistics


class SpeakerAnalyticsService:
    """Computes high-level aggregated speaker metrics across all diarized dialogue segments."""

    DEFAULT_PALETTE = [
        "#4A90E2", "#50E3C2", "#F5A623", "#E35050", "#BD10E0",
        "#7ED321", "#9013FE", "#417505", "#B8E986", "#4A4A4A"
    ]

    def compute_profiles(
        self,
        segments: List[DiarizedSegment],
        total_audio_duration_sec: float,
    ) -> List[SpeakerProfile]:
        """
        Build a list of SpeakerProfile objects from diarized segments.
        """
        if not segments:
            return []

        # Find distinct speakers preserving chronological order
        unique_speakers: List[str] = []
        for s in segments:
            if s.speaker_label not in unique_speakers:
                unique_speakers.append(s.speaker_label)

        profiles: List[SpeakerProfile] = []

        for spk_idx, spk in enumerate(unique_speakers):
            spk_segs = [s for s in segments if s.speaker_label == spk]
            spk_dur = sum(s.duration_sec for s in spk_segs)
            mean_conf = float(np.mean([s.confidence for s in spk_segs])) if spk_segs else 1.0

            # Turn metrics
            num_turns = len(spk_segs)
            durations = [s.duration_sec for s in spk_segs]
            avg_turn = float(np.mean(durations)) if durations else 0.0
            longest = float(np.max(durations)) if durations else 0.0
            shortest = float(np.min(durations)) if durations else 0.0

            # Word count for speaking rate
            total_words = sum(len(s.text.split()) for s in spk_segs if s.text)
            wps = round(total_words / max(1e-5, spk_dur), 2) if spk_dur > 0 else 0.0

            # Pauses between own turns
            pauses: List[float] = []
            for i in range(len(spk_segs) - 1):
                gap = spk_segs[i + 1].start_sec - spk_segs[i].end_sec
                if gap > 0:
                    pauses.append(gap)
            avg_pause = round(float(np.mean(pauses)), 2) if pauses else None

            # Average response latency: time elapsed after other speaker finished before this speaker speaks
            latencies: List[float] = []
            for i, seg in enumerate(segments):
                if seg.speaker_label == spk and i > 0:
                    prev = segments[i - 1]
                    if prev.speaker_label != spk:
                        latency = seg.start_sec - prev.end_sec
                        if latency >= 0:
                            latencies.append(latency)
            avg_latency = round(float(np.mean(latencies)), 2) if latencies else None

            # Aggregate segment acoustic features (mean across speaker's segments)
            f0_means = [s.acoustic_features.f0_mean for s in spk_segs if s.acoustic_features and s.acoustic_features.f0_mean is not None]
            rms_means = [s.acoustic_features.rms_mean for s in spk_segs if s.acoustic_features and s.acoustic_features.rms_mean is not None]

            features = AcousticFeatureSet(
                f0_mean=round(float(np.mean(f0_means)), 2) if f0_means else None,
                rms_mean=round(float(np.mean(rms_means)), 4) if rms_means else None,
            )

            stats = SpeakerStatistics(
                total_speaking_sec=round(spk_dur, 2),
                speaking_percentage=round(spk_dur / max(1e-5, total_audio_duration_sec) * 100, 1),
                num_turns=num_turns,
                avg_turn_sec=round(avg_turn, 2),
                longest_turn_sec=round(longest, 2),
                shortest_turn_sec=round(shortest, 2),
                avg_pause_sec=avg_pause,
                response_latency_sec=avg_latency,
                speaking_rate_wps=wps,
            )

            profiles.append(
                SpeakerProfile(
                    speaker_id=f"speaker_{spk_idx + 1}",
                    speaker_label=spk,
                    color=self.DEFAULT_PALETTE[spk_idx % len(self.DEFAULT_PALETTE)],
                    statistics=stats,
                    features=features,
                    confidence=round(mean_conf, 3),
                    segment_count=num_turns,
                )
            )

        return profiles
