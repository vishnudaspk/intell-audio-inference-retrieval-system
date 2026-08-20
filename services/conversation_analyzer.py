"""
Conversation Analyzer Service.
Extracts turns, turn-taking transitions, silence gaps, short responses, dominant speaker, and balance.
"""

from typing import Dict, List
import numpy as np

from schemas.analysis import (
    ConversationAnalytics,
    ConversationTurn,
    DiarizedSegment,
    ShortResponse,
    SilenceGap,
    SpeakerTransition,
)


class ConversationAnalyzer:
    """Analyzes dialogue turn-taking patterns and structural dynamics of conversation."""

    def analyze(
        self,
        segments: List[DiarizedSegment],
        total_audio_duration_sec: float,
        silence_gap_threshold_sec: float = 0.5,
    ) -> ConversationAnalytics:
        """
        Produce a full ConversationAnalytics object from ordered diarized segments.
        """
        if not segments:
            return ConversationAnalytics(total_duration_sec=total_audio_duration_sec)

        turns: List[ConversationTurn] = []
        transitions: List[SpeakerTransition] = []
        short_responses: List[ShortResponse] = []
        silence_gaps: List[SilenceGap] = []

        # 1. Turn building & short responses
        for t_idx, seg in enumerate(segments):
            words = seg.text.strip().split()
            word_count = len(words)
            # Short response: brief acknowledgments/affirmations (<= 2 words, duration <= 2.0s)
            is_short = (0 < word_count <= 2 and seg.duration_sec <= 2.0)

            turns.append(
                ConversationTurn(
                    turn_index=t_idx,
                    speaker_label=seg.speaker_label,
                    start_sec=seg.start_sec,
                    end_sec=seg.end_sec,
                    duration_sec=seg.duration_sec,
                    text=seg.text,
                    word_count=word_count,
                    is_short_response=is_short,
                )
            )

            if is_short:
                short_responses.append(
                    ShortResponse(
                        start_sec=seg.start_sec,
                        end_sec=seg.end_sec,
                        text=seg.text,
                        speaker_label=seg.speaker_label,
                    )
                )

        # 2. Speaker transitions & Silence gaps
        for i in range(len(segments) - 1):
            cur = segments[i]
            nxt = segments[i + 1]
            gap = round(nxt.start_sec - cur.end_sec, 3)

            # Silence gap check
            if gap >= silence_gap_threshold_sec:
                silence_gaps.append(
                    SilenceGap(
                        start_sec=cur.end_sec,
                        end_sec=nxt.start_sec,
                        duration_sec=gap,
                    )
                )

            # Speaker transition
            if cur.speaker_label != nxt.speaker_label:
                transitions.append(
                    SpeakerTransition(
                        from_speaker=cur.speaker_label,
                        to_speaker=nxt.speaker_label,
                        gap_sec=gap,
                        at_sec=cur.end_sec,
                    )
                )

        # 3. Dominant speaker & balance
        speaker_durations: Dict[str, float] = {}
        for s in segments:
            speaker_durations[s.speaker_label] = speaker_durations.get(s.speaker_label, 0.0) + s.duration_sec

        dominant_spk = max(speaker_durations.items(), key=lambda x: x[1])[0] if speaker_durations else "Speaker 1"
        balance = {
            spk: round(dur / max(1e-5, total_audio_duration_sec) * 100, 1)
            for spk, dur in speaker_durations.items()
        }

        return ConversationAnalytics(
            total_duration_sec=round(total_audio_duration_sec, 2),
            num_turns=len(turns),
            num_speakers=len(speaker_durations),
            turns=turns,
            transitions=transitions,
            dominant_speaker=dominant_spk,
            conversation_balance=balance,
            short_responses=short_responses,
            silence_gaps=silence_gaps,
        )
