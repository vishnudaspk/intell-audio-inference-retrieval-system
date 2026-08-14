"""
Speaker Assignment service matching transcript chunks to speaker-turn segments.
"""

from typing import List

from schemas.models import SpeakerSegment, TranscriptChunk
from utils.logger import logger


class SpeakerAssignmentService:
    """
    Assigns speaker segments to transcript chunks based on temporal overlap.
    CRITICAL: Never fabricates identities. If confidence is 0.0 or no segment overlaps,
    speaker_label is strictly 'Unknown Speaker' and speaker_id is None.
    """

    def assign(
        self,
        chunks: List[TranscriptChunk],
        segments: List[SpeakerSegment],
        confidence_threshold: float = 0.7,
    ) -> List[TranscriptChunk]:
        """
        Assign speaker metadata to each chunk based on max temporal overlap with SpeakerSegments.
        """
        if not chunks:
            return []

        if not segments:
            # No speaker segments available
            for chunk in chunks:
                chunk.speaker_id = None
                chunk.speaker_label = "Unknown Speaker"
                chunk.speaker_confidence = 0.0
            return chunks

        for chunk in chunks:
            best_segment = None
            max_overlap = 0.0

            for seg in segments:
                # Compute overlap interval between chunk and segment
                overlap_start = max(chunk.start_time, seg.start_time)
                overlap_end = min(chunk.end_time, seg.end_time)
                overlap_duration = max(0.0, overlap_end - overlap_start)

                if overlap_duration > max_overlap:
                    max_overlap = overlap_duration
                    best_segment = seg

            if best_segment and best_segment.confidence >= confidence_threshold and best_segment.speaker_id:
                chunk.speaker_id = best_segment.speaker_id
                chunk.speaker_label = best_segment.speaker_label
                chunk.speaker_confidence = best_segment.confidence
            else:
                # Heuristic or low-confidence turn
                chunk.speaker_id = None
                chunk.speaker_label = "Unknown Speaker"
                chunk.speaker_confidence = best_segment.confidence if best_segment else 0.0

            # Store in chunk metadata for backward compatibility
            chunk.metadata["speaker_id"] = chunk.speaker_id
            chunk.metadata["speaker_label"] = chunk.speaker_label
            chunk.metadata["speaker_confidence"] = chunk.speaker_confidence

        logger.info(f"Assigned speaker metadata across {len(chunks)} chunks.")
        return chunks
