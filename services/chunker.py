"""
Temporal transcript chunking engine preserving exact word alignment boundaries and timestamps.
"""

from typing import List, Optional

from config.settings import settings
from schemas.models import Transcript, TranscriptChunk, TranscriptWord
from utils.logger import logger


class TranscriptChunker:
    """
    Splits aligned transcript words into timestamp-preserving temporal chunks.
    Uses deterministic chunk IDs and preserves start/end times from original word alignments.
    """

    def __init__(
        self,
        chunk_size_words: Optional[int] = None,
        overlap_words: Optional[int] = None,
    ):
        self.chunk_size_words = chunk_size_words or settings.CHUNK_SIZE_WORDS
        self.overlap_words = overlap_words or settings.CHUNK_OVERLAP_WORDS

    def chunk_transcript(
        self,
        transcript: Transcript,
        chunk_size_words: Optional[int] = None,
        overlap_words: Optional[int] = None,
    ) -> List[TranscriptChunk]:
        """
        Produce temporal transcript chunks from a Transcript object and its aligned words.
        """
        size = chunk_size_words or self.chunk_size_words
        overlap = overlap_words or self.overlap_words

        if size <= 0:
            size = 60
        if overlap < 0 or overlap >= size:
            overlap = min(10, max(0, size // 5))

        words = [w for w in transcript.words if w.word and w.word.strip()]

        if not words:
            # Fallback if no word alignment present (e.g. raw text fallback)
            if not transcript.text or not transcript.text.strip():
                return []
            
            # Synthesize single chunk from text
            chunk_id = f"{transcript.audio_id}_chk_0000"
            return [
                TranscriptChunk(
                    chunk_id=chunk_id,
                    audio_id=transcript.audio_id,
                    transcript_id=transcript.id,
                    sequence_order=0,
                    text=transcript.text.strip(),
                    start_time=0.0,
                    end_time=transcript.duration if transcript.duration > 0 else 0.0,
                    words=[],
                    language=transcript.language.value if hasattr(transcript.language, "value") else str(transcript.language),
                    metadata={"fallback": True},
                )
            ]

        step = size - overlap
        if step <= 0:
            step = size

        chunks: List[TranscriptChunk] = []
        seq_order = 0
        total_words = len(words)
        idx = 0

        while idx < total_words:
            chunk_words = words[idx : idx + size]
            if not chunk_words:
                break

            # Find start_time from first word with valid start
            start_t = 0.0
            for w in chunk_words:
                if w.start is not None:
                    start_t = w.start
                    break

            # Find end_time from last word with valid end or start
            end_t = start_t
            for w in reversed(chunk_words):
                if w.end is not None:
                    end_t = w.end
                    break
                elif w.start is not None:
                    end_t = w.start
                    break

            chunk_text = " ".join([w.word.strip() for w in chunk_words])
            chunk_id = f"{transcript.audio_id}_chk_{seq_order:04d}"

            lang_val = transcript.language.value if hasattr(transcript.language, "value") else str(transcript.language)

            chunks.append(
                TranscriptChunk(
                    chunk_id=chunk_id,
                    audio_id=transcript.audio_id,
                    transcript_id=transcript.id,
                    sequence_order=seq_order,
                    text=chunk_text,
                    start_time=start_t,
                    end_time=end_t,
                    words=chunk_words,
                    language=lang_val,
                    metadata={
                        "word_count": len(chunk_words),
                        "start_word_index": idx,
                        "end_word_index": idx + len(chunk_words) - 1,
                    },
                )
            )

            seq_order += 1
            idx += step

        logger.info(f"Chunked transcript for audio {transcript.audio_id} into {len(chunks)} temporal chunks.")
        return chunks
