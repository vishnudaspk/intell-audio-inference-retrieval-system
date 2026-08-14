"""
Chapter Generation service combining deterministic boundary detection with LLM title synthesis.
"""

import json
import re
from typing import Dict, List, Optional
import uuid

from config.settings import settings
from schemas.models import Chapter, TranscriptChunk
from services.llm_service import LLMProvider, LMStudioLLMProvider
from utils.logger import logger

SYSTEM_CHAPTER_PROMPT = """You are an audio chapter editor.
Given a list of audio sections with their timestamps and text summaries, generate a concise, descriptive title and 1-sentence summary for each chapter.

Return ONLY a valid JSON array of objects:
[
  {
    "chapter_index": 0,
    "title": "Concise Chapter Title",
    "summary": "Brief factual summary of what happens in this chapter.",
    "dominant_topic": "Topic Name"
  }
]
"""


class ChapterGenerator:
    """
    Detects chapter boundaries from deterministic signals (pauses, topic shifts, term discontinuity)
    and uses LLM to synthesize clean chapter titles and summaries.
    CRITICAL: Timestamps always originate from transcript chunks, never from the LLM.
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm_provider = llm_provider or LMStudioLLMProvider()
        self.enabled = settings.ENABLE_CHAPTER_GENERATION
        self.min_pause_sec = settings.CHAPTER_MIN_PAUSE_SEC
        self.discontinuity_thresh = settings.CHAPTER_DISCONTINUITY_THRESHOLD
        self.max_chapters = settings.CHAPTER_MAX_COUNT

    def generate_chapters(
        self,
        chunks: List[TranscriptChunk],
        audio_id: str,
    ) -> List[Chapter]:
        """
        Produce structured chapters from chunks and back-assign chapter_ids to chunks.
        """
        if not self.enabled or not chunks:
            return []

        # 1. Deterministic boundary detection
        boundary_indices = self._detect_boundaries(chunks)

        # 2. Group chunks into candidate chapter regions
        chapter_groups: List[List[TranscriptChunk]] = []
        start_idx = 0
        for b_idx in boundary_indices:
            if b_idx > start_idx:
                chapter_groups.append(chunks[start_idx:b_idx])
                start_idx = b_idx
        if start_idx < len(chunks):
            chapter_groups.append(chunks[start_idx:])

        if not chapter_groups:
            chapter_groups = [chunks]

        # 3. Build chapter objects with exact timestamps from chunk boundaries
        chapters: List[Chapter] = []
        for seq, group in enumerate(chapter_groups):
            start_t = group[0].start_time
            end_t = group[-1].end_time
            chunk_ids = [c.chunk_id for c in group]
            speaker_ids = list(set(c.speaker_id for c in group if c.speaker_id))

            # Default fallback title
            fallback_title = f"Section {seq + 1} ({self._format_time(start_t)})"

            chapter = Chapter(
                chapter_id=f"{audio_id}_ch_{seq:02d}_{uuid.uuid4().hex[:6]}",
                audio_id=audio_id,
                title=fallback_title,
                summary=None,
                start_time=start_t,
                end_time=end_t,
                dominant_topic=group[0].topic if group[0].topic else None,
                sequence_order=seq,
                speaker_ids=speaker_ids,
                chunk_ids=chunk_ids,
                metadata={"chunk_count": len(group)},
            )
            chapters.append(chapter)

        # 4. Optional LLM chapter title and summary synthesis
        self._synthesize_chapter_titles(chapters, chapter_groups)

        # 5. Back-assign chapter_id to each chunk
        for ch in chapters:
            ch_id = ch.chapter_id
            for chunk in chunks:
                if chunk.chunk_id in ch.chunk_ids:
                    chunk.chapter_id = ch_id
                    chunk.metadata["chapter_id"] = ch_id
                    chunk.metadata["chapter_title"] = ch.title

        logger.info(f"Generated {len(chapters)} chapters for audio {audio_id}.")
        return chapters

    def _detect_boundaries(self, chunks: List[TranscriptChunk]) -> List[int]:
        """Detect boundary indices between chunks based on pauses, topics, and vocabulary shift."""
        boundaries: List[int] = []

        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            curr = chunks[i]

            # Signal 1: Significant pause / silence gap between chunks
            gap = curr.start_time - prev.end_time
            if gap >= self.min_pause_sec:
                boundaries.append(i)
                continue

            # Signal 2: Topic change (if content analysis was run)
            if prev.topic and curr.topic and prev.topic.lower() != curr.topic.lower():
                boundaries.append(i)
                continue

            # Signal 3: Lexical discontinuity (Jaccard distance of terms)
            prev_terms = set(re.sub(r"[^\w\s]", " ", prev.text.lower()).split())
            curr_terms = set(re.sub(r"[^\w\s]", " ", curr.text.lower()).split())
            if prev_terms and curr_terms:
                overlap = len(prev_terms & curr_terms) / len(prev_terms | curr_terms)
                if overlap < self.discontinuity_thresh and (curr.start_time - chunks[boundaries[-1] if boundaries else 0].start_time) > 30.0:
                    boundaries.append(i)

            if len(boundaries) >= self.max_chapters - 1:
                break

        return boundaries

    def _synthesize_chapter_titles(
        self,
        chapters: List[Chapter],
        chapter_groups: List[List[TranscriptChunk]],
    ) -> None:
        """Call LLM once to generate titles and summaries for all chapters."""
        if not self.llm_provider or not self.llm_provider.is_available():
            return

        try:
            prompt_items = []
            for idx, (ch, group) in enumerate(zip(chapters, chapter_groups)):
                # Join chunk summaries or first few sentences
                text_sample = " ".join(c.chunk_summary or c.text for c in group[:3])
                if len(text_sample) > 300:
                    text_sample = text_sample[:300] + "..."
                prompt_items.append(
                    f"Chapter {idx} ({self._format_time(ch.start_time)}–{self._format_time(ch.end_time)}): {text_sample}"
                )

            prompt = "Generate titles and summaries for the following sections:\n\n" + "\n\n".join(prompt_items)

            raw = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=SYSTEM_CHAPTER_PROMPT,
                temperature=0.2,
                json_mode=True,
            )

            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()

            parsed = json.loads(cleaned)
            if not isinstance(parsed, list):
                parsed = parsed.get("chapters", [parsed]) if isinstance(parsed, dict) else []

            for item in parsed:
                if isinstance(item, dict):
                    idx = item.get("chapter_index")
                    if idx is not None and 0 <= idx < len(chapters):
                        if item.get("title"):
                            chapters[idx].title = str(item["title"]).strip()
                        if item.get("summary"):
                            chapters[idx].summary = str(item["summary"]).strip()
                        if item.get("dominant_topic"):
                            chapters[idx].dominant_topic = str(item["dominant_topic"]).strip()

        except Exception as exc:
            logger.debug(f"Chapter title synthesis fallback: {exc}")

    def _format_time(self, seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
