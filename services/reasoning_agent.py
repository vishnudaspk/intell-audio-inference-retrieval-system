"""
Grounded Reasoning Agent and Application-side Timestamp Citation Resolver.
Synthesizes natural language answers from pre-retrieved evidence while enforcing strict grounding constraints.
"""

import json
import time
from typing import Dict, List, Optional

from config.settings import settings
from database.base import BaseRepository
from schemas.models import Citation, QueryIntent, RAGResponse, RetrievalResult, StructuredRAGOutput, TranscriptChunk
from services.llm_service import LLMProvider
from services.temporal_span_resolver import TemporalSpanResolver
from utils.exceptions import IntellAudioError
from utils.logger import logger


SYSTEM_GROUNDING_PROMPT = """You are a factual, audio-grounded retrieval reasoning assistant.
Your job is to answer the user's question using ONLY the provided transcript evidence chunks.

STRICT RULES:
1. Base your answer strictly on the provided evidence chunks.
2. Do NOT invent or assume any facts, events, speakers, or details outside the evidence.
3. Do NOT invent timestamps or write raw timestamp numbers in your prose answer.
4. Instead, you MUST reference the specific chunk IDs (e.g., "chunk_id") that support your statements in the "evidence_ids" JSON field.
5. If the provided evidence is insufficient to answer the question, set "answer" to "I couldn't find enough evidence in the audio to answer that reliably." and "grounded" to false.
6. Return your response in valid JSON format matching the schema below.

JSON SCHEMA:
{
  "answer": "Concise natural language answer based exclusively on evidence...",
  "evidence_ids": ["audio_id_chk_0000", "audio_id_chk_0001"],
  "grounded": true
}
"""


class ReasoningAgent:
    """
    Reasoning Agent responsible for grounded RAG synthesis.
    Communicates with LLMProvider and resolves citations on the application side.
    """

    def __init__(self, llm_provider: LLMProvider, repository: Optional[BaseRepository] = None):
        self.llm_provider = llm_provider
        self.repository = repository

    def answer_question(
        self,
        query: str,
        retrieved_chunks: List[RetrievalResult],
        audio_id: Optional[str] = None,
        query_intent: Optional[QueryIntent] = None,  # Phase 7B
    ) -> RAGResponse:
        """
        Synthesize grounded natural language answer and resolve citations.
        Phase 7B: accepts optional query_intent to enrich response with temporal spans,
        speaker/chapter context, and intent metadata.
        """
        start_time_perf = time.perf_counter()

        # Check evidence availability and relevance thresholds
        if not retrieved_chunks:
            return self._build_abstention_response(
                query=query,
                reason="No evidence retrieved.",
                processing_time=time.perf_counter() - start_time_perf,
            )

        top_score = max((c.score for c in retrieved_chunks), default=0.0)
        if settings.RAG_REQUIRE_EVIDENCE and top_score < settings.RAG_MIN_RELEVANCE_SCORE:
            logger.info(f"Top retrieval score ({top_score:.3f}) below threshold ({settings.RAG_MIN_RELEVANCE_SCORE}). Abstaining.")
            return self._build_abstention_response(
                query=query,
                retrieved_chunks=retrieved_chunks,
                reason=f"Top evidence relevance ({top_score:.2f}) below threshold.",
                processing_time=time.perf_counter() - start_time_perf,
            )

        # Build prompt with evidence chunks
        evidence_prompt_parts = [f"USER QUESTION: {query}\n\nEVIDENCE CHUNKS:\n"]
        chunk_map: Dict[str, TranscriptChunk] = {}

        for idx, res in enumerate(retrieved_chunks, start=1):
            chunk = res.chunk
            chunk_map[chunk.chunk_id] = chunk

            text_to_present = chunk.metadata.get("expanded_context", chunk.text)
            evidence_prompt_parts.append(
                f"--- EVIDENCE CHUNK #{idx} ---\n"
                f"[CHUNK_ID={chunk.chunk_id}]\n"
                f"[AUDIO_ID={chunk.audio_id}]\n"
                f"[TIME_START={chunk.start_time:.2f}s]\n"
                f"[TIME_END={chunk.end_time:.2f}s]\n"
                f"TEXT: {text_to_present}\n"
            )

        evidence_prompt = "\n".join(evidence_prompt_parts)

        # Execute LLM generation with JSON parsing and retry logic
        structured_output = self._generate_structured_answer(evidence_prompt)

        # Application-Side Citation Resolution
        citations: List[Citation] = []
        valid_evidence_count = 0

        for e_id in structured_output.evidence_ids:
            chunk = chunk_map.get(e_id)
            if not chunk and self.repository:
                # Attempt lookup from repository if not in top candidate map
                chunk = self._lookup_chunk_from_repo(e_id)

            if chunk:
                # Resolve speaker label and chapter title for citation if available
                cit_speaker = chunk.speaker_label if chunk.speaker_label and chunk.speaker_label != "Unknown Speaker" else None
                cit_chapter = None
                if chunk.chapter_id and self.repository:
                    try:
                        chapters = self.repository.get_chapters(chunk.audio_id)
                        matching = next((c for c in chapters if c.chapter_id == chunk.chapter_id), None)
                        if matching:
                            cit_chapter = matching.title
                    except Exception:
                        pass

                citations.append(
                    Citation(
                        audio_id=chunk.audio_id,
                        chunk_id=chunk.chunk_id,
                        start_time=chunk.start_time,
                        end_time=chunk.end_time,
                        text=chunk.text,
                        speaker_label=cit_speaker,
                        chapter_title=cit_chapter,
                    )
                )
                valid_evidence_count += 1
            else:
                logger.warning(f"LLM cited invalid/unknown chunk ID '{e_id}'. Citation discarded.")

        # Determine confidence score
        confidence = 0.0
        if structured_output.grounded and citations:
            confidence = min(1.0, max(0.5, top_score * 0.7 + (valid_evidence_count / len(retrieved_chunks)) * 0.3))
        elif structured_output.grounded:
            confidence = 0.4

        processing_time = time.perf_counter() - start_time_perf

        raw_model = getattr(self.llm_provider, "model_name", "Qwen3-8B")
        model_name = str(raw_model) if isinstance(raw_model, str) else "Qwen3-8B"

        # Phase 7B: Resolve temporal spans and enrich response
        top_cited_chunk = chunk_map.get(citations[0].chunk_id) if citations else None
        span_resolver = TemporalSpanResolver()
        primary_timestamp = span_resolver.resolve_primary(citations, query_intent, top_chunk=top_cited_chunk)
        related_sections = span_resolver.resolve_related(citations[1:] if len(citations) > 1 else [])

        # Phase 7B: Extract speaker / chapter from top citation chunk
        speaker: Optional[str] = None
        chapter: Optional[str] = None
        if citations and top_cited_chunk:
            if top_cited_chunk.speaker_label and top_cited_chunk.speaker_label != "Unknown Speaker":
                speaker = top_cited_chunk.speaker_label
            if top_cited_chunk.chapter_id and self.repository:
                try:
                    chapters = self.repository.get_chapters(top_cited_chunk.audio_id)
                    matching = next((c for c in chapters if c.chapter_id == top_cited_chunk.chapter_id), None)
                    if matching:
                        chapter = matching.title
                except Exception:
                    pass

        # Build confidence reason string
        confidence_reason = ""
        if structured_output.grounded and citations:
            confidence_reason = f"{valid_evidence_count}/{len(retrieved_chunks)} chunks cited"
        elif not structured_output.grounded:
            confidence_reason = "Insufficient evidence"

        evidence_summary = (
            f"{len(citations)} citation(s) from {len(retrieved_chunks)} retrieved chunks"
            if citations else "No citations resolved"
        )

        return RAGResponse(
            answer=structured_output.answer,
            confidence=confidence,
            grounded=structured_output.grounded,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            query=query,
            processing_time=processing_time,
            model=model_name,
            retrieval_metadata={
                "top_score": top_score,
                "retrieved_count": len(retrieved_chunks),
                "cited_count": len(citations),
            },
            # Phase 7B enrichment
            primary_timestamp=primary_timestamp,
            related_sections=related_sections,
            speaker=speaker,
            chapter=chapter,
            intent=query_intent.intent if query_intent else None,
            abstained=not structured_output.grounded,
            confidence_reason=confidence_reason,
            evidence_summary=evidence_summary,
        )

    def _generate_structured_answer(self, prompt: str) -> StructuredRAGOutput:
        """Call LLM with json_mode and validate output structure; retry once if needed."""
        try:
            raw_response = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=SYSTEM_GROUNDING_PROMPT,
                temperature=0.1,
                json_mode=True,
            )
            return self._parse_json_response(raw_response)
        except Exception as exc:
            logger.warning(f"Initial LLM response failed JSON parsing ({exc}). Retrying with correction prompt.")
            try:
                correction_prompt = (
                    f"{prompt}\n\n"
                    "CRITICAL NOTICE: Your previous output was invalid JSON. "
                    "Return strictly valid JSON with keys 'answer', 'evidence_ids', and 'grounded'."
                )
                raw_response = self.llm_provider.generate(
                    prompt=correction_prompt,
                    system_prompt=SYSTEM_GROUNDING_PROMPT,
                    temperature=0.0,
                    json_mode=True,
                )
                return self._parse_json_response(raw_response)
            except Exception as retry_exc:
                logger.error(f"Correction retry failed: {retry_exc}. Returning fallback abstention.")
                return StructuredRAGOutput(
                    answer="I couldn't find enough evidence in the audio to answer that reliably.",
                    evidence_ids=[],
                    grounded=False,
                )

    def _parse_json_response(self, text: str) -> StructuredRAGOutput:
        """Parse text as JSON and validate against StructuredRAGOutput schema."""
        cleaned = text.strip()
        # Handle code block markdown if present
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        data = json.loads(cleaned)
        return StructuredRAGOutput(**data)

    def _lookup_chunk_from_repo(self, chunk_id: str) -> Optional[TranscriptChunk]:
        """Try resolving chunk_id directly from repository."""
        if not self.repository or not chunk_id:
            return None
        try:
            # Parse audio_id prefix if chunk_id follows audio_id_chk_XXXX format
            parts = chunk_id.split("_chk_")
            if len(parts) == 2:
                audio_id = parts[0]
                chunks = self.repository.get_chunks(audio_id)
                return next((c for c in chunks if c.chunk_id == chunk_id), None)
            return None
        except Exception as exc:
            logger.debug(f"Repo chunk lookup failed for {chunk_id}: {exc}")
            return None

    def _build_abstention_response(
        self,
        query: str,
        retrieved_chunks: Optional[List[RetrievalResult]] = None,
        reason: str = "",
        processing_time: float = 0.0,
        query_intent: Optional[QueryIntent] = None,
    ) -> RAGResponse:
        raw_model = getattr(self.llm_provider, "model_name", "Qwen3-8B")
        model_name = str(raw_model) if isinstance(raw_model, str) else "Qwen3-8B"
        return RAGResponse(
            answer="I couldn't find enough evidence in the audio to answer that reliably.",
            confidence=0.0,
            grounded=False,
            citations=[],
            retrieved_chunks=retrieved_chunks or [],
            query=query,
            processing_time=processing_time,
            model=model_name,
            retrieval_metadata={"abstain_reason": reason},
            abstained=True,
            confidence_reason=reason,
            intent=query_intent.intent if query_intent else None,
        )
