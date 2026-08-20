"""
Grounded Reasoning Agent and Application-side Timestamp Citation Resolver.
Synthesizes natural language answers from pre-retrieved evidence while enforcing strict grounding constraints.
"""

import json
import time
from typing import Dict, List, Optional

from config.settings import settings
from database.base import BaseRepository
from schemas.models import Citation, RAGResponse, RetrievalResult, StructuredRAGOutput, TranscriptChunk
from services.llm_service import LLMProvider
from utils.exceptions import IntellAudioError
from utils.logger import logger


SYSTEM_GROUNDING_PROMPT = """You are a helpful, articulate, and conversational AI audio assistant (like ChatGPT or Gemini).
Your goal is to answer the user's question clearly, warmly, and naturally, using the information discussed in the provided audio transcript chunks.

CONVERSATIONAL GUIDELINES:
1. Write in a fluent, engaging, and natural human tone. Avoid robotic, repetitive, or overly rigid phrases like "According to chunk_001..." or "The provided evidence states...".
2. Synthesize facts into clean, coherent paragraphs or bullet points where helpful.
3. Stay strictly accurate to what was actually discussed in the audio. Do not hallucinate or extrapolate facts not present in the dialogue.
4. Do NOT output raw timestamp codes inside your prose answer. The platform will automatically attach clickable timestamp citations to your answer.
5. In the "evidence_ids" list, include the chunk IDs (e.g. "audio_id_chk_0000") that directly support your points.
6. If the audio does not mention or contain enough information to address the query, politely and naturally explain that it wasn't covered in this recording.
7. Return your response formatted as valid JSON matching the schema below.

JSON SCHEMA:
{
  "answer": "A clear, natural, and comprehensive conversational answer...",
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
    ) -> RAGResponse:
        """
        Synthesize grounded natural language answer and resolve citations.
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
                citations.append(
                    Citation(
                        audio_id=chunk.audio_id,
                        chunk_id=chunk.chunk_id,
                        start_time=chunk.start_time,
                        end_time=chunk.end_time,
                        text=chunk.text,
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
        )
