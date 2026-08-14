"""
Content Semantic Analysis service extracting topics, intents, actions, targets, and structured metadata.
"""

import json
from typing import Dict, List, Optional

from config.settings import settings
from schemas.models import TranscriptChunk
from services.llm_service import LLMProvider, LMStudioLLMProvider
from utils.logger import logger

SYSTEM_ANALYSIS_PROMPT = """You are a technical audio content semantic analyzer.
Your task is to analyze transcript chunks and extract structured semantic metadata.

For each chunk, extract:
- chunk_id: Must match the provided chunk ID exactly.
- content_type: One of [introduction, explanation, instruction, procedure, demonstration, definition, comparison, question, answer, warning, recommendation, troubleshooting, summary, discussion, conclusion, unknown].
- intent: High-level intent (e.g., remove_component, explain_function, compare_options, issue_warning).
- topic: Main subject/domain (e.g., turbo removal, oil change).
- subtopic: Specific aspect discussed.
- actions: List of action verbs (e.g., ["remove", "unscrew", "disconnect"]).
- objects: List of physical/logical items acted upon (e.g., ["bolt", "hose", "clip"]).
- targets: List of target components/systems (e.g., ["turbo housing", "manifold"]).
- entities: Named entities, brands, models, codes.
- tools: Tools or equipment mentioned (e.g., ["13mm socket", "ratchet"]).
- parts: Replacement/component parts mentioned.
- locations: Locations on assembly/device (e.g., ["underneath", "top left"]).
- quantities: Numbers and measurements (e.g., ["two", "13mm"]).
- conditions: Prerequisites or conditional statements.
- warnings: Any cautions, hazards, or risks mentioned.
- outcomes: Resulting state or expected outcome.
- temporal_references: Temporal sequencing markers (e.g., ["first", "before", "after"]).
- procedure_step: Integer step number if explicitly part of a numbered/ordered procedure, else null.
- chunk_summary: 1-sentence concise factual summary of the chunk.

Return ONLY a valid JSON array of chunk metadata objects.
"""


class ContentAnalyzer:
    """
    Batched semantic content analysis engine for transcript chunks.
    Configurable, graceful degradation on failure or context limit.
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm_provider = llm_provider or LMStudioLLMProvider()
        self.enabled = settings.ENABLE_CONTENT_ANALYSIS
        self.batch_size = settings.CONTENT_ANALYSIS_BATCH_SIZE
        self.max_chars = settings.MAX_CONTENT_ANALYSIS_CHARS
        self.max_retries = settings.CONTENT_ANALYSIS_MAX_RETRIES

    def analyze_chunks(self, chunks: List[TranscriptChunk]) -> List[TranscriptChunk]:
        """
        Analyze chunks in batches and populate semantic metadata fields.
        Returns chunks with updated fields (or unchanged if disabled/failed).
        """
        if not self.enabled or not chunks:
            return chunks

        if not self.llm_provider.is_available():
            logger.warning("LLM provider unavailable for content analysis. Skipping.")
            return chunks

        logger.info(f"Starting batched content analysis for {len(chunks)} chunks (batch size: {self.batch_size}).")

        # Process in batches
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            try:
                self._process_batch(batch)
            except Exception as exc:
                logger.warning(f"Batch content analysis failed for chunks {i}..{i + len(batch)}: {exc}")

        return chunks

    def _process_batch(self, batch: List[TranscriptChunk]) -> None:
        """Construct batched prompt, call LLM with retry, and apply metadata to chunks."""
        prompt_parts = ["Analyze the following transcript chunks:\n"]
        for c in batch:
            prompt_parts.append(f"[CHUNK_ID: {c.chunk_id}]\nTEXT: {c.text}\n")
        prompt = "\n".join(prompt_parts)

        # Truncate if exceeds max_chars
        if len(prompt) > self.max_chars:
            prompt = prompt[: self.max_chars]

        metadata_list = self._call_llm_with_retry(prompt)
        if not metadata_list:
            return

        # Map results back to chunks by chunk_id
        meta_map: Dict[str, dict] = {
            m.get("chunk_id", ""): m for m in metadata_list if isinstance(m, dict) and "chunk_id" in m
        }

        for chunk in batch:
            data = meta_map.get(chunk.chunk_id)
            if not data:
                continue

            chunk.topic = data.get("topic")
            chunk.subtopic = data.get("subtopic")
            chunk.intent = data.get("intent")
            chunk.content_type = data.get("content_type", "unknown")
            chunk.actions = data.get("actions", []) or []
            chunk.objects = data.get("objects", []) or []
            chunk.targets = data.get("targets", []) or []
            chunk.entities = data.get("entities", []) or []
            chunk.tools = data.get("tools", []) or []
            chunk.parts = data.get("parts", []) or []
            chunk.locations = data.get("locations", []) or []
            chunk.quantities = data.get("quantities", []) or []
            chunk.conditions = data.get("conditions", []) or []
            chunk.warnings = data.get("warnings", []) or []
            chunk.outcomes = data.get("outcomes", []) or []
            chunk.temporal_references = data.get("temporal_references", []) or []
            chunk.procedure_step = data.get("procedure_step")
            chunk.chunk_summary = data.get("chunk_summary")

            # Store in chunk.metadata dict for persistence/Qdrant
            chunk.metadata.update(
                {
                    "topic": chunk.topic,
                    "subtopic": chunk.subtopic,
                    "intent": chunk.intent,
                    "content_type": chunk.content_type,
                    "actions": chunk.actions,
                    "objects": chunk.objects,
                    "targets": chunk.targets,
                    "entities": chunk.entities,
                    "tools": chunk.tools,
                    "parts": chunk.parts,
                    "locations": chunk.locations,
                    "quantities": chunk.quantities,
                    "conditions": chunk.conditions,
                    "warnings": chunk.warnings,
                    "outcomes": chunk.outcomes,
                    "temporal_references": chunk.temporal_references,
                    "procedure_step": chunk.procedure_step,
                    "chunk_summary": chunk.chunk_summary,
                }
            )

    def _call_llm_with_retry(self, prompt: str) -> Optional[List[dict]]:
        """Call LLM with retries and parse JSON array output."""
        for attempt in range(self.max_retries):
            try:
                raw = self.llm_provider.generate(
                    prompt=prompt,
                    system_prompt=SYSTEM_ANALYSIS_PROMPT,
                    temperature=0.1,
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
                if isinstance(parsed, list):
                    return parsed
                elif isinstance(parsed, dict) and "chunks" in parsed:
                    return parsed["chunks"]
                elif isinstance(parsed, dict):
                    return [parsed]
            except Exception as exc:
                logger.debug(f"Content analysis LLM attempt {attempt + 1} failed: {exc}")
        return None
