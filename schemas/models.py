"""
Pydantic data models establishing domain contracts across services.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from schemas.enums import JobStatus, LanguageCode, SourceType


class AudioAsset(BaseModel):
    """Represents an ingested audio file asset."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    file_path: str
    format: str
    duration: float = 0.0
    source_type: SourceType = SourceType.UPLOAD
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TranscriptWord(BaseModel):
    """Represents a single word with timestamp alignment."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    confidence: Optional[float] = None


class TranscriptSegment(BaseModel):
    """Represents a segment/sentence of a transcript."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    transcript_id: Optional[str] = None
    sequence_order: int = 0
    start: Optional[float] = None
    end: Optional[float] = None
    text: str
    words: List[TranscriptWord] = Field(default_factory=list)


class Transcript(BaseModel):
    """Represents a full transcript associated with an audio asset."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    audio_id: str
    text: str
    language: LanguageCode = LanguageCode.ENGLISH
    duration: float = 0.0
    segments: List[TranscriptSegment] = Field(default_factory=list)
    words: List[TranscriptWord] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AlignmentResult(BaseModel):
    """Result returned by Gentle forced alignment."""

    audio_id: str
    words: List[TranscriptWord] = Field(default_factory=list)
    raw_response: Optional[dict] = None


class ProcessingJob(BaseModel):
    """Tracks state and performance timing of an audio processing pipeline job."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    audio_id: str
    status: JobStatus = JobStatus.CREATED
    error_message: Optional[str] = None
    timings: Dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SearchResult(BaseModel):
    """Represents a matching word or phrase timestamp result."""

    matched_text: str
    start: float
    end: float
    confidence: Optional[float] = None
    word_index: Optional[int] = None


class SpeakerSegment(BaseModel):
    """A detected speaker turn with time bounds and confidence."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    audio_id: str
    speaker_id: Optional[str] = None
    speaker_label: str = "Unknown Speaker"
    start_time: float
    end_time: float
    confidence: float = 0.0


class Chapter(BaseModel):
    """A semantically coherent temporal section of audio."""

    chapter_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    audio_id: str
    title: str
    summary: Optional[str] = None
    start_time: float
    end_time: float
    dominant_topic: Optional[str] = None
    sequence_order: int = 0
    speaker_ids: List[str] = Field(default_factory=list)
    chunk_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TranscriptChunk(BaseModel):
    """Represents a temporal timestamp-preserving chunk of transcript text."""

    chunk_id: str
    audio_id: str
    transcript_id: str
    sequence_order: int = 0
    text: str
    start_time: float
    end_time: float
    words: List[TranscriptWord] = Field(default_factory=list)
    language: str = "en"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Phase 7A Speaker & Chapter metadata (Optional for backward compatibility)
    speaker_id: Optional[str] = None
    speaker_label: Optional[str] = None
    speaker_confidence: float = 0.0
    chapter_id: Optional[str] = None

    # Phase 7A Semantic Content metadata (Optional for backward compatibility)
    topic: Optional[str] = None
    subtopic: Optional[str] = None
    intent: Optional[str] = None
    content_type: Optional[str] = None
    actions: List[str] = Field(default_factory=list)
    objects: List[str] = Field(default_factory=list)
    targets: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    parts: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    quantities: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    outcomes: List[str] = Field(default_factory=list)
    temporal_references: List[str] = Field(default_factory=list)
    procedure_step: Optional[int] = None
    chunk_summary: Optional[str] = None



class QueryIntent(BaseModel):
    """Extracted intent and semantic attributes from a user's natural-language query."""

    query: str
    normalized_query: str
    intent: str = "unknown"
    actions: List[str] = Field(default_factory=list)
    objects: List[str] = Field(default_factory=list)
    targets: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    content_type_preferences: List[str] = Field(default_factory=list)
    topic: Optional[str] = None
    requires_llm: bool = False


class RelevantTemporalSpan(BaseModel):
    """Application-resolved temporal span grounded in transcript chunk data. LLM never generates this."""

    start_time: float
    end_time: float
    source_chunk_ids: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""


class Citation(BaseModel):
    """Represents an application-resolved, timestamp-grounded source citation."""

    audio_id: str
    chunk_id: str
    start_time: float
    end_time: float
    text: str
    # Phase 7B optional enrichment fields
    speaker_label: Optional[str] = None
    chapter_title: Optional[str] = None


class RetrievalResult(BaseModel):
    """Represents a chunk retrieved by hybrid BM25 and vector search with rank score."""

    chunk: TranscriptChunk
    retrieval_source: str
    score: float
    rank: int
    start_time: float
    end_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StructuredRAGOutput(BaseModel):
    """JSON output model expected from LLM reasoning stage."""

    answer: str
    evidence_ids: List[str] = Field(default_factory=list)
    grounded: bool = True


class RAGResponse(BaseModel):
    """Complete RAG query response payload."""

    answer: str
    confidence: float
    grounded: bool
    citations: List[Citation] = Field(default_factory=list)
    retrieved_chunks: List[RetrievalResult] = Field(default_factory=list)
    query: str
    processing_time: float
    model: str
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)
    # Phase 7B intent-aware enrichment (Optional, backward-compatible)
    primary_timestamp: Optional["RelevantTemporalSpan"] = None
    related_sections: List["RelevantTemporalSpan"] = Field(default_factory=list)
    speaker: Optional[str] = None
    chapter: Optional[str] = None
    intent: Optional[str] = None
    abstained: bool = False
    confidence_reason: str = ""
    evidence_summary: str = ""


class IndexingStatus(BaseModel):
    """Metadata tracking vector & BM25 indexing state for an audio asset."""

    audio_id: str
    status: str
    total_chunks: int = 0
    indexed_chunks: int = 0
    embedding_model: str
    embedding_dimension: int = 0
    embedding_version: str = "1.0"
    chunking_version: str = "1.0"
    error_message: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)

