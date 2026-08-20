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


class Citation(BaseModel):
    """Represents an application-resolved, timestamp-grounded source citation."""

    audio_id: str
    chunk_id: str
    start_time: float
    end_time: float
    text: str


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


# ─────────────────────────────────────────────────────────────────────────────
# V3 Audio Intelligence Models
# ─────────────────────────────────────────────────────────────────────────────


class AudioSegment(BaseModel):
    """
    V3 unified audio segment representation.

    Aggregates:
    - VAD interval (start_sec / end_sec / vad_confidence)
    - Whisper ASR transcript text and per-word data
    - SpeechBrain ECAPA-TDNN speaker embedding (192-dim, L2-normalized)
    - Librosa acoustic features (pitch, energy, spectral)
    - Sequence metadata (audio_id, segment index)
    - V3.2 CASA attribution confidence and decision (optional, backward-compatible)
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    audio_id: str

    # VAD timing
    start_sec: float
    end_sec: float
    duration_sec: float = 0.0
    vad_confidence: float = 0.0

    # ASR
    text: str = ""
    language: str = "en"
    whisper_segment_id: Optional[int] = None
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    words: List[TranscriptWord] = Field(default_factory=list)

    # Speaker representation & intelligence
    speaker_label: Optional[str] = None
    speaker_id: Optional[str] = None
    speaker_embedding: Optional[List[float]] = None

    # Acoustic features (stored as nested JSON)
    acoustic_features: Optional[Dict[str, Any]] = None

    # V3.2 CASA attribution confidence & decision
    speaker_confidence: Optional[float] = None
    """CASA fused confidence score in [0.0, 1.0]. None when CASA is disabled."""
    attribution_decision: Optional[str] = None
    """One of 'CONFIRM', 'CORRECT', 'UNCERTAIN'.  None when CASA is disabled."""
    attribution_evidence: Optional[List[str]] = None
    """Human-readable list of evidence signals used for this attribution."""
    provisional: Optional[bool] = None
    """True when this phrase falls inside the early-dialogue stabilization window."""

    # Metadata
    sequence_order: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)

