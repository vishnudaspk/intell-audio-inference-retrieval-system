"""
Pydantic data models establishing domain contracts across services.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

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
