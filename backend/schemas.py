"""
Pydantic Request/Response schemas for the FastAPI backend endpoints.
"""

from typing import List, Optional

from pydantic import BaseModel

from schemas.models import AudioAsset, Chapter, ProcessingJob, SearchResult, SpeakerSegment, TranscriptWord


class IngestUploadResponse(BaseModel):
    asset: AudioAsset
    job: ProcessingJob


class YouTubeIngestRequest(BaseModel):
    url: str


class SearchRequest(BaseModel):
    audio_id: str
    query: str


class SearchResponse(BaseModel):
    audio_id: str
    query: str
    results_count: int
    results: List[SearchResult]


class TranscriptResponse(BaseModel):
    audio_id: str
    text: str
    language: str
    words: List[TranscriptWord]


class AskRequest(BaseModel):
    query: str
    audio_id: Optional[str] = None
    top_k: Optional[int] = None
    final_k: Optional[int] = None


class IndexResponse(BaseModel):
    audio_id: str
    status: str
    total_chunks: int
    indexed_chunks: int
    embedding_model: str
    error_message: Optional[str] = None


class ChaptersResponse(BaseModel):
    """Phase 7B: Chapters for a given audio asset."""
    audio_id: str
    chapters: List[Chapter]
    count: int


class SpeakersResponse(BaseModel):
    """Phase 7B: Speaker segments for a given audio asset."""
    audio_id: str
    segments: List[SpeakerSegment]
    count: int
    note: str = "Speaker labels are heuristic turn estimates. No real speaker identity is verified."

