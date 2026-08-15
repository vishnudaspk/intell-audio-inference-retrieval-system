"""
Pydantic Request/Response schemas for the FastAPI backend endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from schemas.models import (
    AudioAsset,
    AudioSegment,
    ProcessingJob,
    SearchResult,
    TranscriptSegment,
    TranscriptWord,
)


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
    segments: Optional[List[TranscriptSegment]] = None


class AudioSegmentsResponse(BaseModel):
    audio_id: str
    total_segments: int
    segments: List[AudioSegment]


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
