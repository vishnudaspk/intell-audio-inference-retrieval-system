"""
Pydantic Request/Response schemas for the FastAPI backend endpoints.
"""

from typing import List

from pydantic import BaseModel

from schemas.models import AudioAsset, ProcessingJob, SearchResult, TranscriptWord


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
