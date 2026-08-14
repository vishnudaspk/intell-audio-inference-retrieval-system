"""
FastAPI application backend exposing /health and /api/v1/ endpoints.
"""

from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile, status

from backend.schemas import (
    IngestUploadResponse,
    SearchRequest,
    SearchResponse,
    TranscriptResponse,
    YouTubeIngestRequest,
)
from database.sqlite_db import SQLiteRepository
from retrieval.lexical import LexicalRetrievalEngine
from services.audio_service import AudioService
from services.health_service import HealthService
from utils.exceptions import IntellAudioError
from utils.logger import logger
from workers.audio_worker import AudioWorker

app = FastAPI(
    title="Intell Audio Inference & Retrieval API",
    description="2026-grade Temporal Audio Intelligence & Retrieval Platform API",
    version="2.0.0-phase1",
)

# Application services & repository initialization
repo = SQLiteRepository()
audio_service = AudioService()
worker = AudioWorker(repository=repo, audio_service=audio_service)
retrieval_engine = LexicalRetrievalEngine()


@app.get("/health", tags=["Health"])
def health_check() -> Dict[str, Any]:
    """Get overall system health and dependency readiness status."""
    return HealthService.check_health()


@app.post(
    "/api/v1/ingest/upload",
    response_model=IngestUploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestion"],
)
async def ingest_file_upload(file: UploadFile = File(...)):
    """Ingest uploaded audio file and run processing pipeline."""
    try:
        content = await file.read()
        asset = audio_service.save_uploaded_file(content, file.filename)
        job = worker.process_asset(asset)
        return IngestUploadResponse(asset=asset, job=job)
    except IntellAudioError as exc:
        logger.error(f"API upload failed: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"API upload internal error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.post(
    "/api/v1/ingest/youtube",
    response_model=IngestUploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestion"],
)
def ingest_youtube_audio(payload: YouTubeIngestRequest):
    """Ingest audio from YouTube video URL and run processing pipeline."""
    try:
        asset = audio_service.download_youtube_audio(payload.url)
        job = worker.process_asset(asset)
        return IngestUploadResponse(asset=asset, job=job)
    except IntellAudioError as exc:
        logger.error(f"API YouTube ingest failed: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"API YouTube ingest internal error: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.get("/api/v1/assets/{audio_id}", tags=["Assets"])
def get_asset_metadata(audio_id: str):
    """Get metadata for an ingested audio asset."""
    asset = repo.get_audio_asset(audio_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Audio asset not found")
    return asset


@app.get(
    "/api/v1/assets/{audio_id}/transcript",
    response_model=TranscriptResponse,
    tags=["Assets"],
)
def get_asset_transcript(audio_id: str):
    """Get transcript and word-level timestamp alignments for an audio asset."""
    transcript = repo.get_transcript(audio_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found for this audio asset")
    return TranscriptResponse(
        audio_id=transcript.audio_id,
        text=transcript.text,
        language=transcript.language.value,
        words=transcript.words,
    )


@app.post(
    "/api/v1/search",
    response_model=SearchResponse,
    tags=["Retrieval"],
)
def search_transcript(payload: SearchRequest):
    """Perform exact word or phrase lexical search against transcript timestamps."""
    transcript = repo.get_transcript(payload.audio_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found for search")

    results = retrieval_engine.search(transcript.words, payload.query)
    return SearchResponse(
        audio_id=payload.audio_id,
        query=payload.query,
        results_count=len(results),
        results=results,
    )
