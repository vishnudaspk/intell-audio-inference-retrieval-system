"""
FastAPI application backend exposing /health and /api/v1/ endpoints.
"""

from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.schemas import (
    AskRequest,
    AudioSegmentsResponse,
    IndexResponse,
    IngestUploadResponse,
    SearchRequest,
    SearchResponse,
    TranscriptResponse,
    YouTubeIngestRequest,
)
from database.sqlite_db import SQLiteRepository
from retrieval.bm25 import BM25Index
from retrieval.hybrid import RetrievalPipeline
from retrieval.lexical import LexicalRetrievalEngine
from retrieval.vector_store import QdrantVectorStore
from schemas.models import AudioAsset, ProcessingJob, RAGResponse
from services.audio_service import AudioService
from services.embedding_service import LMStudioEmbeddingProvider
from services.health_service import HealthService
from services.llm_service import LMStudioLLMProvider
from services.reasoning_agent import ReasoningAgent
from utils.exceptions import IntellAudioError
from utils.logger import logger
from workers.audio_worker import AudioWorker
from workers.indexing_worker import IndexingWorker

app = FastAPI(
    title="Intell Audio Inference & Retrieval API",
    description="2026-grade Temporal Audio Intelligence & Retrieval Platform API (V3)",
    version="3.0.0-phase1",
)

# Enable CORS for React Vite frontend (ports 5173, 3000, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Application services & repository initialization
repo = SQLiteRepository()
audio_service = AudioService()
worker = AudioWorker(repository=repo, audio_service=audio_service)
retrieval_engine = LexicalRetrievalEngine()

# Hybrid retrieval & RAG services
bm25_index = BM25Index()
vector_store = QdrantVectorStore()
embedding_provider = LMStudioEmbeddingProvider()
llm_provider = LMStudioLLMProvider()

indexing_worker = IndexingWorker(
    repository=repo,
    bm25_index=bm25_index,
    vector_store=vector_store,
    embedding_provider=embedding_provider,
)

retrieval_pipeline = RetrievalPipeline(
    bm25_index=bm25_index,
    vector_store=vector_store,
    embedding_provider=embedding_provider,
    repository=repo,
)

reasoning_agent = ReasoningAgent(
    llm_provider=llm_provider,
    repository=repo,
)


@app.get("/health", tags=["Health"])
@app.get("/api/v1/health", tags=["Health"])
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
    """Ingest uploaded audio/video file and run V3 Phase 1 pipeline."""
    try:
        content = await file.read()
        asset = audio_service.save_uploaded_file(content, file.filename)
        job = worker.process_asset(asset)

        # Trigger chunking & indexing after transcription
        try:
            indexing_worker.index_audio(asset.id)
        except Exception as idx_exc:
            logger.warning(f"Post-ingestion indexing skipped or degraded: {idx_exc}")

        return IngestUploadResponse(asset=asset, job=job)
    except IntellAudioError as exc:
        logger.error(f"API upload failed: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"API upload internal error: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") from exc


@app.post(
    "/api/v1/ingest/youtube",
    response_model=IngestUploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestion"],
)
def ingest_youtube_audio(payload: YouTubeIngestRequest):
    """Ingest audio from YouTube video URL and run V3 Phase 1 pipeline."""
    try:
        asset = audio_service.download_youtube_audio(payload.url)
        job = worker.process_asset(asset)

        # Trigger chunking & indexing after transcription
        try:
            indexing_worker.index_audio(asset.id)
        except Exception as idx_exc:
            logger.warning(f"Post-ingestion indexing skipped or degraded: {idx_exc}")

        return IngestUploadResponse(asset=asset, job=job)
    except IntellAudioError as exc:
        logger.error(f"API YouTube ingest failed: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"API YouTube ingest internal error: {exc}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {exc}") from exc


@app.get("/api/v1/assets", response_model=List[AudioAsset], tags=["Assets"])
def list_assets():
    """List all ingested media assets."""
    return repo.get_all_audio_assets()


@app.get("/api/v1/assets/{audio_id}", response_model=AudioAsset, tags=["Assets"])
def get_asset_metadata(audio_id: str):
    """Get metadata for an ingested audio asset."""
    asset = repo.get_audio_asset(audio_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Audio asset not found")
    return asset


@app.delete("/api/v1/assets/{audio_id}", tags=["Assets"])
def delete_asset(audio_id: str):
    """Delete an ingested media asset and all associated transcript, segments, and indexes."""
    asset = repo.get_audio_asset(audio_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Audio asset not found")

    try:
        # Delete from vector store & BM25
        try:
            vector_store.delete_audio(audio_id)
        except Exception as e:
            logger.warning(f"Failed to delete vector index for {audio_id}: {e}")

        try:
            bm25_index.delete_audio(audio_id)
        except Exception as e:
            logger.warning(f"Failed to delete BM25 index for {audio_id}: {e}")

        # Delete from database and file system
        repo.delete_audio_asset(audio_id)
        return {"status": "deleted", "audio_id": audio_id}
    except Exception as exc:
        logger.error(f"Failed to delete asset {audio_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Failed to delete asset: {exc}") from exc


@app.post(
    "/api/v1/assets/{audio_id}/process",
    response_model=IngestUploadResponse,
    status_code=status.HTTP_200_OK,
    tags=["Ingestion"],
)
def reprocess_asset(audio_id: str):
    """Re-run the full V3 pipeline (VAD → Whisper → Speaker Embeddings → Acoustics) on an existing catalog asset."""
    asset = repo.get_audio_asset(audio_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Audio asset not found")

    try:
        job = worker.process_asset(asset)

        # Re-index after processing
        try:
            indexing_worker.index_audio(asset.id)
        except Exception as idx_exc:
            logger.warning(f"Post-reprocessing indexing skipped or degraded: {idx_exc}")

        # Return fresh asset (duration may be updated after processing)
        updated_asset = repo.get_audio_asset(audio_id) or asset
        return IngestUploadResponse(asset=updated_asset, job=job)
    except IntellAudioError as exc:
        logger.error(f"API reprocess failed for {audio_id}: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"API reprocess internal error for {audio_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Reprocess failed: {exc}") from exc


@app.get("/api/v1/assets/{audio_id}/media", tags=["Assets"])
def get_asset_media_file(audio_id: str):
    """Stream or download the media file (or normalized WAV) for playback."""
    asset = repo.get_audio_asset(audio_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Audio asset not found")

    file_path = Path(asset.file_path)
    if not file_path.exists():
        # Fallback to normalized WAV if raw original not present
        wav_fallback = audio_service.audio_dir / f"{audio_id}.wav"
        if wav_fallback.exists():
            file_path = wav_fallback
        else:
            raise HTTPException(status_code=404, detail="Media file not found on disk")

    media_type = "audio/wav"
    if file_path.suffix.lower() == ".mp4":
        media_type = "video/mp4"
    elif file_path.suffix.lower() == ".mp3":
        media_type = "audio/mpeg"
    elif file_path.suffix.lower() == ".ogg":
        media_type = "audio/ogg"
    elif file_path.suffix.lower() == ".flac":
        media_type = "audio/flac"
    elif file_path.suffix.lower() == ".m4a":
        media_type = "audio/mp4"

    return FileResponse(path=str(file_path), media_type=media_type, filename=file_path.name)


@app.get("/api/v1/assets/{audio_id}/jobs/{job_id}", response_model=ProcessingJob, tags=["Jobs"])
def get_job_status(audio_id: str, job_id: str):
    """Get status and performance timings for an audio processing job."""
    job = repo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


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


@app.get(
    "/api/v1/assets/{audio_id}/segments",
    response_model=AudioSegmentsResponse,
    tags=["V3 Intelligence"],
)
def get_asset_segments(audio_id: str):
    """Get unified V3 AudioSegment objects with VAD intervals, speaker embeddings, and acoustics."""
    segments = repo.get_audio_segments(audio_id)
    return AudioSegmentsResponse(
        audio_id=audio_id,
        total_segments=len(segments),
        segments=segments,
    )


@app.post(
    "/api/v1/index/{audio_id}",
    response_model=IndexResponse,
    tags=["Indexing"],
)
def index_audio_asset(audio_id: str):
    """Trigger explicit chunking, BM25 indexing, and vector embedding for an audio asset."""
    try:
        idx_status = indexing_worker.index_audio(audio_id)
        return IndexResponse(
            audio_id=idx_status.audio_id,
            status=idx_status.status,
            total_chunks=idx_status.total_chunks,
            indexed_chunks=idx_status.indexed_chunks,
            embedding_model=idx_status.embedding_model,
            error_message=idx_status.error_message,
        )
    except IntellAudioError as exc:
        logger.error(f"Indexing endpoint error: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Indexing endpoint internal error: {exc}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc


@app.get(
    "/api/v1/index/{audio_id}",
    response_model=IndexResponse,
    tags=["Indexing"],
)
def get_indexing_status(audio_id: str):
    """Get vector & BM25 indexing status for an audio asset."""
    status_obj = repo.get_indexing_status(audio_id)
    if not status_obj:
        raise HTTPException(status_code=404, detail="Indexing status not found for this asset")
    return IndexResponse(
        audio_id=status_obj.audio_id,
        status=status_obj.status,
        total_chunks=status_obj.total_chunks,
        indexed_chunks=status_obj.indexed_chunks,
        embedding_model=status_obj.embedding_model,
        error_message=status_obj.error_message,
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


@app.post(
    "/api/v1/ask",
    response_model=RAGResponse,
    tags=["Ask the Audio RAG"],
)
def ask_audio(payload: AskRequest):
    """Ask natural-language question about indexed audio and get grounded answer with timestamp citations."""
    try:
        top_k = payload.top_k or 10
        final_k = payload.final_k or 5

        # 1. Deterministic hybrid retrieval
        retrieved_chunks = retrieval_pipeline.search(
            query=payload.query,
            top_k=top_k,
            final_k=final_k,
            audio_id=payload.audio_id,
        )

        # 2. Grounded reasoning and application citation resolution
        rag_response = reasoning_agent.answer_question(
            query=payload.query,
            retrieved_chunks=retrieved_chunks,
            audio_id=payload.audio_id,
        )

        return rag_response

    except IntellAudioError as exc:
        logger.error(f"Ask the Audio endpoint error: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"Ask the Audio endpoint internal error: {exc}")
        raise HTTPException(status_code=500, detail=f"Ask the Audio failed: {exc}") from exc
