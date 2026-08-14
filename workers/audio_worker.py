"""
Worker pipeline coordinating audio ingestion, ASR transcription, Gentle forced alignment, and DB storage.
"""

import time
from datetime import datetime
from typing import Optional

from database.base import BaseRepository
from database.sqlite_db import SQLiteRepository
from schemas.enums import JobStatus
from schemas.models import AudioAsset, ProcessingJob
from services.alignment_service import AlignmentService
from services.audio_service import AudioService
from services.transcription_service import TranscriptionService
from utils.exceptions import IntellAudioError
from utils.logger import logger


class AudioWorker:
    """Pipeline orchestrator for processing audio assets."""

    def __init__(
        self,
        repository: Optional[BaseRepository] = None,
        audio_service: Optional[AudioService] = None,
        transcription_service: Optional[TranscriptionService] = None,
        alignment_service: Optional[AlignmentService] = None,
    ):
        self.repo = repository or SQLiteRepository()
        self.audio_service = audio_service or AudioService()
        self.transcription_service = transcription_service or TranscriptionService()
        self.alignment_service = alignment_service or AlignmentService()

    def process_asset(self, asset: AudioAsset) -> ProcessingJob:
        """Run the full ingestion -> ASR -> alignment -> storage pipeline for an audio asset."""
        job = ProcessingJob(
            audio_id=asset.id,
            status=JobStatus.CREATED,
        )
        self.repo.save_job(job)
        self.repo.save_audio_asset(asset)

        start_total = time.time()
        timings = {}

        try:
            # 1. Validation & Normalization
            logger.info(f"[Job {job.id}] Audio {asset.id} stage=NORMALIZING")
            job.status = JobStatus.NORMALIZING
            job.updated_at = datetime.utcnow()
            self.repo.save_job(job)

            wav_path = self.audio_service.convert_to_wav(asset)

            # Update asset duration in database after conversion
            self.repo.save_audio_asset(asset)

            # 2. Transcription (ASR)
            logger.info(f"[Job {job.id}] Audio {asset.id} stage=TRANSCRIBING")
            job.status = JobStatus.TRANSCRIBING
            job.updated_at = datetime.utcnow()
            self.repo.save_job(job)

            t_asr_start = time.time()
            transcript = self.transcription_service.transcribe_audio(
                audio_id=asset.id,
                wav_path=wav_path,
            )
            t_asr_end = time.time()
            timings["transcription_sec"] = round(t_asr_end - t_asr_start, 3)

            # 3. Alignment (Gentle)
            logger.info(f"[Job {job.id}] Audio {asset.id} stage=ALIGNING")
            job.status = JobStatus.ALIGNING
            job.updated_at = datetime.utcnow()
            self.repo.save_job(job)

            t_align_start = time.time()
            alignment_result = self.alignment_service.align_transcript(
                wav_path=wav_path,
                transcript_text=transcript.text,
            )
            t_align_end = time.time()
            timings["alignment_sec"] = round(t_align_end - t_align_start, 3)

            # 4. Persistence
            logger.info(f"[Job {job.id}] Audio {asset.id} stage=PERSISTING")
            job.status = JobStatus.PERSISTING
            job.updated_at = datetime.utcnow()
            self.repo.save_job(job)

            transcript.words = alignment_result.words
            self.repo.save_transcript(transcript)
            self.repo.save_alignment_words(asset.id, alignment_result.words)

            t_total_end = time.time()
            timings["total_sec"] = round(t_total_end - start_total, 3)
            timings["audio_duration_sec"] = round(asset.duration, 3)

            job.status = JobStatus.COMPLETED
            job.timings = timings
            job.updated_at = datetime.utcnow()
            self.repo.save_job(job)

            logger.info(f"[Job {job.id}] Audio {asset.id} stage=COMPLETED in {timings['total_sec']}s")
            return job

        except Exception as exc:
            logger.error(f"[Job {job.id}] Audio {asset.id} stage=FAILED error={exc}")
            job.status = JobStatus.FAILED
            job.error_message = str(exc)
            job.updated_at = datetime.utcnow()
            self.repo.save_job(job)
            raise IntellAudioError(f"Pipeline job failed: {exc}") from exc
