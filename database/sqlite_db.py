"""
SQLite implementation of the repository interface.
"""

import csv
import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from config.settings import settings
from database.base import BaseRepository
from schemas.enums import JobStatus, LanguageCode, SourceType
from schemas.models import AudioAsset, ProcessingJob, Transcript, TranscriptWord
from utils.exceptions import StorageError
from utils.logger import logger


class SQLiteRepository(BaseRepository):
    """SQLite database repository for local metadata and transcript persistence."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create database tables if they do not exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audio_assets (
                        id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        format TEXT NOT NULL,
                        duration REAL DEFAULT 0.0,
                        source_type TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS processing_jobs (
                        id TEXT PRIMARY KEY,
                        audio_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        error_message TEXT,
                        timings_json TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (audio_id) REFERENCES audio_assets(id)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS transcripts (
                        id TEXT PRIMARY KEY,
                        audio_id TEXT NOT NULL UNIQUE,
                        text TEXT NOT NULL,
                        language TEXT NOT NULL,
                        duration REAL DEFAULT 0.0,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (audio_id) REFERENCES audio_assets(id)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS transcript_words (
                        id TEXT PRIMARY KEY,
                        audio_id TEXT NOT NULL,
                        word TEXT NOT NULL,
                        start_time REAL,
                        end_time REAL,
                        confidence REAL,
                        sequence_order INTEGER,
                        FOREIGN KEY (audio_id) REFERENCES audio_assets(id)
                    )
                    """
                )
                conn.commit()
        except Exception as exc:
            logger.error(f"Failed to initialize SQLite database: {exc}")
            raise StorageError(f"Database initialization error: {exc}") from exc

    def save_audio_asset(self, asset: AudioAsset) -> AudioAsset:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO audio_assets
                    (id, filename, file_path, format, duration, source_type, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        asset.id,
                        asset.filename,
                        asset.file_path,
                        asset.format,
                        asset.duration,
                        asset.source_type.value,
                        asset.created_at.isoformat(),
                    ),
                )
                conn.commit()
            return asset
        except Exception as exc:
            logger.error(f"Failed to save audio asset {asset.id}: {exc}")
            raise StorageError(f"Failed to save audio asset: {exc}") from exc

    def get_audio_asset(self, audio_id: str) -> Optional[AudioAsset]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM audio_assets WHERE id = ?", (audio_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return AudioAsset(
                    id=row["id"],
                    filename=row["filename"],
                    file_path=row["file_path"],
                    format=row["format"],
                    duration=row["duration"],
                    source_type=SourceType(row["source_type"]),
                )
        except Exception as exc:
            logger.error(f"Failed to retrieve audio asset {audio_id}: {exc}")
            raise StorageError(f"Failed to get audio asset: {exc}") from exc

    def save_job(self, job: ProcessingJob) -> ProcessingJob:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO processing_jobs
                    (id, audio_id, status, error_message, timings_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.id,
                        job.audio_id,
                        job.status.value,
                        job.error_message,
                        json.dumps(job.timings),
                        job.created_at.isoformat(),
                        job.updated_at.isoformat(),
                    ),
                )
                conn.commit()
            return job
        except Exception as exc:
            logger.error(f"Failed to save processing job {job.id}: {exc}")
            raise StorageError(f"Failed to save processing job: {exc}") from exc

    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM processing_jobs WHERE id = ?", (job_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return ProcessingJob(
                    id=row["id"],
                    audio_id=row["audio_id"],
                    status=JobStatus(row["status"]),
                    error_message=row["error_message"],
                    timings=json.loads(row["timings_json"]) if row["timings_json"] else {},
                )
        except Exception as exc:
            logger.error(f"Failed to retrieve processing job {job_id}: {exc}")
            raise StorageError(f"Failed to get processing job: {exc}") from exc

    def save_transcript(self, transcript: Transcript) -> Transcript:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO transcripts
                    (id, audio_id, text, language, duration, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        transcript.id,
                        transcript.audio_id,
                        transcript.text,
                        transcript.language.value,
                        transcript.duration,
                        transcript.created_at.isoformat(),
                    ),
                )
                conn.commit()

            # Export legacy text file for backward compatibility if configured
            legacy_txt = settings.transcript_dir / f"{transcript.audio_id}.txt"
            with open(legacy_txt, "w", encoding="utf-8") as f:
                f.write(transcript.text)

            return transcript
        except Exception as exc:
            logger.error(f"Failed to save transcript for audio {transcript.audio_id}: {exc}")
            raise StorageError(f"Failed to save transcript: {exc}") from exc

    def get_transcript(self, audio_id: str) -> Optional[Transcript]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM transcripts WHERE audio_id = ?", (audio_id,))
                row = cursor.fetchone()
                if not row:
                    return None

                words = self.get_alignment_words(audio_id)
                return Transcript(
                    id=row["id"],
                    audio_id=row["audio_id"],
                    text=row["text"],
                    language=LanguageCode(row["language"]),
                    duration=row["duration"],
                    words=words,
                )
        except Exception as exc:
            logger.error(f"Failed to retrieve transcript for audio {audio_id}: {exc}")
            raise StorageError(f"Failed to get transcript: {exc}") from exc

    def save_alignment_words(self, audio_id: str, words: List[TranscriptWord]) -> None:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM transcript_words WHERE audio_id = ?", (audio_id,))

                records = [
                    (
                        w.id,
                        audio_id,
                        w.word,
                        w.start,
                        w.end,
                        w.confidence,
                        idx,
                    )
                    for idx, w in enumerate(words)
                ]

                cursor.executemany(
                    """
                    INSERT INTO transcript_words
                    (id, audio_id, word, start_time, end_time, confidence, sequence_order)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    records,
                )
                conn.commit()

            # Export legacy CSV file for backward compatibility
            legacy_csv = settings.alignment_dir / f"{audio_id}_alignment.csv"
            with open(legacy_csv, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["word", "start", "end"])
                for w in words:
                    writer.writerow([w.word, "" if w.start is None else w.start, "" if w.end is None else w.end])

        except Exception as exc:
            logger.error(f"Failed to save alignment words for audio {audio_id}: {exc}")
            raise StorageError(f"Failed to save alignment words: {exc}") from exc

    def get_alignment_words(self, audio_id: str) -> List[TranscriptWord]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM transcript_words WHERE audio_id = ? ORDER BY sequence_order ASC",
                    (audio_id,),
                )
                rows = cursor.fetchall()
                return [
                    TranscriptWord(
                        id=row["id"],
                        word=row["word"],
                        start=row["start_time"],
                        end=row["end_time"],
                        confidence=row["confidence"],
                    )
                    for row in rows
                ]
        except Exception as exc:
            logger.error(f"Failed to retrieve alignment words for audio {audio_id}: {exc}")
            raise StorageError(f"Failed to get alignment words: {exc}") from exc
