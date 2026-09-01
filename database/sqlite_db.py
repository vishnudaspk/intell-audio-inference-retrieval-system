"""
SQLite implementation of the repository interface.
"""

import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config.settings import settings
from database.base import BaseRepository
from schemas.enums import JobStatus, LanguageCode, SourceType
from schemas.models import (
    AudioAsset,
    AudioSegment,
    IndexingStatus,
    ProcessingJob,
    Transcript,
    TranscriptChunk,
    TranscriptWord,
)
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
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS transcript_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        audio_id TEXT NOT NULL,
                        transcript_id TEXT NOT NULL,
                        sequence_order INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        start_time REAL NOT NULL,
                        end_time REAL NOT NULL,
                        words_json TEXT NOT NULL,
                        language TEXT NOT NULL,
                        metadata_json TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (audio_id) REFERENCES audio_assets(id)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS indexing_status (
                        audio_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        total_chunks INTEGER DEFAULT 0,
                        indexed_chunks INTEGER DEFAULT 0,
                        embedding_model TEXT NOT NULL,
                        embedding_dimension INTEGER DEFAULT 0,
                        embedding_version TEXT NOT NULL,
                        chunking_version TEXT NOT NULL,
                        error_message TEXT,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY (audio_id) REFERENCES audio_assets(id)
                    )
                    """
                )
                # V3: Audio segments table (VAD + ASR + Speaker Embeddings + Acoustics)
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audio_segments (
                        id TEXT PRIMARY KEY,
                        audio_id TEXT NOT NULL,
                        sequence_order INTEGER NOT NULL,
                        start_sec REAL NOT NULL,
                        end_sec REAL NOT NULL,
                        duration_sec REAL NOT NULL,
                        vad_confidence REAL DEFAULT 0.0,
                        text TEXT NOT NULL DEFAULT '',
                        language TEXT NOT NULL DEFAULT 'en',
                        speaker_label TEXT,
                        speaker_id TEXT,
                        whisper_segment_id INTEGER,
                        avg_logprob REAL,
                        no_speech_prob REAL,
                        words_json TEXT,
                        speaker_embedding_json TEXT,
                        acoustic_features_json TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (audio_id) REFERENCES audio_assets(id)
                    )
                    """
                )

                # Migration check: ensure speaker_label and speaker_id columns exist
                cursor.execute("PRAGMA table_info(audio_segments)")
                columns = [col["name"] for col in cursor.fetchall()]
                if "speaker_label" not in columns:
                    cursor.execute("ALTER TABLE audio_segments ADD COLUMN speaker_label TEXT")
                if "speaker_id" not in columns:
                    cursor.execute("ALTER TABLE audio_segments ADD COLUMN speaker_id TEXT")
                # V3.2 CASA migration
                if "speaker_confidence" not in columns:
                    cursor.execute("ALTER TABLE audio_segments ADD COLUMN speaker_confidence REAL")
                if "attribution_decision" not in columns:
                    cursor.execute("ALTER TABLE audio_segments ADD COLUMN attribution_decision TEXT")
                if "attribution_evidence_json" not in columns:
                    cursor.execute("ALTER TABLE audio_segments ADD COLUMN attribution_evidence_json TEXT")
                if "provisional" not in columns:
                    cursor.execute("ALTER TABLE audio_segments ADD COLUMN provisional INTEGER")
                # Analysis results table (V3.2+ Canonical AnalysisResult JSON)
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        job_id TEXT PRIMARY KEY,
                        audio_id TEXT NOT NULL,
                        result_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
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

    def get_all_audio_assets(self) -> List[AudioAsset]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM audio_assets ORDER BY created_at DESC")
                rows = cursor.fetchall()
                return [
                    AudioAsset(
                        id=row["id"],
                        filename=row["filename"],
                        file_path=row["file_path"],
                        format=row["format"],
                        duration=row["duration"],
                        source_type=SourceType(row["source_type"]),
                    )
                    for row in rows
                ]
        except Exception as exc:
            logger.error(f"Failed to retrieve all audio assets: {exc}")
            raise StorageError(f"Failed to get all audio assets: {exc}") from exc


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

    def save_chunks(self, audio_id: str, chunks: List[TranscriptChunk]) -> None:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM transcript_chunks WHERE audio_id = ?", (audio_id,))
                records = [
                    (
                        c.chunk_id,
                        c.audio_id,
                        c.transcript_id,
                        c.sequence_order,
                        c.text,
                        c.start_time,
                        c.end_time,
                        json.dumps([w.model_dump() for w in c.words]),
                        c.language,
                        json.dumps(c.metadata),
                        datetime.utcnow().isoformat(),
                    )
                    for c in chunks
                ]
                cursor.executemany(
                    """
                    INSERT INTO transcript_chunks
                    (chunk_id, audio_id, transcript_id, sequence_order, text, start_time, end_time, words_json, language, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    records,
                )
                conn.commit()
        except Exception as exc:
            logger.error(f"Failed to save transcript chunks for audio {audio_id}: {exc}")
            raise StorageError(f"Failed to save chunks: {exc}") from exc

    def get_chunks(self, audio_id: str) -> List[TranscriptChunk]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM transcript_chunks WHERE audio_id = ? ORDER BY sequence_order ASC",
                    (audio_id,),
                )
                rows = cursor.fetchall()
                return [self._parse_chunk_row(row) for row in rows]
        except Exception as exc:
            logger.error(f"Failed to retrieve chunks for audio {audio_id}: {exc}")
            raise StorageError(f"Failed to get chunks: {exc}") from exc

    def get_all_chunks(self) -> List[TranscriptChunk]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM transcript_chunks ORDER BY audio_id, sequence_order ASC")
                rows = cursor.fetchall()
                return [self._parse_chunk_row(row) for row in rows]
        except Exception as exc:
            logger.error(f"Failed to retrieve all transcript chunks: {exc}")
            raise StorageError(f"Failed to get all chunks: {exc}") from exc

    def delete_chunks(self, audio_id: str) -> None:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM transcript_chunks WHERE audio_id = ?", (audio_id,))
                cursor.execute("DELETE FROM indexing_status WHERE audio_id = ?", (audio_id,))
                conn.commit()
        except Exception as exc:
            logger.error(f"Failed to delete chunks for audio {audio_id}: {exc}")
            raise StorageError(f"Failed to delete chunks: {exc}") from exc

    def _parse_chunk_row(self, row: sqlite3.Row) -> TranscriptChunk:
        words_raw = json.loads(row["words_json"]) if row["words_json"] else []
        words = [TranscriptWord(**w) for w in words_raw]
        metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        return TranscriptChunk(
            chunk_id=row["chunk_id"],
            audio_id=row["audio_id"],
            transcript_id=row["transcript_id"],
            sequence_order=row["sequence_order"],
            text=row["text"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            words=words,
            language=row["language"],
            metadata=metadata,
        )

    def save_indexing_status(self, status: IndexingStatus) -> IndexingStatus:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO indexing_status
                    (audio_id, status, total_chunks, indexed_chunks, embedding_model, embedding_dimension, embedding_version, chunking_version, error_message, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        status.audio_id,
                        status.status,
                        status.total_chunks,
                        status.indexed_chunks,
                        status.embedding_model,
                        status.embedding_dimension,
                        status.embedding_version,
                        status.chunking_version,
                        status.error_message,
                        status.updated_at.isoformat(),
                    ),
                )
                conn.commit()
            return status
        except Exception as exc:
            logger.error(f"Failed to save indexing status for audio {status.audio_id}: {exc}")
            raise StorageError(f"Failed to save indexing status: {exc}") from exc

    def get_indexing_status(self, audio_id: str) -> Optional[IndexingStatus]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM indexing_status WHERE audio_id = ?", (audio_id,))
                row = cursor.fetchone()
                if not row:
                    return None
                return IndexingStatus(
                    audio_id=row["audio_id"],
                    status=row["status"],
                    total_chunks=row["total_chunks"],
                    indexed_chunks=row["indexed_chunks"],
                    embedding_model=row["embedding_model"],
                    embedding_dimension=row["embedding_dimension"],
                    embedding_version=row["embedding_version"],
                    chunking_version=row["chunking_version"],
                    error_message=row["error_message"],
                    updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else datetime.utcnow(),
                )
        except Exception as exc:
            logger.error(f"Failed to retrieve indexing status for audio {audio_id}: {exc}")
            raise StorageError(f"Failed to get indexing status: {exc}") from exc

    # ─────────────────────────────────────────────────────────────────────────
    # V3: Audio Segments (VAD + ASR + Speaker Embeddings + Acoustic Features)
    # ─────────────────────────────────────────────────────────────────────────

    def save_audio_segments(self, audio_id: str, segments: List[AudioSegment]) -> None:
        """Persist a list of V3 AudioSegment objects, replacing any existing segments for the asset."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM audio_segments WHERE audio_id = ?", (audio_id,))

                records = [
                    (
                        seg.id,
                        seg.audio_id,
                        seg.sequence_order,
                        seg.start_sec,
                        seg.end_sec,
                        seg.duration_sec,
                        seg.vad_confidence,
                        seg.text,
                        seg.language,
                        seg.speaker_label,
                        seg.speaker_id,
                        seg.whisper_segment_id,
                        seg.avg_logprob,
                        seg.no_speech_prob,
                        json.dumps([w.model_dump() for w in seg.words]) if seg.words else None,
                        json.dumps(seg.speaker_embedding) if seg.speaker_embedding is not None else None,
                        json.dumps(seg.acoustic_features) if seg.acoustic_features is not None else None,
                        seg.created_at.isoformat(),
                        # V3.2 CASA fields
                        seg.speaker_confidence,
                        seg.attribution_decision,
                        json.dumps(seg.attribution_evidence) if seg.attribution_evidence is not None else None,
                        1 if seg.provisional else (0 if seg.provisional is not None else None),
                    )
                    for seg in segments
                ]

                cursor.executemany(
                    """
                    INSERT INTO audio_segments
                    (id, audio_id, sequence_order, start_sec, end_sec, duration_sec,
                     vad_confidence, text, language, speaker_label, speaker_id,
                     whisper_segment_id, avg_logprob, no_speech_prob, words_json,
                     speaker_embedding_json, acoustic_features_json, created_at,
                     speaker_confidence, attribution_decision,
                     attribution_evidence_json, provisional)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    records,
                )
                conn.commit()

            logger.info(f"Saved {len(segments)} audio segment(s) for asset {audio_id}.")
        except Exception as exc:
            logger.error(f"Failed to save audio segments for {audio_id}: {exc}")
            raise StorageError(f"Failed to save audio segments: {exc}") from exc

    def get_audio_segments(self, audio_id: str) -> List[AudioSegment]:
        """Retrieve all V3 AudioSegment objects for an audio asset, ordered by sequence."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM audio_segments WHERE audio_id = ? ORDER BY sequence_order ASC",
                    (audio_id,),
                )
                rows = cursor.fetchall()
                return [self._parse_audio_segment_row(row) for row in rows]
        except Exception as exc:
            logger.error(f"Failed to retrieve audio segments for {audio_id}: {exc}")
            raise StorageError(f"Failed to get audio segments: {exc}") from exc

    def _parse_audio_segment_row(self, row: sqlite3.Row) -> AudioSegment:
        words_raw = json.loads(row["words_json"]) if row["words_json"] else []
        words = [TranscriptWord(**w) for w in words_raw]

        embedding_raw = json.loads(row["speaker_embedding_json"]) if row["speaker_embedding_json"] else None
        acoustic_raw = json.loads(row["acoustic_features_json"]) if row["acoustic_features_json"] else None

        keys = row.keys() if hasattr(row, "keys") else []
        speaker_label = row["speaker_label"] if "speaker_label" in keys else None
        speaker_id = row["speaker_id"] if "speaker_id" in keys else None
        # V3.2 CASA fields (may be absent in older DB rows)
        speaker_confidence = row["speaker_confidence"] if "speaker_confidence" in keys else None
        attribution_decision = row["attribution_decision"] if "attribution_decision" in keys else None
        attribution_evidence_json = row["attribution_evidence_json"] if "attribution_evidence_json" in keys else None
        attribution_evidence = json.loads(attribution_evidence_json) if attribution_evidence_json else None
        provisional_raw = row["provisional"] if "provisional" in keys else None
        provisional = bool(provisional_raw) if provisional_raw is not None else None

        return AudioSegment(
            id=row["id"],
            audio_id=row["audio_id"],
            sequence_order=row["sequence_order"],
            start_sec=row["start_sec"],
            end_sec=row["end_sec"],
            duration_sec=row["duration_sec"],
            vad_confidence=row["vad_confidence"],
            text=row["text"],
            language=row["language"],
            speaker_label=speaker_label,
            speaker_id=speaker_id,
            whisper_segment_id=row["whisper_segment_id"],
            avg_logprob=row["avg_logprob"],
            no_speech_prob=row["no_speech_prob"],
            words=words,
            speaker_embedding=embedding_raw,
            acoustic_features=acoustic_raw,
            # V3.2 CASA
            speaker_confidence=speaker_confidence,
            attribution_decision=attribution_decision,
            attribution_evidence=attribution_evidence,
            provisional=provisional,
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.utcnow(),
        )

    def delete_audio_asset(self, audio_id: str) -> bool:
        """
        Delete an audio asset and cascade delete all associated processing jobs,
        transcripts, word alignments, chunks, indexing statuses, and audio segments.
        Also safely removes media files from disk if present.
        """
        try:
            asset = self.get_audio_asset(audio_id)
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM audio_segments WHERE audio_id = ?", (audio_id,))
                cursor.execute("DELETE FROM indexing_status WHERE audio_id = ?", (audio_id,))
                cursor.execute("DELETE FROM transcript_chunks WHERE audio_id = ?", (audio_id,))
                cursor.execute("DELETE FROM transcript_words WHERE audio_id = ?", (audio_id,))
                cursor.execute("DELETE FROM transcripts WHERE audio_id = ?", (audio_id,))
                cursor.execute("DELETE FROM processing_jobs WHERE audio_id = ?", (audio_id,))
                cursor.execute("DELETE FROM audio_assets WHERE id = ?", (audio_id,))
                conn.commit()

            # Clean up media files on disk
            if asset:
                raw_path = Path(asset.file_path)
                if raw_path.exists():
                    try:
                        raw_path.unlink()
                    except Exception as e:
                        logger.warning(f"Could not remove raw file {raw_path}: {e}")
                wav_path = raw_path.parent / f"{audio_id}.wav"
                if wav_path.exists():
                    try:
                        wav_path.unlink()
                    except Exception as e:
                        logger.warning(f"Could not remove wav file {wav_path}: {e}")

            logger.info(f"Successfully deleted audio asset {audio_id} from SQLite.")
            return True
        except Exception as exc:
            logger.error(f"Failed to delete audio asset {audio_id}: {exc}")
            raise StorageError(f"Failed to delete audio asset: {exc}") from exc

    def save_analysis_result(self, job_id: str, audio_id: str, result_json: str) -> None:
        """Persist a canonical AnalysisResult serialized JSON."""
        try:
            now_iso = datetime.utcnow().isoformat()
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO analysis_results
                    (job_id, audio_id, result_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (job_id, audio_id, result_json, now_iso),
                )
                conn.commit()
            logger.debug(f"Saved AnalysisResult for job {job_id}")
        except Exception as exc:
            logger.error(f"Failed to save analysis result for job {job_id}: {exc}")
            raise StorageError(f"Failed to save analysis result: {exc}") from exc

    def get_analysis_result(self, job_id: str) -> Optional[dict]:
        """Fetch a canonical AnalysisResult JSON by job_id."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT result_json FROM analysis_results WHERE job_id = ?",
                    (job_id,),
                )
                row = cursor.fetchone()
                if row and row["result_json"]:
                    return json.loads(row["result_json"])
                return None
        except Exception as exc:
            logger.error(f"Failed to fetch analysis result for job {job_id}: {exc}")
            raise StorageError(f"Failed to fetch analysis result: {exc}") from exc
