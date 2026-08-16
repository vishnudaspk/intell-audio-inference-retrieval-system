"""
Audio Worker — V3 Phase 1I
Pipeline orchestrator for V3 audio intelligence foundation.

Pipeline stages:
  1. Ingestion & Normalization   → 16kHz mono PCM WAV (AudioService)
  2. VAD                         → Speech intervals with confidence (VADService)
  3. ASR (Whisper)               → Transcript + per-segment text/words (TranscriptionService)
  4. Speaker Embeddings          → ECAPA-TDNN 192-dim vectors per segment (SpeakerEmbeddingService)
  5. Acoustic Features           → Pitch, energy, spectral per segment (AcousticFeatureService)
  6. Assembly                    → Unified AudioSegment objects
  7. Persistence                 → SQLite (audio_segments table) + legacy Transcript record
"""

import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from database.base import BaseRepository
from database.sqlite_db import SQLiteRepository
from schemas.enums import JobStatus
from schemas.models import AudioAsset, AudioSegment, ProcessingJob, TranscriptWord
from services.acoustic_service import AcousticFeatureService
from services.audio_service import AudioService
from services.speaker_embedding_service import SpeakerEmbeddingService
from services.transcription_service import TranscriptionService
from services.vad_service import VADService
from utils.exceptions import IntellAudioError
from utils.logger import logger


def _match_whisper_to_vad(
    vad_segments: list,  # [(start, end, conf), ...]
    whisper_segments: list,  # list of segment dicts from TranscriptionService raw
) -> dict:
    """
    For each VAD segment, find the best-matching Whisper segment(s) by temporal overlap.
    Returns a dict mapping VAD segment index → merged text and word list.
    """
    result = {}
    for vad_idx, (vad_start, vad_end, _) in enumerate(vad_segments):
        matched_text_parts = []
        matched_words = []
        matched_seg_id = None
        best_avg_logprob = None
        best_no_speech_prob = None

        for ws in whisper_segments:
            ws_start = ws.get("start", 0.0)
            ws_end = ws.get("end", 0.0)
            # Overlap check
            overlap = min(vad_end, ws_end) - max(vad_start, ws_start)
            if overlap > 0:
                matched_text_parts.append(ws.get("text", "").strip())
                matched_words.extend(ws.get("words", []))
                if matched_seg_id is None:
                    matched_seg_id = ws.get("id")
                    best_avg_logprob = ws.get("avg_logprob")
                    best_no_speech_prob = ws.get("no_speech_prob")

        result[vad_idx] = {
            "text": " ".join(matched_text_parts),
            "words": matched_words,
            "whisper_segment_id": matched_seg_id,
            "avg_logprob": best_avg_logprob,
            "no_speech_prob": best_no_speech_prob,
        }
    return result


class AudioWorker:
    """V3 Pipeline orchestrator for audio intelligence foundation processing."""

    def __init__(
        self,
        repository: Optional[BaseRepository] = None,
        audio_service: Optional[AudioService] = None,
        vad_service: Optional[VADService] = None,
        transcription_service: Optional[TranscriptionService] = None,
        speaker_embedding_service: Optional[SpeakerEmbeddingService] = None,
        acoustic_service: Optional[AcousticFeatureService] = None,
        extract_acoustics: bool = True,
        extract_embeddings: bool = True,
    ):
        self.repo = repository or SQLiteRepository()
        self.audio_service = audio_service or AudioService()
        self.vad_service = vad_service or VADService()
        self.transcription_service = transcription_service or TranscriptionService()
        self.speaker_embedding_service = speaker_embedding_service or SpeakerEmbeddingService()
        self.acoustic_service = acoustic_service or AcousticFeatureService()
        self.extract_acoustics = extract_acoustics
        self.extract_embeddings = extract_embeddings

    def process_asset(self, asset: AudioAsset) -> ProcessingJob:
        """
        Run the full V3 intelligence pipeline for an audio asset.

        Returns a completed or failed ProcessingJob with timing breakdown.
        """
        job = ProcessingJob(audio_id=asset.id, status=JobStatus.CREATED)
        self.repo.save_job(job)
        self.repo.save_audio_asset(asset)

        t_start = time.time()
        timings: dict = {}

        try:
            # ── Stage 1: Normalization ──────────────────────────────────────
            self._update_job(job, JobStatus.NORMALIZING)
            logger.info(f"[Job {job.id}] Stage=NORMALIZING asset={asset.id}")

            t0 = time.time()
            wav_path = self.audio_service.normalize_to_wav(asset)
            self.repo.save_audio_asset(asset)  # Persist duration update
            timings["normalization_sec"] = round(time.time() - t0, 3)

            # ── Stage 2: VAD ───────────────────────────────────────────────
            self._update_job(job, JobStatus.TRANSCRIBING)  # re-use TRANSCRIBING status
            logger.info(f"[Job {job.id}] Stage=VAD asset={asset.id}")

            t0 = time.time()
            raw_vad_segments = self.vad_service.detect_segments(wav_path)
            filtered_vad_segments = self.vad_service.filter_short_segments(raw_vad_segments)
            final_vad_segments = self.vad_service.merge_close_segments(filtered_vad_segments, max_gap_sec=0.3)
            timings["vad_sec"] = round(time.time() - t0, 3)

            num_filtered = len(raw_vad_segments) - len(filtered_vad_segments)
            num_merged = len(filtered_vad_segments) - len(final_vad_segments)
            logger.info(
                f"[Job {job.id}] VAD summary: raw={len(raw_vad_segments)} | "
                f"filtered_short(<0.25s)={num_filtered} | "
                f"merged_close(<=0.30s)={num_merged} | "
                f"final_segments={len(final_vad_segments)}"
            )
            vad_segments = final_vad_segments

            if not vad_segments:
                logger.warning(f"[Job {job.id}] No speech detected in asset {asset.id}. Completing with 0 segments.")
                return self._finalize_job(job, timings, t_start, segments=[])

            # ── Stage 3: ASR (Whisper) ────────────────────────────────────
            logger.info(f"[Job {job.id}] Stage=ASR asset={asset.id}")

            t0 = time.time()
            transcript = self.transcription_service.transcribe_audio(
                audio_id=asset.id,
                wav_path=wav_path,
            )
            timings["asr_sec"] = round(time.time() - t0, 3)

            # Persist transcript (backward-compatible with indexing pipeline)
            self.repo.save_transcript(transcript)
            # Persist word-level alignment
            if transcript.words:
                self.repo.save_alignment_words(asset.id, transcript.words)

            # Extract raw whisper segment dicts for temporal matching
            raw_whisper_segments = [
                {
                    "id": seg.sequence_order,
                    "start": seg.start or 0.0,
                    "end": seg.end or 0.0,
                    "text": seg.text,
                    "avg_logprob": None,
                    "no_speech_prob": None,
                    "words": [
                        {
                            "word": w.word,
                            "start": w.start or 0.0,
                            "end": w.end or 0.0,
                            "probability": w.confidence or 0.0,
                        }
                        for w in seg.words
                    ],
                }
                for seg in transcript.segments
            ]

            # Map Whisper output onto VAD intervals
            vad_to_whisper = _match_whisper_to_vad(vad_segments, raw_whisper_segments)

            # ── Stage 4: Diarization & Speaker Embeddings ─────────────────
            diarized_segments = []
            diarization_diagnostics = {}
            if self.extract_embeddings:
                logger.info(f"[Job {job.id}] Stage=SPEAKER_DIARIZATION asset={asset.id}")
                t0 = time.time()
                try:
                    words_for_diarization = [
                        {
                            "word": w.word,
                            "start_time": w.start or 0.0,
                            "end_time": w.end or 0.0,
                            "confidence": w.confidence,
                        }
                        for w in (transcript.words or [])
                    ]
                    speech_intervals = [(s, e) for s, e, _ in vad_segments]
                    diarized_segments, diarization_diagnostics = self.speaker_embedding_service.diarize_audio(
                        wav_path=wav_path,
                        speech_intervals=speech_intervals,
                        transcript_words=words_for_diarization if words_for_diarization else None,
                    )
                    timings["speaker_diarization_diagnostics"] = diarization_diagnostics
                except Exception as exc:
                    logger.warning(f"[Job {job.id}] Speaker diarization failed (non-fatal): {exc}")
                timings["speaker_embedding_sec"] = round(time.time() - t0, 3)

            # Determine segments to assemble: prefer diarized phrase units if available, else VAD intervals
            if diarized_segments:
                assembled_units = diarized_segments
            else:
                assembled_units = [
                    {
                        "start_sec": s,
                        "end_sec": e,
                        "duration_sec": e - s,
                        "vad_confidence": conf,
                        "text": vad_to_whisper.get(i, {}).get("text", ""),
                        "words": vad_to_whisper.get(i, {}).get("words", []),
                        "speaker_label": None,
                    }
                    for i, (s, e, conf) in enumerate(vad_segments)
                ]

            # ── Stage 5: Acoustic Features & Embeddings for Assembled Units ──
            acoustic_features_list = [None] * len(assembled_units)
            embeddings_list = [None] * len(assembled_units)
            assembled_intervals = [(u["start_sec"], u["end_sec"]) for u in assembled_units]

            if self.extract_acoustics:
                logger.info(f"[Job {job.id}] Stage=ACOUSTIC_FEATURES asset={asset.id}")
                t0 = time.time()
                try:
                    acoustic_features_list = self.acoustic_service.extract_batch(
                        wav_path,
                        assembled_intervals,
                    )
                except Exception as exc:
                    logger.warning(f"[Job {job.id}] Acoustic extraction failed (non-fatal): {exc}")
                timings["acoustic_sec"] = round(time.time() - t0, 3)

            if self.extract_embeddings:
                try:
                    embeddings_list = self.speaker_embedding_service.embed_segments(
                        wav_path,
                        assembled_intervals,
                    )
                except Exception as exc:
                    logger.warning(f"[Job {job.id}] Segment embedding extraction failed (non-fatal): {exc}")

            # ── Stage 6: Assemble AudioSegment objects ────────────────────
            logger.info(f"[Job {job.id}] Stage=ASSEMBLY asset={asset.id}")
            audio_segments: List[AudioSegment] = []

            for idx, unit in enumerate(assembled_units):
                unit_words = unit.get("words", [])
                words: List[TranscriptWord] = [
                    TranscriptWord(
                        word=w.get("word", ""),
                        start=w.get("start", w.get("start_time")),
                        end=w.get("end", w.get("end_time")),
                        confidence=w.get("confidence", w.get("probability")),
                    )
                    if isinstance(w, dict)
                    else w
                    for w in unit_words
                ]

                emb = embeddings_list[idx] if idx < len(embeddings_list) else None
                embedding_list = emb.tolist() if emb is not None and np.linalg.norm(emb) > 1e-6 else None

                acoustic = acoustic_features_list[idx] if idx < len(acoustic_features_list) else None
                acoustic_dict = acoustic.to_dict() if acoustic is not None else None

                vad_conf = unit.get("vad_confidence", 0.90)

                seg = AudioSegment(
                    audio_id=asset.id,
                    sequence_order=idx,
                    start_sec=round(unit["start_sec"], 4),
                    end_sec=round(unit["end_sec"], 4),
                    duration_sec=round(unit["end_sec"] - unit["start_sec"], 4),
                    vad_confidence=round(vad_conf, 4),
                    text=unit.get("text", ""),
                    language=transcript.language.value,
                    speaker_label=unit.get("speaker_label"),
                    whisper_segment_id=unit.get("whisper_segment_id"),
                    avg_logprob=unit.get("avg_logprob"),
                    no_speech_prob=unit.get("no_speech_prob"),
                    words=words,
                    speaker_embedding=embedding_list,
                    acoustic_features=acoustic_dict,
                )
                audio_segments.append(seg)

            # ── Stage 7: Persistence ──────────────────────────────────────
            self._update_job(job, JobStatus.PERSISTING)
            logger.info(f"[Job {job.id}] Stage=PERSISTING {len(audio_segments)} segment(s)")

            self.repo.save_audio_segments(asset.id, audio_segments)

            return self._finalize_job(job, timings, t_start, audio_segments)

        except Exception as exc:
            logger.error(f"[Job {job.id}] asset={asset.id} FAILED: {exc}")
            job.status = JobStatus.FAILED
            job.error_message = str(exc)
            job.updated_at = datetime.utcnow()
            self.repo.save_job(job)
            raise IntellAudioError(f"V3 pipeline job failed: {exc}") from exc

    def _update_job(self, job: ProcessingJob, status: JobStatus) -> None:
        job.status = status
        job.updated_at = datetime.utcnow()
        self.repo.save_job(job)

    def _finalize_job(
        self,
        job: ProcessingJob,
        timings: dict,
        t_start: float,
        segments: List[AudioSegment],
    ) -> ProcessingJob:
        timings["total_sec"] = round(time.time() - t_start, 3)
        timings["segments_produced"] = len(segments)

        job.status = JobStatus.COMPLETED
        job.timings = timings
        job.updated_at = datetime.utcnow()
        self.repo.save_job(job)

        logger.info(
            f"[Job {job.id}] COMPLETED - "
            f"{len(segments)} segments in {timings['total_sec']}s | "
            f"timings={timings}"
        )
        return job
