"""
Pipeline Orchestrator for the Intell Audio Intelligence Platform.
Coordinates ingestion, analysis, intelligence processing, event emission, and AnalysisResult assembly.
"""

import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from config.settings import settings
from core.event_bus import EventBus, event_bus
from database.base import BaseRepository
from database.sqlite_db import SQLiteRepository
from schemas.analysis import (
    AcousticFeatureSet,
    AnalysisMetadata,
    AnalysisResult,
    AudioInfo,
    AudioQuality,
    BandEnergies,
    ClusterInfo,
    ConversationAnalytics,
    ConversationTurn,
    DiarizationResult,
    DiarizedSegment,
    EmbeddingPoint,
    EmbeddingViz,
    HardwareInfo,
    ProcessingEvent,
    ProcessingInfo,
    ProcessingStage,
    ShortResponse,
    SilenceGap,
    SpeakerProfile,
    SpeakerStatistics,
    SpeakerTransition,
    TemporalModel,
    TranscriptionResult,
    TranscriptSegmentResult,
    TranscriptWord,
    VADResult,
    VADSegment,
)
from schemas.enums import JobStatus, StageStatus
from schemas.models import AudioAsset, ProcessingJob
from services.acoustic_service import AcousticFeatureService
from services.audio_service import AudioService
from services.casa_config import CASAConfig
from services.casa_engine import CASAEngine
from services.speaker_embedding_service import SpeakerEmbeddingService
from services.transcription_service import TranscriptionService
from services.vad_service import VADService
from utils.exceptions import IntellAudioError
from utils.logger import logger


# Stage weights for overall progress calculation (sum = 100)
STAGE_WEIGHTS = {
    "normalization": 3,
    "audio_quality": 2,
    "vad": 5,
    "whisper": 25,
    "speaker_embedding": 30,
    "ahc": 5,
    "hmm": 8,
    "casa": 5,
    "acoustic": 8,
    "conversation": 4,
    "assembly": 3,
    "persistence": 2,
}
TOTAL_WEIGHT = sum(STAGE_WEIGHTS.values())


class PipelineOrchestrator:
    """
    Asynchronous, event-emitting pipeline orchestrator.
    Executes all audio intelligence subsystems and produces a canonical AnalysisResult.
    """

    def __init__(
        self,
        repository: Optional[BaseRepository] = None,
        bus: Optional[EventBus] = None,
        audio_service: Optional[AudioService] = None,
        vad_service: Optional[VADService] = None,
        transcription_service: Optional[TranscriptionService] = None,
        speaker_embedding_service: Optional[SpeakerEmbeddingService] = None,
        acoustic_service: Optional[AcousticFeatureService] = None,
    ):
        self.repo = repository or SQLiteRepository()
        self.bus = bus or event_bus
        self.audio_service = audio_service or AudioService()
        self.vad_service = vad_service or VADService()
        self.transcription_service = transcription_service or TranscriptionService()
        self.speaker_embedding_service = speaker_embedding_service or SpeakerEmbeddingService()
        self.acoustic_service = acoustic_service or AcousticFeatureService()

    def _calc_overall_progress(self, completed_stages: List[str], current_stage: str, current_pct: int) -> int:
        completed_score = sum(STAGE_WEIGHTS.get(s, 0) for s in completed_stages)
        current_score = STAGE_WEIGHTS.get(current_stage, 0) * (current_pct / 100.0)
        return min(100, int((completed_score + current_score) / TOTAL_WEIGHT * 100))

    def _emit_event(
        self,
        job_id: str,
        stage: str,
        status: StageStatus,
        progress: int,
        completed_stages: List[str],
        processed: Optional[int] = None,
        total: Optional[int] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        t_start: float = 0.0,
    ) -> None:
        overall = self._calc_overall_progress(completed_stages, stage, progress)
        elapsed_ms = int((time.time() - t_start) * 1000) if t_start > 0 else 0
        event = ProcessingEvent(
            job_id=job_id,
            stage=stage,
            status=status,
            progress=progress,
            overall_progress=overall,
            processed=processed,
            total=total,
            elapsed_ms=elapsed_ms,
            message=message,
            error=error,
            timestamp=datetime.utcnow(),
        )
        self.bus.emit_sync(job_id, event.model_dump())

    def run_pipeline(self, asset: AudioAsset, job: Optional[ProcessingJob] = None) -> AnalysisResult:
        """
        Execute full intelligence pipeline synchronously in worker thread, emitting events.
        """
        if not job:
            job = ProcessingJob(audio_id=asset.id, status=JobStatus.RUNNING)
        else:
            job.status = JobStatus.RUNNING
        self.repo.save_job(job)
        self.repo.save_audio_asset(asset)

        t_job_start = time.time()
        completed_stages: List[str] = []
        stages_info: List[ProcessingStage] = []

        # Initialize hardware info
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else None
        hardware = HardwareInfo(
            device="cuda" if cuda_avail else "cpu",
            cuda_available=cuda_avail,
            gpu_name=gpu_name,
        )

        try:
            # ── 1. Normalization ─────────────────────────────────────────────
            stage_name = "normalization"
            t0 = time.time()
            self._emit_event(job.id, stage_name, StageStatus.RUNNING, 0, completed_stages, t_start=t_job_start, message="Normalizing audio to 16kHz mono WAV")
            wav_path = self.audio_service.normalize_to_wav(asset)
            self.repo.save_audio_asset(asset)
            dur_norm = round(time.time() - t0, 3)
            completed_stages.append(stage_name)
            self._emit_event(job.id, stage_name, StageStatus.COMPLETED, 100, completed_stages, t_start=t_job_start)
            stages_info.append(ProcessingStage(name=stage_name, status=StageStatus.COMPLETED, duration_sec=dur_norm))

            # ── 2. Audio Quality Analysis ────────────────────────────────────
            stage_name = "audio_quality"
            t0 = time.time()
            self._emit_event(job.id, stage_name, StageStatus.RUNNING, 0, completed_stages, t_start=t_job_start, message="Analyzing audio signal characteristics")
            
            # Simple direct calculation or fallback
            audio_quality = AudioQuality()
            try:
                import soundfile as sf
                data, sr = sf.read(str(wav_path))
                rms = float(np.sqrt(np.mean(data**2)))
                clipping = bool(np.any(np.abs(data) >= 0.999))
                # Dynamic range
                nonzero = np.abs(data[np.abs(data) > 1e-5])
                dyn_range = float(20 * np.log10(np.max(nonzero) / (np.min(nonzero) + 1e-8))) if len(nonzero) > 0 else 0.0
                audio_quality = AudioQuality(
                    rms_energy=round(rms, 4),
                    clipping_detected=clipping,
                    dynamic_range_db=round(dyn_range, 2),
                    warnings=["Audio clipping detected"] if clipping else [],
                )
            except Exception as e:
                logger.warning(f"Audio quality analysis warning: {e}")

            dur_aq = round(time.time() - t0, 3)
            completed_stages.append(stage_name)
            self._emit_event(job.id, stage_name, StageStatus.COMPLETED, 100, completed_stages, t_start=t_job_start)
            stages_info.append(ProcessingStage(name=stage_name, status=StageStatus.COMPLETED, duration_sec=dur_aq))

            # ── 3. Voice Activity Detection (VAD) ────────────────────────────
            stage_name = "vad"
            t0 = time.time()
            self._emit_event(job.id, stage_name, StageStatus.RUNNING, 0, completed_stages, t_start=t_job_start, message="Running Silero VAD")
            raw_vad = self.vad_service.detect_segments(wav_path)
            filtered_vad = self.vad_service.filter_short_segments(raw_vad)
            final_vad = self.vad_service.merge_close_segments(filtered_vad, max_gap_sec=0.3)
            dur_vad = round(time.time() - t0, 3)
            completed_stages.append(stage_name)
            self._emit_event(job.id, stage_name, StageStatus.COMPLETED, 100, completed_stages, t_start=t_job_start)
            stages_info.append(ProcessingStage(name=stage_name, status=StageStatus.COMPLETED, duration_sec=dur_vad, model_info={"engine": "silero-vad"}))

            total_speech_sec = sum(e - s for s, e, _ in final_vad)
            total_dur = asset.duration or (final_vad[-1][1] if final_vad else 1.0)
            vad_res = VADResult(
                engine="silero-vad",
                threshold=0.5,
                segments=[VADSegment(start_sec=s, end_sec=e, duration_sec=round(e-s, 3), confidence=c) for s, e, c in final_vad],
                speech_duration_sec=round(total_speech_sec, 3),
                silence_duration_sec=round(max(0.0, total_dur - total_speech_sec), 3),
                speech_ratio=round(total_speech_sec / max(1e-5, total_dur), 3),
                total_segments=len(final_vad),
            )
            audio_quality.silence_ratio = round(max(0.0, 1.0 - vad_res.speech_ratio), 3)

            # ── 4. Transcription (Whisper) ───────────────────────────────────
            stage_name = "whisper"
            t0 = time.time()
            self._emit_event(job.id, stage_name, StageStatus.RUNNING, 0, completed_stages, t_start=t_job_start, message="Transcribing speech with faster-whisper")
            transcript = self.transcription_service.transcribe_audio(audio_id=asset.id, wav_path=wav_path)
            dur_asr = round(time.time() - t0, 3)
            self.repo.save_transcript(transcript)
            if transcript.words:
                self.repo.save_alignment_words(asset.id, transcript.words)
            completed_stages.append(stage_name)
            self._emit_event(job.id, stage_name, StageStatus.COMPLETED, 100, completed_stages, t_start=t_job_start)
            stages_info.append(ProcessingStage(name=stage_name, status=StageStatus.COMPLETED, duration_sec=dur_asr, model_info={"model": settings.WHISPER_MODEL}))

            tx_result = TranscriptionResult(
                engine="faster-whisper",
                model=settings.WHISPER_MODEL,
                language=transcript.language.value if hasattr(transcript.language, "value") else str(transcript.language),
                full_text=transcript.text,
                duration_sec=transcript.duration,
                processing_sec=dur_asr,
                segments=[
                    TranscriptSegmentResult(
                        id=seg.id,
                        sequence_order=seg.sequence_order,
                        start_sec=seg.start or 0.0,
                        end_sec=seg.end or 0.0,
                        duration_sec=round((seg.end or 0.0) - (seg.start or 0.0), 3),
                        text=seg.text,
                        words=[TranscriptWord(id=w.id, word=w.word, start=w.start, end=w.end, confidence=w.confidence) for w in seg.words],
                    )
                    for seg in transcript.segments
                ],
                word_timestamps=[TranscriptWord(id=w.id, word=w.word, start=w.start, end=w.end, confidence=w.confidence) for w in transcript.words],
            )

            # ── 5. Speaker Diarization & ECAPA Embeddings ────────────────────
            stage_name = "speaker_embedding"
            t0 = time.time()
            self._emit_event(job.id, stage_name, StageStatus.RUNNING, 0, completed_stages, t_start=t_job_start, message="Extracting ECAPA-TDNN speaker embeddings")
            
            words_for_diarization = [
                {"word": w.word, "start_time": w.start or 0.0, "end_time": w.end or 0.0, "confidence": w.confidence}
                for w in (transcript.words or [])
            ]
            speech_intervals = [(s, e) for s, e, _ in final_vad]
            
            diarized_segments, diarization_diagnostics = self.speaker_embedding_service.diarize_audio(
                wav_path=wav_path,
                speech_intervals=speech_intervals,
                transcript_words=words_for_diarization if words_for_diarization else None,
            )
            dur_emb = round(time.time() - t0, 3)
            completed_stages.append(stage_name)
            self._emit_event(job.id, stage_name, StageStatus.COMPLETED, 100, completed_stages, t_start=t_job_start)
            stages_info.append(ProcessingStage(name=stage_name, status=StageStatus.COMPLETED, duration_sec=dur_emb, model_info={"model": "speechbrain/spkrec-ecapa-voxceleb"}))

            # ── 6. CASA Attribution Layer ───────────────────────────────────
            stage_name = "casa"
            t0 = time.time()
            if diarized_segments and settings.CASA_ENABLED:
                self._emit_event(job.id, stage_name, StageStatus.RUNNING, 0, completed_stages, t_start=t_job_start, message="Applying CASA multimodal attribution fusion")
                casa_engine = CASAEngine(config=CASAConfig())
                phrase_embeddings = diarization_diagnostics.get("phrase_embeddings") or []
                speaker_centroids = diarization_diagnostics.get("speaker_centroids") or {}
                casa_results = casa_engine.apply(
                    diarized_segments=diarized_segments,
                    phrase_embeddings=phrase_embeddings if phrase_embeddings else None,
                    speaker_centroids=speaker_centroids if speaker_centroids else None,
                )
                for res in casa_results:
                    idx = res.phrase_index
                    if idx < len(diarized_segments):
                        diarized_segments[idx]["speaker_label"] = res.proposed_speaker
                        diarized_segments[idx]["speaker_confidence"] = res.confidence
                        diarized_segments[idx]["attribution_decision"] = res.decision
                        diarized_segments[idx]["provisional"] = res.provisional
            
            dur_casa = round(time.time() - t0, 3)
            completed_stages.append(stage_name)
            self._emit_event(job.id, stage_name, StageStatus.COMPLETED, 100, completed_stages, t_start=t_job_start)
            stages_info.append(ProcessingStage(name=stage_name, status=StageStatus.COMPLETED, duration_sec=dur_casa))

            # ── 7. Acoustic Feature Extraction ──────────────────────────────
            stage_name = "acoustic"
            t0 = time.time()
            self._emit_event(job.id, stage_name, StageStatus.RUNNING, 0, completed_stages, t_start=t_job_start, message="Computing acoustic features and spectral stats")
            
            # Extract features per diarized segment
            assembled_diar_segs: List[DiarizedSegment] = []
            for idx, dseg in enumerate(diarized_segments):
                st = dseg.get("start_sec", 0.0)
                et = dseg.get("end_sec", 0.0)
                ac_feat = self.acoustic_service.extract_features(wav_path, st, et)
                feat_set = AcousticFeatureSet(
                    f0_mean=ac_feat.f0_mean,
                    f0_median=ac_feat.f0_median,
                    f0_std=ac_feat.f0_std,
                    f0_range=ac_feat.f0_range,
                    f0_voiced_fraction=ac_feat.f0_voiced_fraction,
                    rms_mean=ac_feat.rms_mean,
                    rms_std=ac_feat.rms_std,
                    spectral_centroid_mean=ac_feat.spectral_centroid_mean,
                    spectral_bandwidth_mean=ac_feat.spectral_bandwidth_mean,
                    spectral_rolloff_mean=ac_feat.spectral_rolloff_mean,
                    zcr_mean=ac_feat.zero_crossing_rate_mean,
                    mfcc_means=ac_feat.mfcc_means or [],
                )
                assembled_diar_segs.append(
                    DiarizedSegment(
                        id=f"seg_{job.id}_{idx}",
                        sequence_order=idx,
                        start_sec=st,
                        end_sec=et,
                        duration_sec=round(et - st, 3),
                        text=dseg.get("text", ""),
                        speaker_label=dseg.get("speaker_label", "Speaker 1"),
                        speaker_id=dseg.get("speaker_id"),
                        confidence=dseg.get("speaker_confidence", 1.0),
                        attribution_decision=dseg.get("attribution_decision"),
                        provisional=dseg.get("provisional", False),
                        acoustic_features=feat_set,
                    )
                )

            dur_ac = round(time.time() - t0, 3)
            completed_stages.append(stage_name)
            self._emit_event(job.id, stage_name, StageStatus.COMPLETED, 100, completed_stages, t_start=t_job_start)
            stages_info.append(ProcessingStage(name=stage_name, status=StageStatus.COMPLETED, duration_sec=dur_ac))

            # ── 8. Speaker Analytics & Conversation Analysis ────────────────
            stage_name = "conversation"
            t0 = time.time()
            self._emit_event(job.id, stage_name, StageStatus.RUNNING, 0, completed_stages, t_start=t_job_start, message="Computing conversational turn-taking analytics")

            # Group by speaker
            speaker_labels = sorted(list(set(s.speaker_label for s in assembled_diar_segs)))
            speaker_profiles: List[SpeakerProfile] = []
            palette = ["#4A90E2", "#50E3C2", "#F5A623", "#E35050", "#BD10E0", "#7ED321"]

            for spk_idx, spk in enumerate(speaker_labels):
                spk_segs = [s for s in assembled_diar_segs if s.speaker_label == spk]
                spk_dur = sum(s.duration_sec for s in spk_segs)
                mean_conf = float(np.mean([s.confidence for s in spk_segs])) if spk_segs else 1.0
                
                # Turn statistics
                turns_count = len(spk_segs)
                avg_turn = round(spk_dur / max(1, turns_count), 2)
                longest = round(max((s.duration_sec for s in spk_segs), default=0.0), 2)
                shortest = round(min((s.duration_sec for s in spk_segs), default=0.0), 2)

                stats = SpeakerStatistics(
                    total_speaking_sec=round(spk_dur, 2),
                    speaking_percentage=round(spk_dur / max(1e-5, total_dur) * 100, 1),
                    num_turns=turns_count,
                    avg_turn_sec=avg_turn,
                    longest_turn_sec=longest,
                    shortest_turn_sec=shortest,
                )
                speaker_profiles.append(
                    SpeakerProfile(
                        speaker_id=f"speaker_{spk_idx+1}",
                        speaker_label=spk,
                        color=palette[spk_idx % len(palette)],
                        statistics=stats,
                        confidence=round(mean_conf, 3),
                        segment_count=len(spk_segs),
                    )
                )

            # Conversation turns
            turns: List[ConversationTurn] = []
            for t_idx, seg in enumerate(assembled_diar_segs):
                words_cnt = len(seg.text.split())
                turns.append(
                    ConversationTurn(
                        turn_index=t_idx,
                        speaker_label=seg.speaker_label,
                        start_sec=seg.start_sec,
                        end_sec=seg.end_sec,
                        duration_sec=seg.duration_sec,
                        text=seg.text,
                        word_count=words_cnt,
                        is_short_response=(words_cnt > 0 and words_cnt < 5),
                    )
                )

            # Transitions
            transitions: List[SpeakerTransition] = []
            for i in range(len(assembled_diar_segs) - 1):
                cur = assembled_diar_segs[i]
                nxt = assembled_diar_segs[i + 1]
                if cur.speaker_label != nxt.speaker_label:
                    gap = round(nxt.start_sec - cur.end_sec, 3)
                    transitions.append(
                        SpeakerTransition(
                            from_speaker=cur.speaker_label,
                            to_speaker=nxt.speaker_label,
                            gap_sec=gap,
                            at_sec=cur.end_sec,
                        )
                    )

            # Dominant speaker
            dominant = max(speaker_profiles, key=lambda p: p.statistics.total_speaking_sec).speaker_label if speaker_profiles else "Speaker 1"
            conv_balance = {p.speaker_label: p.statistics.speaking_percentage for p in speaker_profiles}

            conv_analytics = ConversationAnalytics(
                total_duration_sec=round(total_dur, 2),
                num_turns=len(turns),
                num_speakers=len(speaker_labels),
                turns=turns,
                transitions=transitions,
                dominant_speaker=dominant,
                conversation_balance=conv_balance,
            )

            dur_conv = round(time.time() - t0, 3)
            completed_stages.append(stage_name)
            self._emit_event(job.id, stage_name, StageStatus.COMPLETED, 100, completed_stages, t_start=t_job_start)
            stages_info.append(ProcessingStage(name=stage_name, status=StageStatus.COMPLETED, duration_sec=dur_conv))

            # ── 9. Assembly & Final Persistence ──────────────────────────────
            stage_name = "assembly"
            t0 = time.time()
            self._emit_event(job.id, stage_name, StageStatus.RUNNING, 0, completed_stages, t_start=t_job_start, message="Assembling final AnalysisResult")

            # Diarization summary result
            diar_result = DiarizationResult(
                num_speakers=len(speaker_labels),
                method="AHC+eigengap+CASA",
                parameters={"min_ecapa_dur": 1.2, "casa_enabled": settings.CASA_ENABLED},
                segments=assembled_diar_segs,
                cluster_info=ClusterInfo(
                    num_clusters=len(speaker_labels),
                    cluster_sizes={p.speaker_label: p.segment_count for p in speaker_profiles},
                    mean_cosine_similarity=round(float(diarization_diagnostics.get("mean_cosine_sim", 1.0)), 3),
                ),
            )

            # Total processing stats
            total_job_sec = round(time.time() - t_job_start, 3)
            rtf = round(total_job_sec / max(1e-5, total_dur), 3)

            proc_info = ProcessingInfo(
                stages=stages_info,
                total_duration_sec=total_job_sec,
                audio_duration_sec=round(total_dur, 2),
                realtime_factor=rtf,
                hardware=hardware,
            )

            result = AnalysisResult(
                metadata=AnalysisMetadata(job_id=job.id, audio_id=asset.id),
                audio=AudioInfo(
                    filename=asset.filename,
                    format=asset.format or "wav",
                    duration_sec=round(total_dur, 2),
                    source_type=asset.source_type,
                ),
                audio_quality=audio_quality,
                vad=vad_res,
                transcription=tx_result,
                diarization=diar_result,
                speakers=speaker_profiles,
                conversation=conv_analytics,
                processing=proc_info,
            )

            # Persist canonical result
            self.repo.save_analysis_result(job.id, asset.id, result.model_dump_json())

            # Complete job in SQLite
            job.status = JobStatus.COMPLETED
            job.timings = {s.name: s.duration_sec for s in stages_info if s.duration_sec}
            self.repo.save_job(job)

            completed_stages.append(stage_name)
            self._emit_event(
                job.id,
                stage_name,
                StageStatus.COMPLETED,
                100,
                completed_stages,
                t_start=t_job_start,
                message=f"Pipeline complete in {total_job_sec}s (RTF: {rtf}x)",
            )

            return result

        except Exception as exc:
            logger.error(f"[Job {job.id}] Pipeline execution failed: {exc}", exc_info=True)
            job.status = JobStatus.FAILED
            job.error_message = str(exc)
            self.repo.save_job(job)
            self._emit_event(
                job.id,
                "pipeline",
                StageStatus.FAILED,
                0,
                completed_stages,
                t_start=t_job_start,
                error=str(exc),
                message=f"Pipeline failed: {exc}",
            )
            raise IntellAudioError(f"Audio processing pipeline failed: {exc}") from exc
