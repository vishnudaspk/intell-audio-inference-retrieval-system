"""
READ-ONLY CASA Evaluation Script on Podcast 1.mp3
=================================================
Performs a pure in-memory, read-only inference pass of Podcast 1.mp3 using the
existing V3.1 pipeline (Audio Normalization -> VAD -> Whisper ASR -> ECAPA Diarization)
and V3.2 CASA (Conversation-Aware Speaker Attribution).

Guarantees:
- Does NOT write or alter SQLite database records.
- Does NOT modify the source audio file.
- Saves evaluation diagnostics separately to evaluation/ as JSON and CSV.
- Extracts all available acoustic, temporal, and linguistic evidence from CASA.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from schemas.enums import SourceType
from schemas.models import AudioAsset
from services.audio_service import AudioService
from services.casa_config import CASAConfig
from services.casa_engine import CASAEngine
from services.speaker_embedding_service import SpeakerEmbeddingService
from services.transcription_service import TranscriptionService
from services.vad_service import VADService
from utils.logger import logger


def run_podcast1_evaluation():
    input_audio_path = PROJECT_ROOT / "file" / "Podcast 1.mp3"
    if not input_audio_path.exists():
        print(f"[ERROR] Target audio file not found at: {input_audio_path}")
        sys.exit(1)

    print("\n" + "=" * 90)
    print("  READ-ONLY CASA Evaluation on: Podcast 1.mp3")
    print("=" * 90)
    print(f"Source file: {input_audio_path} ({input_audio_path.stat().st_size / (1024*1024):.2f} MB)")

    # ── Step 1: Media Normalization (Temporary WAV) ───────────────────────────
    print("\n[1/5] Normalizing audio to 16kHz mono PCM WAV (temporary evaluation copy)...")
    audio_service = AudioService()
    asset = AudioAsset(
        id="eval_podcast1_readonly",
        filename="Podcast 1.mp3",
        file_path=str(input_audio_path),
        format="mp3",
        source_type=SourceType.UPLOAD,
    )

    t0 = time.time()
    wav_path = audio_service.normalize_to_wav(asset)
    norm_time = time.time() - t0
    print(f"      Normalized in {norm_time:.2f}s -> {wav_path} (Duration: {asset.duration:.2f}s)")

    # ── Step 2: VAD (Voice Activity Detection) ────────────────────────────────
    print("\n[2/5] Running Silero VAD speech segmentation...")
    vad_service = VADService()
    t0 = time.time()
    vad_segments = vad_service.detect_segments(wav_path)
    vad_time = time.time() - t0
    print(f"      Detected {len(vad_segments)} speech intervals in {vad_time:.2f}s")

    # ── Step 3: ASR (Whisper Transcription) ───────────────────────────────────
    print("\n[3/5] Running Whisper ASR for word-level timestamps...")
    transcription_service = TranscriptionService()
    t0 = time.time()
    transcript = transcription_service.transcribe_audio(
        audio_id=asset.id,
        wav_path=wav_path,
        word_timestamps=True,
    )
    asr_time = time.time() - t0
    total_words = len(transcript.words) if transcript.words else 0
    print(f"      Transcribed {total_words} words across {len(transcript.segments)} segments in {asr_time:.2f}s")
    print(f"      Language: {transcript.language}")

    # ── Step 4: ECAPA Diarization (V3.1 Baseline) ─────────────────────────────
    print("\n[4/5] Running SpeechBrain ECAPA-TDNN Speaker Diarization (V3.1 Baseline)...")
    speaker_service = SpeakerEmbeddingService()
    t0 = time.time()
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

    diarized_segments, diarization_diagnostics = speaker_service.diarize_audio(
        wav_path=wav_path,
        speech_intervals=speech_intervals,
        transcript_words=words_for_diarization if words_for_diarization else None,
    )
    diar_time = time.time() - t0

    phrase_embeddings = diarization_diagnostics.get("phrase_embeddings") or []
    speaker_centroids = diarization_diagnostics.get("speaker_centroids") or {}
    estimated_k = diarization_diagnostics.get("estimated_speakers", len(set(s.get("speaker_label") for s in diarized_segments)))
    print(f"      Diarized {len(diarized_segments)} dialogue phrases in {diar_time:.2f}s")
    print(f"      Estimated distinct speakers: {estimated_k}")
    print(f"      Cluster breakdown: {diarization_diagnostics.get('cluster_sizes')}")

    # ── Step 5: CASA Reasoning Engine (V3.2) ──────────────────────────────────
    print("\n[5/5] Running CASA Dialogue Consistency & Speaker Attribution Engine (V3.2)...")
    casa_config = CASAConfig()
    casa_engine = CASAEngine(config=casa_config)
    t0 = time.time()
    casa_results = casa_engine.apply(
        diarized_segments=diarized_segments,
        phrase_embeddings=phrase_embeddings if phrase_embeddings else None,
        speaker_centroids=speaker_centroids if speaker_centroids else None,
    )
    casa_time = time.time() - t0
    print(f"      Completed CASA evaluation in {casa_time:.3f}s")

    # ── Extract Script Semantics & Detailed Attribution Records ──────────────
    records: List[Dict[str, Any]] = []

    confirm_count = 0
    correct_count = 0
    uncertain_count = 0
    provisional_count = 0
    changed_count = 0
    confidence_values: List[float] = []

    for r in casa_results:
        idx = r.phrase_index
        seg = diarized_segments[idx] if idx < len(diarized_segments) else {}

        start_sec = round(seg.get("start_sec", 0.0), 3)
        end_sec = round(seg.get("end_sec", 0.0), 3)
        duration_sec = round(seg.get("duration_sec", end_sec - start_sec), 3)
        text = seg.get("text", "")

        original_spk = r.original_speaker
        final_spk = r.proposed_speaker
        is_changed = (original_spk != final_spk)

        if r.decision == "CONFIRM":
            confirm_count += 1
        elif r.decision == "CORRECT":
            correct_count += 1
        elif r.decision == "UNCERTAIN":
            uncertain_count += 1

        if r.provisional:
            provisional_count += 1

        if is_changed:
            changed_count += 1

        confidence_values.append(r.confidence)

        # Categorize evidence items
        acoustic_ev = [ev for ev in r.evidence if "acoustic" in ev.lower() or "continuity" in ev.lower() or "centroid" in ev.lower()]
        temporal_ev = [ev for ev in ev_list if "pause" in ev.lower() or "turn" in ev.lower() or "duration" in ev.lower()] if 'ev_list' in locals() else [ev for ev in r.evidence if "pause" in ev.lower() or "turn" in ev.lower() or "duration" in ev.lower()]
        linguistic_ev = [ev for ev in r.evidence if "dialogue pattern" in ev.lower() or "phrase type" in ev.lower() or "filler" in ev.lower()]

        record = {
            "phrase_index": idx,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": duration_sec,
            "transcript_text": text,
            "original_diarization_speaker": original_spk,
            "casa_assigned_speaker": final_spk,
            "attribution_changed": is_changed,
            "decision_type": r.decision,
            "attribution_confidence": round(r.confidence, 4),
            "provisional_status": r.provisional,
            "acoustic_score": round(r.acoustic_score, 4),
            "temporal_score": round(r.temporal_score, 4),
            "linguistic_score": round(r.linguistic_score, 4),
            "alternate_speaker": r.alternate_speaker,
            "alternate_confidence": round(r.alternate_confidence, 4) if r.alternate_confidence else None,
            "acoustic_evidence": acoustic_ev,
            "temporal_evidence": temporal_ev,
            "linguistic_evidence": linguistic_ev,
            "all_evidence": r.evidence,
        }
        records.append(record)

    # ── Step 6: Save Output as JSON and CSV ───────────────────────────────────
    eval_dir = PROJECT_ROOT / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    json_out_path = eval_dir / "podcast1_script_semantics.json"
    csv_out_path = eval_dir / "podcast1_script_semantics.csv"

    output_payload = {
        "metadata": {
            "audio_file": "Podcast 1.mp3",
            "audio_duration_sec": asset.duration,
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline_version": "V3.2 CASA",
            "total_dialogue_phrases": len(records),
            "estimated_speakers": estimated_k,
            "decision_summary": {
                "CONFIRM": confirm_count,
                "CORRECT": correct_count,
                "UNCERTAIN": uncertain_count,
                "PROVISIONAL": provisional_count,
                "ATTRIBUTION_CHANGED": changed_count,
            },
            "confidence_statistics": {
                "mean": round(float(np.mean(confidence_values)), 4) if confidence_values else 0.0,
                "std": round(float(np.std(confidence_values)), 4) if confidence_values else 0.0,
                "min": round(float(np.min(confidence_values)), 4) if confidence_values else 0.0,
                "max": round(float(np.max(confidence_values)), 4) if confidence_values else 0.0,
                "median": round(float(np.median(confidence_values)), 4) if confidence_values else 0.0,
            },
            "timings_sec": {
                "normalization": round(norm_time, 3),
                "vad": round(vad_time, 3),
                "asr": round(asr_time, 3),
                "diarization": round(diar_time, 3),
                "casa": round(casa_time, 3),
                "total": round(norm_time + vad_time + asr_time + diar_time + casa_time, 3),
            },
        },
        "phrases": records,
    }

    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    csv_fieldnames = [
        "phrase_index",
        "start_sec",
        "end_sec",
        "duration_sec",
        "transcript_text",
        "original_diarization_speaker",
        "casa_assigned_speaker",
        "attribution_changed",
        "decision_type",
        "attribution_confidence",
        "provisional_status",
        "acoustic_score",
        "temporal_score",
        "linguistic_score",
        "alternate_speaker",
        "alternate_confidence",
        "acoustic_evidence",
        "temporal_evidence",
        "linguistic_evidence",
        "all_evidence",
    ]

    with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        for r in records:
            row_dict = r.copy()
            row_dict["acoustic_evidence"] = " | ".join(row_dict["acoustic_evidence"])
            row_dict["temporal_evidence"] = " | ".join(row_dict["temporal_evidence"])
            row_dict["linguistic_evidence"] = " | ".join(row_dict["linguistic_evidence"])
            row_dict["all_evidence"] = " | ".join(row_dict["all_evidence"])
            writer.writerow(row_dict)

    # ── Summary Report ────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  EVALUATION SUMMARY: Podcast 1.mp3")
    print("=" * 90)
    print(f"Total Dialogue Phrases       : {len(records)}")
    print(f"Estimated Speaker Count      : {estimated_k}")
    print(f"CASA CONFIRM Decisions       : {confirm_count} ({confirm_count/len(records)*100:.1f}%)")
    print(f"CASA CORRECT Decisions       : {correct_count} ({correct_count/len(records)*100:.1f}%)")
    print(f"CASA UNCERTAIN Decisions     : {uncertain_count} ({uncertain_count/len(records)*100:.1f}%)")
    print(f"Provisional Early Phrases    : {provisional_count} ({provisional_count/len(records)*100:.1f}%)")
    print(f"Total Attributions Changed   : {changed_count}")
    print("-" * 90)
    print("Confidence Statistics:")
    print(f"  Mean Confidence            : {np.mean(confidence_values):.4f}")
    print(f"  Std Dev                    : {np.std(confidence_values):.4f}")
    print(f"  Min / Max / Median         : {np.min(confidence_values):.4f} / {np.max(confidence_values):.4f} / {np.median(confidence_values):.4f}")
    print("=" * 90)
    print("\nOutput Files Generated:")
    print(f"  JSON: {json_out_path}")
    print(f"  CSV : {csv_out_path}\n")


if __name__ == "__main__":
    run_podcast1_evaluation()
