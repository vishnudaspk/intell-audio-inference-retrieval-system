# Intell Audio Inference & Retrieval System — V3

**Branch:** `v3-multimodal-intelligence`
**Safety tag:** `v3-pre-phase1`

## Overview

V3 is a ground-up redesign of the audio intelligence pipeline, replacing the
legacy Gentle forced-alignment + PocketSphinx stack with a modern, fully
local, GPU-accelerated foundation.

## V3 Pipeline Architecture

```
Audio/Video File
       │
       ▼
[1] AudioService.normalize_to_wav()        → 16kHz mono PCM WAV
       │
       ▼
[2] VADService (Silero VAD)                → Speech intervals: [(start, end, confidence), ...]
       │
       ▼
[3] TranscriptionService (faster-whisper) → Transcript + per-segment text & word timestamps
       │
       ├──────────────────────────────┐
       ▼                              ▼
[4] SpeakerEmbeddingService          [5] AcousticFeatureService
    (SpeechBrain ECAPA-TDNN)              (librosa: F0, RMS, spectral, MFCCs)
    → 192-dim L2-normalized vectors       → AcousticFeatures per segment
       │                              │
       └──────────────┬───────────────┘
                      ▼
[6] AudioSegment Assembly             → Unified AudioSegment objects
                      │
                      ▼
[7] SQLiteRepository.save_audio_segments()  → audio_segments table
```

## Technology Stack

| Component | Library | Model |
|-----------|---------|-------|
| Media Normalization | `torchaudio` | N/A — built-in resampler |
| Voice Activity Detection | `silero-vad` | Silero VAD v6 |
| Speech-to-Text | `faster-whisper` | `base.en` (CTranslate2) |
| Speaker Embeddings | `speechbrain` | `spkrec-ecapa-voxceleb` (192-dim ECAPA-TDNN) |
| Acoustic Features | `librosa` | N/A — signal processing |

## Hardware Targets

- **GPU:** NVIDIA RTX 4060 Laptop (8GB VRAM), CUDA 12.6
- **CPU:** Intel i9-13900H, 32GB RAM
- **OS:** Windows 11, Python 3.11.9

## Phases

- **[Phase 1](./phase-1.md)** — Audio Intelligence Foundation ✅
- Phase 2 — Speaker Diarization & Clustering _(planned)_
- Phase 3 — Temporal & Content Analysis _(planned)_

## Backward Compatibility

- Legacy tables (`transcript_words`, `transcripts`, `transcript_chunks`) preserved
- `AlignmentResult` model retained in `schemas/models.py` (deprecated, no runtime)
- New `audio_segments` table added for V3 output
- Indexing pipeline (`IndexingWorker`) and RAG stack unaffected by Phase 1

## Configuration

All V3 settings are in [`config/settings.py`](../../config/settings.py) under the
`# V3` sections. Override via `.env` file.

Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `WHISPER_MODEL_SIZE` | `base.en` | Whisper model variant |
| `WHISPER_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |
| `VAD_THRESHOLD` | `0.5` | Speech probability threshold |
| `SPEAKER_EMBEDDING_MODEL` | `speechbrain/spkrec-ecapa-voxceleb` | ECAPA-TDNN HF model |
| `EXTRACT_ACOUSTICS` | `True` | Enable librosa acoustic extraction |
