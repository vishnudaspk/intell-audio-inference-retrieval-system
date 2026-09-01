# Phase 1 — Audio Intelligence Foundation

**Status:** ✅ Complete  
**Branch:** `v3-multimodal-intelligence`

## Objective

Establish the foundational audio processing pipeline for V3:
normalized ingestion → speech detection → transcription → speaker representation → acoustic analysis.

## Implemented Components

### Phase 1A — Engine Cleanup
- Deleted `engines/gentle_engine.py`, `engines/pocketsphinx_engine.py`, `services/alignment_service.py`
- Refactored `engines/base.py`: defined `VADEngine`, `TranscriptionEngine`, `AudioSource` protocols
- Updated `engines/__init__.py`, `services/__init__.py`, `services/health_service.py`

### Phase 1B — Configuration & Dependencies
- Updated `config/settings.py` with all V3 settings
- Updated `.env.example` removing legacy keys
- Updated `requirements.txt`: added `faster-whisper`, `speechbrain`, `librosa`, `soundfile`, `silero-vad`
- Installed `torch==2.13.0+cu126`, `torchaudio==2.11.0+cu126` (CUDA 12.6, RTX 4060)

### Phase 1C — Media Ingestion (`services/audio_service.py`)
- Supports: MP3, WAV, M4A, FLAC, OGG, MP4 (audio track)
- Normalization: `torchaudio` for resampling → `soundfile` for writing 16-bit PCM WAV
- Output: 16 kHz mono WAV at `data/audio/{asset_id}.wav`
- Backward-compatible alias: `convert_to_wav()` → `normalize_to_wav()`

### Phase 1D — Voice Activity Detection
- **`engines/vad_engine.py`**: `SileroVADEngine` — lazy loads Silero VAD v6, returns `(start, end, confidence)` tuples
- **`services/vad_service.py`**: `VADService` — settings-based defaults, `merge_close_segments()`, `filter_short_segments()`

### Phase 1E — Whisper ASR
- **`engines/whisper_engine.py`**: `WhisperTranscriptionEngine` — `faster-whisper` CTranslate2, auto CUDA/CPU, word-level timestamps
- **`services/transcription_service.py`**: Converts raw output to `Transcript`/`TranscriptSegment`/`TranscriptWord` domain objects
- **`engines/factory.py`**: Updated to resolve `whisper` ASR and `silero` VAD engines

### Phase 1F — Speaker Embeddings (`services/speaker_embedding_service.py`)
- Model: `speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN, 192-dim)
- Lazy-loads from HuggingFace Hub to `data/models/speechbrain/`
- Short segments (< 0.5s): returns zero embedding without crashing
- Output: L2-normalized `numpy.ndarray` of shape `(192,)`
- Batch API: `embed_segments(wav_path, [(start, end), ...])`

### Phase 1G — Acoustic Features (`services/acoustic_service.py`)
- Stateless `librosa`-based extraction:
  - **Pitch/F0**: `librosa.pyin` — mean, median, min, max, std, voiced fraction
  - **Energy**: RMS mean, std, max
  - **Spectral**: centroid, bandwidth, rolloff, zero-crossing rate
  - **MFCCs**: 13 coefficients (means)
- `AcousticFeatures` dataclass with `to_dict()`/`from_dict()` for JSON serialization
- Batch API: `extract_batch(wav_path, [(start, end), ...])`

### Phase 1H — Unified Data Model
- **`schemas/models.py`**: Added `AudioSegment` — unified V3 segment aggregating VAD interval, ASR text/words, speaker embedding, acoustic features
- **`database/sqlite_db.py`**: Added `audio_segments` table; `save_audio_segments()` and `get_audio_segments()` repository methods

### Phase 1I — Pipeline Orchestrator (`workers/audio_worker.py`)
Full 7-stage pipeline:
1. Normalization (AudioService)
2. VAD (VADService) 
3. ASR (TranscriptionService + backward-compat transcript save)
4. Speaker Embeddings (SpeakerEmbeddingService)
5. Acoustic Features (AcousticFeatureService)
6. AudioSegment assembly with temporal VAD↔Whisper matching
7. SQLite persistence

Features: per-stage timing, graceful non-fatal embedding/acoustic failures, `extract_embeddings`/`extract_acoustics` flags.

### Phase 1J — Tests (40 tests, 100% pass)
| File | Tests | Coverage |
|------|-------|----------|
| `tests/unit/test_vad_service.py` | 10 | Merge, filter, delegation |
| `tests/unit/test_whisper_asr.py` | 11 | Language mapping, transcript assembly |
| `tests/unit/test_acoustic_service.py` | 9 | F0, RMS, MFCC, edge cases |
| `tests/unit/test_speaker_embeddings.py` | 3 | Zero embedding, L2 norm, batch |
| `tests/integration/test_phase1_pipeline.py` | 7 | Full pipeline, error handling, disabled features |

### Phase 1K — Documentation
- `docs/v3/README.md` — architecture, stack, hardware targets
- `docs/v3/phase-1.md` — this document

## Notable Design Decisions

- **`soundfile` instead of `torchaudio.load/save`**: torchaudio 2.11+cu126 requires `torchcodec` for I/O. All file I/O uses `soundfile` directly; `torchaudio` is retained only for Resampling transforms.
- **VAD-first, then Whisper**: We run Silero VAD first for speaker-aligned intervals, then transcribe the full file with Whisper and match segments temporally.
- **Non-crashing acoustics/embeddings**: Both stages are wrapped with try/except; failures produce zero vectors / empty features rather than failing the whole pipeline.
- **Legacy tables preserved**: `transcript_words`, `transcripts`, `transcript_chunks` remain intact for indexing worker compatibility.

## Running Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_vad_service.py tests/unit/test_whisper_asr.py tests/unit/test_acoustic_service.py tests/unit/test_speaker_embeddings.py tests/integration/test_phase1_pipeline.py -v
```

Expected: **40 passed**.
