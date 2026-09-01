"""
Whisper ASR Engine — V3 Phase 1E
Implements TranscriptionEngine using faster-whisper (CTranslate2 backend).
Supports CUDA with float16, CPU with int8, automatic device/compute type selection.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from engines.base import TranscriptionEngine
from utils.exceptions import AudioProcessingError
from utils.logger import logger


def _resolve_device_and_compute(device: str, compute_type: str):
    """Resolve 'auto' device and compute type to concrete values."""
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if compute_type == "auto":
        if device == "cuda":
            compute_type = "float16"
        else:
            compute_type = "int8"

    return device, compute_type


class WhisperTranscriptionEngine(TranscriptionEngine):
    """
    Whisper ASR via faster-whisper (CTranslate2).
    Model is loaded lazily on first use.

    Output format of `transcribe()`:
    {
        "text": str,                       # Full transcript text
        "language": str,                   # Detected language code (e.g. "en")
        "language_probability": float,
        "duration": float,                 # Audio duration in seconds
        "segments": [                      # Per-segment details
            {
                "id": int,
                "start": float,
                "end": float,
                "text": str,
                "avg_logprob": float,
                "no_speech_prob": float,
                "words": [                 # Word-level timestamps (if requested)
                    {"word": str, "start": float, "end": float, "probability": float}
                ]
            }
        ]
    }
    """

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 5,
        models_dir: Optional[Path] = None,
    ):
        self.model_size = model_size
        self.device_setting = device
        self.compute_type_setting = compute_type
        self.beam_size = beam_size
        self.models_dir = models_dir
        self._model = None
        self._resolved_device: Optional[str] = None
        self._resolved_compute: Optional[str] = None

    def _load_model(self) -> None:
        """Lazily load faster-whisper model."""
        if self._model is not None:
            return

        from faster_whisper import WhisperModel

        device, compute_type = _resolve_device_and_compute(
            self.device_setting, self.compute_type_setting
        )
        self._resolved_device = device
        self._resolved_compute = compute_type

        # download_root controls where model weights are cached
        download_root = str(self.models_dir / "faster_whisper") if self.models_dir else None

        logger.info(
            f"Loading Whisper model '{self.model_size}' "
            f"[device={device}, compute={compute_type}]..."
        )

        try:
            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
                download_root=download_root,
            )
            logger.info(f"Whisper model '{self.model_size}' loaded on {device}.")
        except Exception as exc:
            logger.error(f"Failed to load Whisper model '{self.model_size}': {exc}")
            raise AudioProcessingError(f"Whisper model load error: {exc}") from exc

    def is_available(self) -> bool:
        """Check if Whisper model can be loaded."""
        try:
            self._load_model()
            return self._model is not None
        except Exception:
            return False

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        word_timestamps: bool = True,
        vad_filter: bool = False,
        beam_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using faster-whisper.

        Args:
            audio_path: Path to the (normalized) WAV file.
            language: BCP-47 language code or None for auto-detect.
            word_timestamps: If True, request per-word timestamps from Whisper.
            vad_filter: Use Whisper's built-in VAD filter (False = use our Silero VAD layer).
            beam_size: Override beam size; defaults to instance beam_size.

        Returns:
            Normalized dict with keys: text, language, language_probability,
            duration, segments (each with optional words list).
        """
        self._load_model()

        if not audio_path.exists():
            raise AudioProcessingError(f"Audio file not found for transcription: {audio_path}")

        _beam = beam_size if beam_size is not None else self.beam_size

        logger.info(
            f"Whisper transcribing {audio_path.name} "
            f"[lang={language or 'auto'}, beam={_beam}, word_ts={word_timestamps}]"
        )

        try:
            segments_gen, info = self._model.transcribe(
                str(audio_path),
                language=language,
                beam_size=_beam,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
            )

            segments: List[Dict[str, Any]] = []
            full_text_parts: List[str] = []

            for seg in segments_gen:
                words_data: List[Dict[str, Any]] = []
                if word_timestamps and seg.words:
                    words_data = [
                        {
                            "word": w.word,
                            "start": round(w.start, 4),
                            "end": round(w.end, 4),
                            "probability": round(w.probability, 4),
                        }
                        for w in seg.words
                    ]

                seg_dict = {
                    "id": seg.id,
                    "start": round(seg.start, 4),
                    "end": round(seg.end, 4),
                    "text": seg.text.strip(),
                    "avg_logprob": round(seg.avg_logprob, 4),
                    "no_speech_prob": round(seg.no_speech_prob, 4),
                    "words": words_data,
                }
                segments.append(seg_dict)
                full_text_parts.append(seg.text.strip())

            full_text = " ".join(full_text_parts)

            result = {
                "text": full_text,
                "language": info.language,
                "language_probability": round(info.language_probability, 4),
                "duration": round(info.duration, 3),
                "segments": segments,
            }

            logger.info(
                f"Whisper transcription complete: {len(segments)} segments, "
                f"lang={info.language} ({info.language_probability:.2f}), "
                f"duration={info.duration:.1f}s"
            )
            return result

        except AudioProcessingError:
            raise
        except Exception as exc:
            logger.error(f"Whisper transcription failed for {audio_path}: {exc}")
            raise AudioProcessingError(f"Whisper transcription error: {exc}") from exc
