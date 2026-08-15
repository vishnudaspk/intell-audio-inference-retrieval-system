"""
Speaker Embedding Service — V3 Phase 1F
Generates 192-dim speaker embeddings using SpeechBrain ECAPA-TDNN.
Model: speechbrain/spkrec-ecapa-voxceleb
Handles short segments gracefully (< 0.5s returns zero embedding).
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from config.settings import settings
from utils.exceptions import AudioProcessingError
from utils.logger import logger

ECAPA_EMBEDDING_DIM = 192
MIN_SEGMENT_DURATION_SEC = 0.5  # SpeechBrain ECAPA needs ≥ ~0.5s to produce meaningful embedding


class SpeakerEmbeddingService:
    """
    Service for extracting per-segment speaker embedding vectors using
    SpeechBrain's ECAPA-TDNN model (spkrec-ecapa-voxceleb, 192-dim).

    Usage:
        service = SpeakerEmbeddingService()
        embedding = service.embed_segment(wav_path, start_sec=1.2, end_sec=5.6)
    """

    def __init__(
        self,
        model_source: Optional[str] = None,
        savedir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.model_source = model_source or settings.SPEAKER_EMBEDDING_MODEL
        self.savedir = savedir or (settings.models_dir / "speechbrain")
        self._device_setting = device or settings.SPEAKER_EMBEDDING_DEVICE
        self._model = None
        self._resolved_device: Optional[str] = None

    def _resolve_device(self) -> str:
        if self._device_setting == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if self._device_setting == "cuda":
            return "cuda:0"
        return self._device_setting

    def _load_model(self) -> None:
        """Lazily load SpeechBrain ECAPA-TDNN from HuggingFace Hub."""
        if self._model is not None:
            return

        try:
            from speechbrain.inference.classifiers import EncoderClassifier
        except ImportError:
            # Fallback for older SpeechBrain releases
            from speechbrain.pretrained import EncoderClassifier

        device = self._resolve_device()
        self._resolved_device = device
        self.savedir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Loading SpeechBrain ECAPA-TDNN model '{self.model_source}' "
            f"[device={device}, savedir={self.savedir}]..."
        )

        try:
            self._model = EncoderClassifier.from_hparams(
                source=self.model_source,
                savedir=str(self.savedir),
                run_opts={"device": device},
            )
            self._model.eval()
            logger.info("SpeechBrain ECAPA-TDNN loaded successfully.")
        except Exception as exc:
            logger.error(f"Failed to load SpeechBrain ECAPA-TDNN: {exc}")
            raise AudioProcessingError(f"SpeechBrain model load error: {exc}") from exc

    def is_available(self) -> bool:
        try:
            self._load_model()
            return self._model is not None
        except Exception:
            return False

    def embed_segment(
        self,
        wav_path: Path,
        start_sec: float,
        end_sec: float,
    ) -> np.ndarray:
        """
        Extract a 192-dim L2-normalized speaker embedding for an audio segment.

        For segments shorter than MIN_SEGMENT_DURATION_SEC, returns a zero vector
        instead of crashing to allow pipeline continuity.

        Args:
            wav_path: Normalized 16kHz mono WAV file path.
            start_sec: Segment start in seconds.
            end_sec: Segment end in seconds.

        Returns:
            numpy array of shape (192,), L2-normalized.
        """
        self._load_model()

        duration = end_sec - start_sec
        if duration < MIN_SEGMENT_DURATION_SEC:
            logger.debug(
                f"Segment [{start_sec:.2f}–{end_sec:.2f}]s is too short ({duration:.3f}s < {MIN_SEGMENT_DURATION_SEC}s). "
                "Returning zero embedding."
            )
            return np.zeros(ECAPA_EMBEDDING_DIM, dtype=np.float32)

        if not wav_path.exists():
            raise AudioProcessingError(f"WAV file not found for embedding: {wav_path}")

        try:
            import soundfile as sf

            audio_np, sr = sf.read(str(wav_path), dtype="float32")
            # soundfile returns (samples,) for mono; convert to (1, samples) tensor
            if audio_np.ndim == 1:
                audio_np = audio_np[np.newaxis, :]
            else:
                audio_np = audio_np.T  # (channels, samples)
            waveform = torch.from_numpy(audio_np)

            # Slice to requested time segment
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            segment = waveform[:, start_sample:end_sample]

            if segment.shape[-1] == 0:
                logger.warning(f"Empty segment slice for [{start_sec:.2f}–{end_sec:.2f}]s — returning zero embedding.")
                return np.zeros(ECAPA_EMBEDDING_DIM, dtype=np.float32)

            # Move to model device
            segment = segment.to(self._resolved_device)

            with torch.no_grad():
                # SpeechBrain EncoderClassifier.encode_batch expects (batch, time)
                embedding = self._model.encode_batch(segment)
                # Shape: (1, 1, 192) → squeeze to (192,)
                embedding_np = embedding.squeeze().cpu().numpy().astype(np.float32)

            # L2-normalize
            norm = np.linalg.norm(embedding_np)
            if norm > 1e-8:
                embedding_np = embedding_np / norm

            return embedding_np

        except AudioProcessingError:
            raise
        except Exception as exc:
            logger.error(f"Speaker embedding failed for segment [{start_sec:.2f}–{end_sec:.2f}]s: {exc}")
            raise AudioProcessingError(f"Speaker embedding error: {exc}") from exc

    def embed_segments(
        self,
        wav_path: Path,
        segments: List[Tuple[float, float]],
    ) -> List[np.ndarray]:
        """
        Batch-embed multiple (start_sec, end_sec) segments from a single WAV file.

        Returns a list of embeddings in the same order as the input segments.
        Short segments receive zero embeddings gracefully.
        """
        self._load_model()

        embeddings: List[np.ndarray] = []
        for i, (start, end) in enumerate(segments):
            try:
                emb = self.embed_segment(wav_path, start, end)
                embeddings.append(emb)
            except AudioProcessingError as exc:
                logger.warning(f"Embedding failed for segment {i} [{start:.2f}–{end:.2f}]s: {exc}. Using zero.")
                embeddings.append(np.zeros(ECAPA_EMBEDDING_DIM, dtype=np.float32))

        return embeddings

    def cluster_segments(
        self,
        embeddings: List[Optional[np.ndarray]],
        similarity_threshold: float = 0.50,
    ) -> List[Optional[str]]:
        """
        Assign meaningful speaker labels ('Speaker 1', 'Speaker 2', etc.) to segments
        based on cosine similarity clustering of ECAPA-TDNN speaker embeddings.

        Segments with None or zero embeddings receive None.
        Preserves deterministic, grounded speaker grouping without fake identities.
        """
        labels: List[Optional[str]] = [None] * len(embeddings)
        centroids: List[np.ndarray] = []

        for idx, emb in enumerate(embeddings):
            if emb is None:
                continue
            emb_arr = np.asarray(emb, dtype=np.float32).flatten()
            norm = np.linalg.norm(emb_arr)
            if norm < 1e-6:
                # Segment was too short or silent
                continue

            emb_norm = emb_arr / norm

            if not centroids:
                centroids.append(emb_norm)
                labels[idx] = "Speaker 1"
            else:
                sims = [float(np.dot(emb_norm, c)) for c in centroids]
                max_sim_idx = int(np.argmax(sims))
                if sims[max_sim_idx] >= similarity_threshold:
                    labels[idx] = f"Speaker {max_sim_idx + 1}"
                    # Update running centroid
                    centroids[max_sim_idx] = centroids[max_sim_idx] + emb_norm
                    c_norm = np.linalg.norm(centroids[max_sim_idx])
                    if c_norm > 1e-6:
                        centroids[max_sim_idx] /= c_norm
                else:
                    centroids.append(emb_norm)
                    labels[idx] = f"Speaker {len(centroids)}"

        return labels
