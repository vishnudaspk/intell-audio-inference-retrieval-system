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
        durations: Optional[List[float]] = None,
        similarity_threshold: float = 0.65,
        max_speakers: int = 10,
    ) -> List[Optional[str]]:
        """
        Assign grounded speaker labels ('Speaker 1', 'Speaker 2', etc.) to pre-extracted segment embeddings.
        Preserves backward compatibility for unit tests and direct segment cluster tasks.
        """
        labels: List[Optional[str]] = [None] * len(embeddings)
        if not embeddings:
            return labels

        valid_indices: List[int] = []
        valid_embs: List[np.ndarray] = []

        for i, emb in enumerate(embeddings):
            if emb is None:
                continue
            emb_arr = np.asarray(emb, dtype=np.float32).flatten()
            norm = np.linalg.norm(emb_arr)
            if norm < 1e-6:
                continue
            valid_indices.append(i)
            valid_embs.append(emb_arr / norm)

        if not valid_embs:
            return labels

        if len(valid_embs) == 1:
            labels[valid_indices[0]] = "Speaker 1"
            return labels

        X = np.array(valid_embs)
        n_samples = len(X)

        if n_samples == 2:
            cos_sim = float(np.dot(X[0], X[1]))
            if cos_sim >= similarity_threshold:
                labels[valid_indices[0]] = "Speaker 1"
                labels[valid_indices[1]] = "Speaker 1"
            else:
                labels[valid_indices[0]] = "Speaker 1"
                labels[valid_indices[1]] = "Speaker 2"
            return labels

        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist, squareform
        from sklearn.metrics import silhouette_score

        dist_condensed = pdist(X, metric="cosine")
        dist_mat = squareform(dist_condensed)

        if np.max(dist_mat) < (1.0 - similarity_threshold):
            for idx in valid_indices:
                labels[idx] = "Speaker 1"
            return labels

        Z = linkage(dist_condensed, method="average")
        best_k = 2
        best_score = -1.0
        max_possible_k = min(max_speakers, n_samples - 1)

        if max_possible_k >= 2:
            for k in range(2, max_possible_k + 1):
                cids = fcluster(Z, t=k, criterion="maxclust")
                if len(set(cids)) > 1:
                    score = float(silhouette_score(dist_mat, cids, metric="precomputed"))
                    if score > best_score:
                        best_score = score
                        best_k = k

            if best_score < 0.10 and np.mean(dist_mat) < (1.0 - similarity_threshold + 0.15):
                best_k = 1

        if best_k == 1:
            for idx in valid_indices:
                labels[idx] = "Speaker 1"
            return labels

        raw_cids = fcluster(Z, t=best_k, criterion="maxclust")
        cluster_order: dict = {}
        next_num = 1
        for cid in raw_cids:
            if cid not in cluster_order:
                cluster_order[cid] = f"Speaker {next_num}"
                next_num += 1

        for idx, cid in zip(valid_indices, raw_cids):
            labels[idx] = cluster_order[cid]

        return labels

    def diarize_audio(
        self,
        wav_path: Path,
        speech_intervals: List[Tuple[float, float]],
        transcript_words: Optional[List[dict]] = None,
        win_len: float = 2.0,
        win_hop: float = 1.0,
        max_speakers: int = 10,
    ) -> Tuple[List[dict], dict]:
        """
        Full speaker diarization pipeline:
          1. Generate overlapping speaker analysis windows (~1.5–2.5s) over active speech regions.
          2. Extract 192-dim ECAPA-TDNN speaker embeddings for all windows.
          3. L2-normalize and compute affinity/cosine distance matrix with power sharpening.
          4. Automatically estimate speaker count via spectral Laplacian eigengap / silhouette.
          5. Cluster window embeddings into distinct speakers (Speaker 1, Speaker 2, ...).
          6. Map speaker identities back to Whisper word/phrase boundaries via temporal overlap & voting.

        Returns:
            (diarized_segments, diagnostics_dict)
        """
        self._load_model()

        if not speech_intervals:
            return [], {
                "num_windows": 0,
                "num_embeddings": 0,
                "embedding_dim": ECAPA_EMBEDDING_DIM,
                "estimated_speakers": 0,
                "distinct_speakers": 0,
                "cluster_sizes": {},
                "mean_cosine_sim": 0.0,
            }

        # Step 1: Generate overlapping analysis windows (used for K estimation
        # when transcript_words is absent, and as fallback at all times)
        windows: List[Tuple[float, float]] = []
        for st, et in speech_intervals:
            dur = et - st
            if dur <= win_len:
                windows.append((round(st, 3), round(et, 3)))
            else:
                cur = st
                while cur + 0.8 <= et:
                    w_end = min(cur + win_len, et)
                    windows.append((round(cur, 3), round(w_end, 3)))
                    cur += win_hop
                if windows and windows[-1][1] < et - 0.3:
                    windows.append((round(max(st, et - win_len), 3), round(et, 3)))

        if not windows:
            return [], {
                "num_windows": 0,
                "num_embeddings": 0,
                "embedding_dim": ECAPA_EMBEDDING_DIM,
                "estimated_speakers": 0,
                "distinct_speakers": 0,
                "cluster_sizes": {},
                "mean_cosine_sim": 0.0,
            }

        # Step 2: Extract embeddings for VAD sliding windows
        raw_embs = self.embed_segments(wav_path, windows)
        valid_indices = [i for i, e in enumerate(raw_embs) if np.linalg.norm(e) > 1e-6]

        if not valid_indices:
            return [], {
                "num_windows": len(windows),
                "num_embeddings": 0,
                "embedding_dim": ECAPA_EMBEDDING_DIM,
                "estimated_speakers": 0,
                "distinct_speakers": 0,
                "cluster_sizes": {},
                "mean_cosine_sim": 0.0,
            }

        valid_windows = [windows[i] for i in valid_indices]
        X = np.array([raw_embs[i] / np.linalg.norm(raw_embs[i]) for i in valid_indices])
        n_samples = len(X)

        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist, squareform
        from sklearn.metrics import silhouette_score

        dist_condensed = pdist(X, metric="cosine")
        dist_mat = squareform(dist_condensed)
        cos_sim_mat = 1.0 - dist_mat
        mean_sim = float(np.mean(cos_sim_mat[np.triu_indices(n_samples, k=1)])) if n_samples > 1 else 1.0

        # Step 3 & 4: Eigengap / silhouette speaker count estimation
        if n_samples == 1:
            best_k = 1
        elif n_samples == 2:
            best_k = 1 if float(np.dot(X[0], X[1])) >= 0.65 else 2
        else:
            # Affinity matrix with cubic power sharpening
            A = np.maximum(0, np.dot(X, X.T)) ** 3
            np.fill_diagonal(A, 0.0)
            d_sum = np.sum(A, axis=1)
            d_inv_sqrt = 1.0 / np.sqrt(np.maximum(1e-8, d_sum))
            L_sym = np.eye(n_samples) - (A * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
            eigenvalues, _ = np.linalg.eigh(L_sym)

            k_candidates = list(range(2, min(max_speakers + 1, n_samples)))
            if k_candidates:
                gaps = [float(eigenvalues[k] - eigenvalues[k - 1]) for k in k_candidates]
                best_k = k_candidates[int(np.argmax(gaps))]
            else:
                best_k = 2

            # Check for 1 speaker (if maximum distance across all windows is small)
            if np.max(dist_mat) < 0.35 or (mean_sim > 0.80):
                best_k = 1

        # Step 5: Hierarchical clustering of VAD windows (used as fallback)
        if best_k == 1:
            raw_cids = np.ones(n_samples, dtype=int)
        else:
            Z = linkage(dist_condensed, method="average")
            raw_cids = fcluster(Z, t=best_k, criterion="maxclust")

        # Map cluster IDs to chronological labels ('Speaker 1', 'Speaker 2', ...)
        cluster_order: dict = {}
        next_num = 1
        for cid in raw_cids:
            if cid not in cluster_order:
                cluster_order[cid] = f"Speaker {next_num}"
                next_num += 1

        window_labels = [cluster_order[cid] for cid in raw_cids]

        # Compute cluster centroids
        cluster_centroids: dict = {}
        for cid in set(raw_cids):
            c_embs = X[raw_cids == cid]
            mean_vec = np.mean(c_embs, axis=0)
            mean_vec /= np.linalg.norm(mean_vec)
            cluster_centroids[cid] = mean_vec

        # Step 6: Map to transcript word phrases or VAD regions
        diarized_segments: List[dict] = []

        # ── helper: read a field from dict, sqlite3.Row, or attribute ─────────
        def _get_field(obj, *keys):
            for k in keys:
                if hasattr(obj, k):
                    val = getattr(obj, k)
                    if val is not None:
                        return val
                if isinstance(obj, dict) and k in obj:
                    val = obj[k]
                    if val is not None:
                        return val
                if hasattr(obj, "__getitem__"):
                    try:
                        val = obj[k]
                        if val is not None:
                            return val
                    except Exception:
                        pass
            return None

        if transcript_words:
            # ── Build dialogue phrases from word-gap / punctuation boundaries ─
            phrases: List[dict] = []
            cur_words = [transcript_words[0]]

            for w in transcript_words[1:]:
                w_st = _get_field(w, "start_time", "start") or 0.0
                last_et = _get_field(cur_words[-1], "end_time", "end") or 0.0
                gap = w_st - last_et
                last_word = str(_get_field(cur_words[-1], "word") or "").strip()
                is_punct = last_word.endswith((".", "?", "!", "...", ";", ":"))

                if gap >= 0.35 or (is_punct and gap >= 0.15):
                    p_st = _get_field(cur_words[0], "start_time", "start") or 0.0
                    p_et = _get_field(cur_words[-1], "end_time", "end") or 0.0
                    p_txt = "".join(str(_get_field(x, "word") or "") for x in cur_words).strip()
                    phrases.append({
                        "start_sec": round(p_st, 3),
                        "end_sec": round(p_et, 3),
                        "duration_sec": round(p_et - p_st, 3),
                        "text": p_txt,
                        "words": cur_words,
                    })
                    cur_words = [w]
                else:
                    cur_words.append(w)

            if cur_words:
                p_st = _get_field(cur_words[0], "start_time", "start") or 0.0
                p_et = _get_field(cur_words[-1], "end_time", "end") or 0.0
                p_txt = "".join(str(_get_field(x, "word") or "") for x in cur_words).strip()
                phrases.append({
                    "start_sec": round(p_st, 3),
                    "end_sec": round(p_et, 3),
                    "duration_sec": round(p_et - p_st, 3),
                    "text": p_txt,
                    "words": cur_words,
                })

            # ── Phrase-aligned analysis windows ───────────────────────────────
            # For each dialogue phrase we generate analysis windows that are
            # ANCHORED to the phrase boundary so they never mix two speakers:
            #   • short phrases (≤ LONG_PHRASE): one center-padded window
            #   • long phrases  (> LONG_PHRASE): 1.5s sub-windows with 0.75s hop
            # We track which phrase each window belongs to via win_phrase_idx so
            # that attribution is a simple majority-vote per phrase.
            MIN_ECAPA_DUR = 1.2   # min window length fed to ECAPA
            LONG_PHRASE   = 2.5   # threshold to switch to sub-windows
            SUB_WIN_LEN   = 1.5   # sub-window length for long phrases
            SUB_WIN_HOP   = 0.75  # sub-window hop

            phrase_windows: List[Tuple[float, float]] = []
            win_phrase_idx: List[int] = []

            for p_idx, phrase in enumerate(phrases):
                pst, pet = phrase["start_sec"], phrase["end_sec"]
                dur = pet - pst
                if dur <= LONG_PHRASE:
                    pad = max(0.0, (MIN_ECAPA_DUR - dur) / 2.0)
                    phrase_windows.append((round(max(0.0, pst - pad), 3), round(pet + pad, 3)))
                    win_phrase_idx.append(p_idx)
                else:
                    cur = pst
                    while cur + 0.5 <= pet:
                        w_et = min(cur + SUB_WIN_LEN, pet)
                        phrase_windows.append((round(cur, 3), round(w_et, 3)))
                        win_phrase_idx.append(p_idx)
                        cur += SUB_WIN_HOP
                    if phrase_windows and phrase_windows[-1][1] < pet - 0.2:
                        phrase_windows.append((round(max(pst, pet - SUB_WIN_LEN), 3), round(pet, 3)))
                        win_phrase_idx.append(p_idx)

            # ── Extract phrase-window embeddings ──────────────────────────────
            phrase_raw_embs = self.embed_segments(wav_path, phrase_windows)
            p_valid = [i for i, e in enumerate(phrase_raw_embs) if np.linalg.norm(e) > 1e-6]

            if p_valid:
                PX = np.array([phrase_raw_embs[i] / np.linalg.norm(phrase_raw_embs[i]) for i in p_valid])
                p_valid_src = [win_phrase_idx[i] for i in p_valid]
                pn = len(PX)

                # Re-estimate K on phrase-anchored embeddings (more accurate than VAD windows)
                if pn >= 4:
                    PA = np.maximum(0, np.dot(PX, PX.T)) ** 3
                    np.fill_diagonal(PA, 0.0)
                    pd_sum = np.sum(PA, axis=1)
                    pd_inv_sqrt = 1.0 / np.sqrt(np.maximum(1e-8, pd_sum))
                    PL = np.eye(pn) - (PA * pd_inv_sqrt[:, None]) * pd_inv_sqrt[None, :]
                    p_eig, _ = np.linalg.eigh(PL)
                    pk_cands = list(range(2, min(max_speakers + 1, pn)))
                    if pk_cands:
                        p_gaps = [float(p_eig[k] - p_eig[k - 1]) for k in pk_cands]
                        phrase_k = pk_cands[int(np.argmax(p_gaps))]
                    else:
                        phrase_k = best_k
                    p_dist_mat = squareform(pdist(PX, metric="cosine"))
                    if float(np.max(p_dist_mat)) < 0.35 or float(np.mean(1.0 - p_dist_mat)) > 0.80:
                        phrase_k = 1
                else:
                    phrase_k = best_k

                if phrase_k == 1:
                    p_cids = np.ones(pn, dtype=int)
                else:
                    PZ = linkage(pdist(PX, metric="cosine"), method="average")
                    p_cids = fcluster(PZ, t=phrase_k, criterion="maxclust")

                # Phrase-window cluster centroids
                pw_centroids: dict = {}
                for cid in set(p_cids):
                    mask = p_cids == cid
                    mv = np.mean(PX[mask], axis=0)
                    mv /= np.linalg.norm(mv)
                    pw_centroids[cid] = mv

                # Majority-vote: best cluster for each phrase
                from collections import Counter
                phrase_votes: dict = {}
                for arr_i, src_p in enumerate(p_valid_src):
                    phrase_votes.setdefault(src_p, Counter())[p_cids[arr_i]] += 1

                # Assign speakers in chronological (first-occurrence) order
                phrase_cluster_order: dict = {}
                pn_num = 1
                phrase_assigned_cids: List[int] = []
                # V3.2: collect one representative embedding per phrase for CASA
                phrase_repr_embeddings: List[Optional[np.ndarray]] = []
                for p_idx in range(len(phrases)):
                    votes = phrase_votes.get(p_idx)
                    if votes:
                        best_cid = votes.most_common(1)[0][0]
                        # Representative embedding: the phrase-window embedding closest to centroid
                        cid_mask_indices = [
                            arr_i for arr_i, src_p in enumerate(p_valid_src)
                            if src_p == p_idx and p_cids[arr_i] == best_cid
                        ]
                        if cid_mask_indices:
                            rep_emb = PX[cid_mask_indices[0]]
                        else:
                            rep_emb = None
                    else:
                        # No phrase-window covered this phrase → direct embedding
                        pst = phrases[p_idx]["start_sec"]
                        pet = phrases[p_idx]["end_sec"]
                        dur = pet - pst
                        pad = max(0.0, (MIN_ECAPA_DUR - dur) / 2.0)
                        fb_emb = self.embed_segment(wav_path, max(0.0, pst - pad), pet + pad)
                        if np.linalg.norm(fb_emb) > 1e-6:
                            fb_norm = fb_emb / np.linalg.norm(fb_emb)
                            sims = {cid: float(np.dot(fb_norm, c)) for cid, c in pw_centroids.items()}
                            best_cid = max(sims, key=sims.get)
                            rep_emb = fb_norm
                        else:
                            best_cid = p_cids[0]
                            rep_emb = None
                    phrase_assigned_cids.append(best_cid)
                    phrase_repr_embeddings.append(rep_emb)
                    if best_cid not in phrase_cluster_order:
                        phrase_cluster_order[best_cid] = f"Speaker {pn_num}"
                        pn_num += 1

                for phrase, cid in zip(phrases, phrase_assigned_cids):
                    phrase["speaker_label"] = phrase_cluster_order[cid]
                    diarized_segments.append(phrase)

                # V3.2: build speaker-label-keyed centroids for CASA
                speaker_label_centroids: dict = {
                    phrase_cluster_order[cid]: vec
                    for cid, vec in pw_centroids.items()
                    if cid in phrase_cluster_order
                }

            else:
                # All phrase-window embeddings were zero → fall back to VAD overlap voting
                for phrase in phrases:
                    pst, pet = phrase["start_sec"], phrase["end_sec"]
                    overlaps_fb: List[Tuple[float, int]] = []
                    for (w_st, w_et), cid in zip(valid_windows, raw_cids):
                        overlap_dur = min(pet, w_et) - max(pst, w_st)
                        if overlap_dur > 0.05:
                            overlaps_fb.append((overlap_dur, cid))
                    if overlaps_fb:
                        cid_weights_fb: dict = {}
                        for dur_w, cid in overlaps_fb:
                            cid_weights_fb[cid] = cid_weights_fb.get(cid, 0.0) + dur_w
                        best_cid = max(cid_weights_fb, key=cid_weights_fb.get)
                    else:
                        best_cid = raw_cids[0]
                    phrase["speaker_label"] = cluster_order[best_cid]
                    diarized_segments.append(phrase)
                # V3.2: no phrase-window embeddings available in this path
                phrase_repr_embeddings = [None] * len(phrases)
                speaker_label_centroids = {}
        else:
            # Fallback to speech intervals when no transcript words
            for st, et in speech_intervals:
                overlaps = []
                for (w_st, w_et), cid in zip(valid_windows, raw_cids):
                    overlap_dur = min(et, w_et) - max(st, w_st)
                    if overlap_dur > 0.05:
                        overlaps.append((overlap_dur, cid))
                best_cid = raw_cids[0]
                if overlaps:
                    cid_weights: dict = {}
                    for dur_w, cid in overlaps:
                        cid_weights[cid] = cid_weights.get(cid, 0.0) + dur_w
                    best_cid = max(cid_weights, key=cid_weights.get)

                diarized_segments.append({
                    "start_sec": round(st, 3),
                    "end_sec": round(et, 3),
                    "duration_sec": round(et - st, 3),
                    "text": "",
                    "words": [],
                    "speaker_label": cluster_order[best_cid],
                })
            # V3.2: no phrase-level data when using VAD-interval fallback
            phrase_repr_embeddings = [None] * len(diarized_segments)
            speaker_label_centroids = {}

        # Step 7: Build diagnostics
        from collections import Counter
        cluster_sizes = Counter(seg["speaker_label"] for seg in diarized_segments)
        diagnostics = {
            "num_windows": len(valid_windows),
            "num_embeddings": len(X),
            "embedding_dim": ECAPA_EMBEDDING_DIM,
            "estimated_speakers": best_k,
            "distinct_speakers": len(cluster_sizes),
            "cluster_sizes": dict(cluster_sizes),
            "mean_cosine_sim": round(mean_sim, 4),
            # V3.2 CASA support — phrase-level embeddings and label-keyed centroids
            "phrase_embeddings": phrase_repr_embeddings,
            "speaker_centroids": speaker_label_centroids,
        }

        logger.info(f"[Diarization Diagnostics] {diagnostics}")
        return diarized_segments, diagnostics
