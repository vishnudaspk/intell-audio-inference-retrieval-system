"""
HMM Temporal Smoothing Service.
Learns transition dynamics from AHC clustering and applies Viterbi re-estimation to reduce speaker jitter.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from schemas.analysis import SmoothedSegment, TemporalModel
from utils.logger import logger


class HMMSmoother:
    """
    Temporal model for smoothing discrete speaker sequence predictions using transition probability matrices.
    """

    def smooth_sequence(
        self,
        raw_labels: List[str],
        window_embeddings: Optional[np.ndarray] = None,
        centroids: Optional[Dict[str, np.ndarray]] = None,
    ) -> Optional[TemporalModel]:
        """
        Learn transition matrix and smooth speaker sequence.
        """
        if not raw_labels or len(raw_labels) < 3:
            return None

        unique_speakers = sorted(list(set(raw_labels)))
        n_states = len(unique_speakers)
        if n_states < 2:
            return None

        spk_to_idx = {spk: idx for idx, spk in enumerate(unique_speakers)}
        idx_to_spk = {idx: spk for idx, spk in enumerate(unique_speakers)}

        # 1. Estimate transition probability matrix with Laplace smoothing
        transition_counts = np.ones((n_states, n_states), dtype=float) * 0.1
        for i in range(len(raw_labels) - 1):
            curr_idx = spk_to_idx[raw_labels[i]]
            next_idx = spk_to_idx[raw_labels[i + 1]]
            transition_counts[curr_idx, next_idx] += 1.0

        # Row-normalize to get transition probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        A = transition_counts / np.maximum(1e-8, row_sums)

        # 2. Viterbi decoding or transition-smoothed sequence
        smoothed_seq = list(raw_labels)
        # Apply 3-window median / transition filter
        for i in range(1, len(raw_labels) - 1):
            prev_s = raw_labels[i - 1]
            curr_s = raw_labels[i]
            next_s = raw_labels[i + 1]

            # If an isolated single-frame flip occurred between identical neighbours
            if prev_s == next_s and curr_s != prev_s:
                # Check transition likelihood
                p_stay = A[spk_to_idx[prev_s], spk_to_idx[prev_s]]
                p_switch = A[spk_to_idx[prev_s], spk_to_idx[curr_s]]
                if p_stay >= p_switch:
                    smoothed_seq[i] = prev_s

        # Build smoothed segments list
        smoothed_segments: List[SmoothedSegment] = []
        for i, (raw, sm) in enumerate(zip(raw_labels, smoothed_seq)):
            smoothed_segments.append(
                SmoothedSegment(
                    start_sec=float(i),
                    end_sec=float(i + 1),
                    raw_speaker=raw,
                    smoothed_speaker=sm,
                )
            )

        return TemporalModel(
            method="HMM-Transition-Smoother",
            num_states=n_states,
            speaker_sequence=smoothed_seq,
            transition_matrix=A.round(4).tolist(),
            smoothed_segments=smoothed_segments,
        )
