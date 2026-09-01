"""
CASA Configuration — V3.2 Speaker Intelligence
All tunable weights, thresholds, and flags for the Conversation-Aware Speaker
Attribution engine.  Nothing in casa_engine.py is hardcoded; every numeric
constant lives here so teams can tune without touching engine logic.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CASAConfig:
    """
    Configuration for the Conversation-Aware Speaker Attribution (CASA) engine.

    Weight groups must satisfy:
      w_acoustic + w_temporal + w_linguistic == 1.0   (normal phrases)
      short_utt_w_acoustic + short_utt_w_temporal + short_utt_w_linguistic == 1.0

    All thresholds are in [0.0, 1.0] unless otherwise noted.
    """

    # ── Enable / disable ──────────────────────────────────────────────────────
    enable_casa: bool = True
    """Master switch.  When False, CASA is a transparent pass-through."""

    enable_early_dialogue_stabilization: bool = True
    """Re-evaluate provisional (early-dialogue) phrases after the full CASA pass."""

    # ── Fusion weights — normal phrases ───────────────────────────────────────
    w_acoustic: float = 0.55
    """Weight for acoustic evidence (ECAPA embedding similarity, centroid gap)."""

    w_temporal: float = 0.25
    """Weight for temporal evidence (pause, continuity, turn signals)."""

    w_linguistic: float = 0.20
    """Weight for linguistic / dialogue-pattern evidence."""

    # ── Fusion weights — short utterances ────────────────────────────────────
    short_utt_w_acoustic: float = 0.30
    """Reduced acoustic weight for very short phrases (ECAPA unreliable < 0.8 s)."""

    short_utt_w_temporal: float = 0.35
    """Boosted temporal weight for short phrases."""

    short_utt_w_linguistic: float = 0.35
    """Boosted linguistic weight for short phrases."""

    # ── Short-utterance detection ─────────────────────────────────────────────
    short_utt_max_words: int = 3
    """Phrases with this many words or fewer are treated as short utterances."""

    short_utt_max_duration_sec: float = 0.8
    """Phrases shorter than this (seconds) are treated as short utterances."""

    # ── Decision thresholds ───────────────────────────────────────────────────
    confirm_threshold: float = 0.70
    """CASA confidence ≥ this → CONFIRM original acoustic label."""

    uncertain_threshold: float = 0.45
    """CASA confidence < this AND correction evidence present → CORRECT label;
    otherwise → UNCERTAIN (keep original, lower confidence exposed)."""

    # ── False-correction guard ────────────────────────────────────────────────
    min_correction_confidence_delta: float = 0.15
    """Minimum confidence *advantage* of the proposed new speaker over the
    original speaker before CASA is allowed to emit CORRECT.
    Prevents micro-corrections on ambiguous evidence."""

    min_correction_signals: int = 1
    """Minimum number of independent non-acoustic signals (temporal or linguistic)
    that must point to a different speaker before CORRECT is emitted."""

    # ── Temporal evidence ────────────────────────────────────────────────────
    turn_transition_pause_sec: float = 0.40
    """Pauses ≥ this threshold indicate a likely speaker-turn boundary."""

    long_pause_sec: float = 1.20
    """Pauses ≥ this are treated as strong turn-change evidence."""

    continuity_window: int = 3
    """Number of adjacent phrases to inspect for local speaker continuity."""

    # ── Acoustic evidence ────────────────────────────────────────────────────
    strong_acoustic_sim_threshold: float = 0.75
    """Cosine similarity to centroid ≥ this → strong acoustic match."""

    weak_acoustic_sim_threshold: float = 0.45
    """Cosine similarity to centroid < this → weak / unreliable acoustic match."""

    centroid_separation_threshold: float = 0.15
    """Minimum cosine distance gap between assigned centroid and next-closest
    centroid for acoustic evidence to be considered reliable."""

    # ── Linguistic / dialogue rules ───────────────────────────────────────────
    question_answer_speaker_change_prior: float = 0.75
    """Prior probability of a speaker change after a question."""

    greeting_response_speaker_change_prior: float = 0.70
    """Prior probability of a speaker change after a greeting."""

    statement_acknowledgement_speaker_change_prior: float = 0.55
    """Prior probability of a speaker change when an acknowledgement follows a statement."""

    continuation_same_speaker_prior: float = 0.80
    """Prior probability of the SAME speaker continuing after a clear continuation phrase."""

    # ── Early-dialogue stabilization ─────────────────────────────────────────
    early_dialogue_window_sec: float = 5.0
    """Phrases starting before this time (seconds) are marked provisional."""

    provisional_reeval_confirm_threshold: float = 0.70
    """Confidence threshold for accepting a re-evaluated provisional label."""

    # ── Filler / affirmation words ────────────────────────────────────────────
    filler_words: List[str] = field(default_factory=lambda: [
        "oh", "ah", "uh", "um", "hmm", "hm", "mhm", "mm",
        "yeah", "yep", "yes", "no", "nope", "nah",
        "okay", "ok", "right", "sure", "exactly", "absolutely",
        "indeed", "fine", "great", "good", "wow", "really",
        "mine", "it's mine", "that's right", "thats right",
        "of course", "all right", "alright", "go ahead",
    ])

    question_markers: List[str] = field(default_factory=lambda: [
        "what", "who", "where", "when", "why", "how",
        "is", "are", "was", "were", "do", "does", "did",
        "can", "could", "would", "should", "will", "have",
        "has", "had", "may", "might", "shall",
    ])

    greeting_words: List[str] = field(default_factory=lambda: [
        "hello", "hi", "hey", "howdy", "greetings",
        "good morning", "good afternoon", "good evening",
        "nice to meet", "pleased to meet", "how are you",
    ])

    def __post_init__(self) -> None:
        """Validate weight sums and threshold ordering."""
        normal_sum = round(self.w_acoustic + self.w_temporal + self.w_linguistic, 6)
        if abs(normal_sum - 1.0) > 1e-4:
            raise ValueError(
                f"Normal fusion weights must sum to 1.0, got {normal_sum:.6f} "
                f"(w_acoustic={self.w_acoustic}, w_temporal={self.w_temporal}, "
                f"w_linguistic={self.w_linguistic})"
            )

        short_sum = round(
            self.short_utt_w_acoustic + self.short_utt_w_temporal + self.short_utt_w_linguistic,
            6,
        )
        if abs(short_sum - 1.0) > 1e-4:
            raise ValueError(
                f"Short-utterance fusion weights must sum to 1.0, got {short_sum:.6f}"
            )

        if not (0.0 <= self.uncertain_threshold < self.confirm_threshold <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 ≤ uncertain_threshold < confirm_threshold ≤ 1.0; "
                f"got uncertain={self.uncertain_threshold}, confirm={self.confirm_threshold}"
            )
