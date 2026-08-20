"""
CASA Engine — V3.2 Speaker Intelligence
Conversation-Aware Speaker Attribution layer.

This module sits AFTER SpeakerEmbeddingService.diarize_audio() and BEFORE
AudioWorker assembles AudioSegment objects.  It never replaces ECAPA; it only
validates or gently corrects the initial acoustic attribution.

Pipeline position:
  ECAPA diarization → diarized_segments
  → CASAEngine.apply()
  → enriched diarized_segments (speaker_label may be updated, confidence added)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from services.casa_config import CASAConfig
from utils.logger import logger


# ─────────────────────────────────────────────────────────────────────────────
# Linguistic phrase types
# ─────────────────────────────────────────────────────────────────────────────

class PhraseType:
    QUESTION      = "QUESTION"
    GREETING      = "GREETING"
    AFFIRMATION   = "AFFIRMATION"   # yes, no, okay, right, exactly …
    SHORT_FILLER  = "SHORT_FILLER"  # oh, ah, uh, mm, wow …
    CONTINUATION  = "CONTINUATION"  # "and", "so", "but", "also" leading
    STATEMENT     = "STATEMENT"     # default
    OTHER         = "OTHER"


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CASAResult:
    """Attribution decision for a single diarized phrase."""

    phrase_index: int
    proposed_speaker: str          # final label (may equal original)
    original_speaker: str          # ECAPA-assigned label before CASA
    confidence: float              # 0.0–1.0 fused confidence
    decision: str                  # "CONFIRM" | "CORRECT" | "UNCERTAIN"
    evidence: List[str]            # human-readable signal descriptions
    provisional: bool = False      # True for early-dialogue phrases (< window_sec)

    # ── Internal sub-scores (exposed for testing / diagnostics) ──────────────
    acoustic_score: float = 0.0
    temporal_score: float = 0.0
    linguistic_score: float = 0.0

    # ── Alternate speaker candidate (for CORRECT decisions) ──────────────────
    alternate_speaker: Optional[str] = None
    alternate_confidence: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Linguistic Classifier
# ─────────────────────────────────────────────────────────────────────────────

class LinguisticClassifier:
    """Classify a phrase text into a PhraseType using lightweight heuristics."""

    def __init__(self, config: CASAConfig) -> None:
        self._cfg = config
        self._fillers = {w.lower() for w in config.filler_words}
        self._q_markers = {w.lower() for w in config.question_markers}

        # Precompile greeting regex with word boundaries to prevent substring matching
        # (e.g. avoid matching "hi" inside "this" or "china")
        greeting_patterns = [re.escape(g.lower()) for g in config.greeting_words]
        self._greeting_re = re.compile(
            r"\b(" + "|".join(greeting_patterns) + r")\b",
            re.IGNORECASE,
        )

        # Continuation-start patterns
        self._continuation_re = re.compile(
            r"^\s*(and|so|but|also|then|additionally|furthermore|moreover|"
            r"however|well|i mean|you know|basically|actually|anyway)\b",
            re.IGNORECASE,
        )

        self._affirmation_phrases = {
            "yes", "no", "yeah", "yep", "nope", "nah", "okay",
            "ok", "right", "sure", "exactly", "absolutely",
            "of course", "certainly", "indeed", "fine", "correct",
            "that's right", "thats right", "it's mine", "its mine",
            "all right", "alright", "got it", "i see",
        }

    def classify(self, text: str, duration_sec: float, num_words: int) -> str:
        """Return a PhraseType constant for the given phrase."""
        if not text or not text.strip():
            return PhraseType.OTHER

        txt_lower = text.strip().lower()
        stripped = txt_lower.rstrip(".,!?;:")
        words = txt_lower.split()

        # ── Affirmation (yes/no/okay/right/that's right/sure + short) ─────────
        if stripped in self._affirmation_phrases or (
            len(words) <= 4 and all(w.rstrip(".,!?") in self._affirmation_phrases for w in words)
        ):
            return PhraseType.AFFIRMATION

        # ── Exact filler match (single short phrase) ──────────────────────────
        if stripped in self._fillers:
            return PhraseType.SHORT_FILLER

        # ── Short utterance that contains only filler words ───────────────────
        if len(words) <= 3:
            if all(w.rstrip(".,!?") in self._fillers for w in words):
                return PhraseType.SHORT_FILLER

        # ── Continuation (start of phrase) ────────────────────────────────────
        if self._continuation_re.match(text.strip()):
            return PhraseType.CONTINUATION

        # ── Question (ends with "?" or starts with interrogative word) ────────
        if text.rstrip().endswith("?"):
            return PhraseType.QUESTION
        first_word = words[0].rstrip(".,!?;:") if words else ""
        if first_word in self._q_markers:
            return PhraseType.QUESTION

        # ── Greeting (whole-word boundary check) ──────────────────────────────
        if self._greeting_re.search(text):
            return PhraseType.GREETING

        return PhraseType.STATEMENT


# ─────────────────────────────────────────────────────────────────────────────
# Acoustic Evidence Scorer
# ─────────────────────────────────────────────────────────────────────────────

class AcousticEvidenceScorer:
    """
    Derive an acoustic confidence score [0,1] for a phrase's speaker assignment.

    Uses:
    - embedding-to-centroid cosine similarity
    - centroid separation (gap to next-closest centroid)
    - local speaker continuity (matching neighbours)
    """

    def __init__(self, config: CASAConfig) -> None:
        self._cfg = config

    def score(
        self,
        phrase_idx: int,
        speaker_label: str,
        phrase_embeddings: Optional[List[Optional[np.ndarray]]],
        speaker_centroids: Optional[Dict[str, np.ndarray]],
        all_labels: List[str],
    ) -> Tuple[float, List[str], Optional[str], float]:
        """
        Returns:
            (score, evidence_list, best_alternate_speaker, alternate_sim)
        """
        evidence: List[str] = []
        cfg = self._cfg

        # ── No centroid data ─────────────────────────────────────────────────
        if not speaker_centroids:
            evidence.append("no centroid data: acoustic evidence neutral")
            return 0.65, evidence, None, 0.0

        centroid = speaker_centroids.get(speaker_label)
        if centroid is None:
            evidence.append("centroid missing for assigned speaker")
            return 0.50, evidence, None, 0.0

        # ── Phrase embedding similarity to assigned centroid ──────────────────
        emb: Optional[np.ndarray] = None
        if phrase_embeddings and phrase_idx < len(phrase_embeddings):
            emb = phrase_embeddings[phrase_idx]

        sim_to_assigned: float = 0.60  # neutral prior when no embedding
        if emb is not None and np.linalg.norm(emb) > 1e-6:
            emb_norm = emb / np.linalg.norm(emb)
            sim_to_assigned = float(np.dot(emb_norm, centroid))

        # ── Single speaker centroid case ──────────────────────────────────────
        if len(speaker_centroids) == 1:
            if sim_to_assigned >= cfg.strong_acoustic_sim_threshold:
                acoustic_score = 0.90
                evidence.append(f"strong acoustic similarity to {speaker_label} (sim={sim_to_assigned:.2f})")
            elif sim_to_assigned >= cfg.weak_acoustic_sim_threshold:
                acoustic_score = 0.60 + 0.25 * (
                    (sim_to_assigned - cfg.weak_acoustic_sim_threshold)
                    / (cfg.strong_acoustic_sim_threshold - cfg.weak_acoustic_sim_threshold + 1e-9)
                )
                evidence.append(f"moderate acoustic similarity to {speaker_label} (sim={sim_to_assigned:.2f})")
            else:
                acoustic_score = 0.35
                evidence.append(f"weak acoustic similarity to {speaker_label} (sim={sim_to_assigned:.2f})")
            return acoustic_score, evidence, None, 0.0

        # ── Similarity to all other centroids → find best alternate ──────────
        best_alt_label: Optional[str] = None
        best_alt_sim: float = -1.0
        for lbl, c in speaker_centroids.items():
            if lbl == speaker_label:
                continue
            if emb is not None and np.linalg.norm(emb) > 1e-6:
                alt_sim = float(np.dot(emb_norm, c))  # type: ignore[name-defined]
            else:
                alt_sim = 0.0
            if alt_sim > best_alt_sim:
                best_alt_sim = alt_sim
                best_alt_label = lbl

        # Centroid separation
        separation = sim_to_assigned - best_alt_sim

        # ── Build acoustic score ──────────────────────────────────────────────
        if sim_to_assigned >= cfg.strong_acoustic_sim_threshold and separation >= cfg.centroid_separation_threshold:
            acoustic_score = 0.90
            evidence.append(f"strong acoustic similarity to {speaker_label} (sim={sim_to_assigned:.2f}, sep={separation:.2f})")
        elif sim_to_assigned >= cfg.weak_acoustic_sim_threshold:
            acoustic_score = 0.60 + 0.25 * (
                (sim_to_assigned - cfg.weak_acoustic_sim_threshold)
                / (cfg.strong_acoustic_sim_threshold - cfg.weak_acoustic_sim_threshold + 1e-9)
            )
            evidence.append(f"moderate acoustic similarity to {speaker_label} (sim={sim_to_assigned:.2f})")
        else:
            acoustic_score = 0.35
            evidence.append(f"weak acoustic similarity to {speaker_label} (sim={sim_to_assigned:.2f})")

        # ── Local continuity ──────────────────────────────────────────────────
        window = cfg.continuity_window
        neighbors = (
            all_labels[max(0, phrase_idx - window): phrase_idx]
            + all_labels[phrase_idx + 1: phrase_idx + 1 + window]
        )
        valid_neighbors = [l for l in neighbors if l is not None]
        if valid_neighbors:
            same_count = sum(1 for l in valid_neighbors if l == speaker_label)
            continuity_ratio = same_count / len(valid_neighbors)
            if continuity_ratio >= 0.6:
                acoustic_score = min(1.0, acoustic_score + 0.05)
                evidence.append(f"local continuity: {same_count}/{len(valid_neighbors)} neighbors match")
            elif continuity_ratio == 0.0:
                acoustic_score = max(0.0, acoustic_score - 0.05)
                evidence.append("local discontinuity: no neighbors match assigned speaker")

        return acoustic_score, evidence, best_alt_label, best_alt_sim


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Evidence Scorer
# ─────────────────────────────────────────────────────────────────────────────

class TemporalEvidenceScorer:
    """Derive temporal evidence from adjacent phrase boundaries."""

    def __init__(self, config: CASAConfig) -> None:
        self._cfg = config

    def score(
        self,
        phrase_idx: int,
        speaker_label: str,
        segments: List[dict],
    ) -> Tuple[float, List[str], bool]:
        """
        Returns:
            (score, evidence_list, is_turn_transition_expected)
        """
        cfg = self._cfg
        evidence: List[str] = []

        n = len(segments)
        seg = segments[phrase_idx]

        # ── Temporal context ──────────────────────────────────────────────────
        prev_seg = segments[phrase_idx - 1] if phrase_idx > 0 else None
        next_seg = segments[phrase_idx + 1] if phrase_idx < n - 1 else None

        pause_before = (
            seg["start_sec"] - prev_seg["end_sec"] if prev_seg else None
        )
        pause_after = (
            next_seg["start_sec"] - seg["end_sec"] if next_seg else None
        )
        prev_speaker = prev_seg.get("speaker_label") if prev_seg else None
        next_speaker = next_seg.get("speaker_label") if next_seg else None

        # ── Turn transition evidence ──────────────────────────────────────────
        turn_expected = False
        temporal_score = 0.60  # neutral baseline

        if prev_speaker is not None:
            if prev_speaker == speaker_label:
                # Same speaker as before → continuity boost
                temporal_score += 0.15
                evidence.append(f"speaker continuity from previous turn ({prev_speaker})")
                if pause_before is not None and pause_before < cfg.turn_transition_pause_sec:
                    temporal_score += 0.05
                    evidence.append(f"short pause before ({pause_before:.2f}s) supports continuity")
            else:
                # Different speaker — expected turn transition
                turn_expected = True
                temporal_score += 0.05
                evidence.append(f"expected turn transition from {prev_speaker}")
                if pause_before is not None and pause_before >= cfg.turn_transition_pause_sec:
                    temporal_score += 0.10
                    evidence.append(f"turn-boundary pause before ({pause_before:.2f}s ≥ {cfg.turn_transition_pause_sec}s)")
                elif pause_before is not None and pause_before >= cfg.long_pause_sec:
                    temporal_score += 0.15
                    evidence.append(f"long pause before ({pause_before:.2f}s) strongly suggests turn change")

        # ── Next speaker check ────────────────────────────────────────────────
        if next_speaker is not None and next_speaker == speaker_label:
            temporal_score = min(1.0, temporal_score + 0.05)
            evidence.append(f"forward continuity: next phrase also {speaker_label}")

        # ── Duration of current phrase ────────────────────────────────────────
        duration = seg.get("duration_sec", seg["end_sec"] - seg["start_sec"])
        if duration < cfg.short_utt_max_duration_sec:
            # Short phrases: less certain, soften temporal confidence
            temporal_score *= 0.90
            evidence.append(f"short phrase duration ({duration:.2f}s) reduces temporal certainty")

        temporal_score = max(0.0, min(1.0, temporal_score))
        return temporal_score, evidence, turn_expected


# ─────────────────────────────────────────────────────────────────────────────
# Linguistic Evidence Scorer
# ─────────────────────────────────────────────────────────────────────────────

class LinguisticEvidenceScorer:
    """Apply dialogue-pattern rules to produce a linguistic evidence score."""

    def __init__(self, config: CASAConfig) -> None:
        self._cfg = config
        self._classifier = LinguisticClassifier(config)

    def score(
        self,
        phrase_idx: int,
        speaker_label: str,
        segments: List[dict],
    ) -> Tuple[float, List[str], bool]:
        """
        Returns:
            (score, evidence_list, dialogue_suggests_speaker_change)
        """
        cfg = self._cfg
        evidence: List[str] = []

        seg = segments[phrase_idx]
        text = seg.get("text", "")
        words = seg.get("words", [])
        duration = seg.get("duration_sec", seg["end_sec"] - seg["start_sec"])
        num_words = len(words) if words else len(text.split())

        curr_type = self._classifier.classify(text, duration, num_words)

        # Previous phrase context
        prev_seg = segments[phrase_idx - 1] if phrase_idx > 0 else None
        prev_type = PhraseType.OTHER
        prev_speaker = None
        if prev_seg:
            prev_text = prev_seg.get("text", "")
            prev_words = prev_seg.get("words", [])
            prev_dur = prev_seg.get("duration_sec", prev_seg["end_sec"] - prev_seg["start_sec"])
            prev_num_words = len(prev_words) if prev_words else len(prev_text.split())
            prev_type = self._classifier.classify(prev_text, prev_dur, prev_num_words)
            prev_speaker = prev_seg.get("speaker_label")

        dialogue_change_suggested = False
        linguistic_score = 0.55  # neutral baseline

        # ── No previous context → neutral ─────────────────────────────────────
        if prev_seg is None:
            evidence.append("no prior context (first phrase)")
            return linguistic_score, evidence, False

        # ── Apply dialogue rules ──────────────────────────────────────────────
        if prev_type == PhraseType.QUESTION:
            if curr_type in (PhraseType.STATEMENT, PhraseType.AFFIRMATION, PhraseType.GREETING):
                if prev_speaker == speaker_label:
                    # Same speaker answered their own question → suspicious
                    dialogue_change_suggested = True
                    linguistic_score = 0.30
                    evidence.append(
                        f"dialogue pattern: question→answer expected turn change "
                        f"(same speaker {speaker_label} — suspicious)"
                    )
                else:
                    # Different speaker answers → expected, boost confidence
                    linguistic_score = 0.80
                    evidence.append(
                        "dialogue pattern: expected question→response from different speaker"
                    )
            elif curr_type == PhraseType.SHORT_FILLER:
                if prev_speaker == speaker_label:
                    dialogue_change_suggested = True
                    linguistic_score = 0.40
                    evidence.append("dialogue pattern: question followed by filler — likely different speaker")
                else:
                    linguistic_score = 0.65
                    evidence.append("dialogue pattern: question→filler response from different speaker")

        elif prev_type == PhraseType.GREETING:
            if curr_type in (PhraseType.GREETING, PhraseType.STATEMENT, PhraseType.AFFIRMATION):
                if prev_speaker == speaker_label:
                    dialogue_change_suggested = True
                    linguistic_score = 0.35
                    evidence.append("dialogue pattern: reciprocal greeting expected from different speaker")
                else:
                    linguistic_score = 0.80
                    evidence.append("dialogue pattern: expected greeting reciprocation from different speaker")

        elif prev_type == PhraseType.STATEMENT:
            if curr_type == PhraseType.AFFIRMATION:
                if prev_speaker != speaker_label:
                    linguistic_score = 0.70
                    evidence.append("dialogue pattern: acknowledgement of statement from different speaker")
                else:
                    # Same speaker self-acknowledging → slight suspicion
                    linguistic_score = 0.45
                    evidence.append("dialogue pattern: statement→acknowledgement from same speaker (slight suspicion)")

            elif curr_type == PhraseType.CONTINUATION:
                # Strong signal for same speaker
                if prev_speaker == speaker_label:
                    linguistic_score = 0.82
                    evidence.append("dialogue pattern: continuation from same speaker (expected)")
                else:
                    linguistic_score = 0.45
                    evidence.append("dialogue pattern: continuation from different speaker (less expected)")

        elif prev_type == PhraseType.SHORT_FILLER:
            # After a filler we have less certainty
            linguistic_score = 0.52
            evidence.append("dialogue pattern: following filler — reduced certainty")

        elif prev_type == PhraseType.AFFIRMATION:
            # After ack/yes/no, either speaker could speak
            linguistic_score = 0.55
            evidence.append("dialogue pattern: following affirmation — neutral")

        # ── Short-filler current phrase ────────────────────────────────────────
        if curr_type == PhraseType.SHORT_FILLER:
            evidence.append(f"current phrase is SHORT_FILLER ({repr(text.strip())}): linguistic weight reduced")
            # Scale score toward neutral — the signal is weak
            linguistic_score = 0.50 + (linguistic_score - 0.50) * 0.5

        # ── Classify note ─────────────────────────────────────────────────────
        evidence.append(f"phrase type: {curr_type} | prev type: {prev_type}")

        linguistic_score = max(0.0, min(1.0, linguistic_score))
        return linguistic_score, evidence, dialogue_change_suggested


# ─────────────────────────────────────────────────────────────────────────────
# CASA Engine
# ─────────────────────────────────────────────────────────────────────────────

class CASAEngine:
    """
    Conversation-Aware Speaker Attribution Engine.

    Takes the output of SpeakerEmbeddingService.diarize_audio() and produces
    a CASAResult for every phrase, deciding whether to CONFIRM, CORRECT, or
    mark as UNCERTAIN.

    CASA is a *correction/validation layer*, not a replacement for ECAPA.
    If evidence is weak, it preserves the original acoustic label.
    """

    def __init__(self, config: Optional[CASAConfig] = None) -> None:
        self._cfg = config or CASAConfig()
        self._acoustic_scorer = AcousticEvidenceScorer(self._cfg)
        self._temporal_scorer = TemporalEvidenceScorer(self._cfg)
        self._linguistic_scorer = LinguisticEvidenceScorer(self._cfg)

    # ── Public API ────────────────────────────────────────────────────────────

    def apply(
        self,
        diarized_segments: List[dict],
        phrase_embeddings: Optional[List[Optional[np.ndarray]]] = None,
        speaker_centroids: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[CASAResult]:
        """
        Run the CASA pass over all diarized phrases.

        Args:
            diarized_segments: List of phrase dicts from diarize_audio() —
                each must have keys: start_sec, end_sec, text, words, speaker_label.
            phrase_embeddings: Per-phrase ECAPA embedding vectors (may be None).
            speaker_centroids: Cluster centroid embeddings keyed by speaker label
                (e.g. {"Speaker 1": np.ndarray, "Speaker 2": np.ndarray}).

        Returns:
            List[CASAResult] — one entry per phrase, in order.
        """
        if not self._cfg.enable_casa or not diarized_segments:
            # Pass-through: CONFIRM everything with neutral confidence
            return self._passthrough(diarized_segments)

        n = len(diarized_segments)
        all_labels = [seg.get("speaker_label") for seg in diarized_segments]

        results: List[CASAResult] = []

        for i, seg in enumerate(diarized_segments):
            result = self._evaluate_phrase(
                phrase_idx=i,
                segments=diarized_segments,
                all_labels=all_labels,
                phrase_embeddings=phrase_embeddings,
                speaker_centroids=speaker_centroids,
            )
            results.append(result)

        # ── Early-dialogue stabilization pass ────────────────────────────────
        if self._cfg.enable_early_dialogue_stabilization:
            results = self._stabilize_early_dialogue(
                results=results,
                segments=diarized_segments,
                speaker_centroids=speaker_centroids,
                phrase_embeddings=phrase_embeddings,
            )

        logger.debug(
            f"[CASA] Processed {n} phrases: "
            f"CONFIRM={sum(1 for r in results if r.decision == 'CONFIRM')} | "
            f"CORRECT={sum(1 for r in results if r.decision == 'CORRECT')} | "
            f"UNCERTAIN={sum(1 for r in results if r.decision == 'UNCERTAIN')}"
        )
        return results

    # ── Per-phrase evaluation ─────────────────────────────────────────────────

    def _evaluate_phrase(
        self,
        phrase_idx: int,
        segments: List[dict],
        all_labels: List[str],
        phrase_embeddings: Optional[List[Optional[np.ndarray]]],
        speaker_centroids: Optional[Dict[str, np.ndarray]],
    ) -> CASAResult:
        seg = segments[phrase_idx]
        original_speaker = seg.get("speaker_label") or "Speaker 1"
        cfg = self._cfg

        # ── Is this a short utterance? ────────────────────────────────────────
        words = seg.get("words", [])
        text = seg.get("text", "")
        num_words = len(words) if words else len(text.split())
        duration = seg.get("duration_sec", seg["end_sec"] - seg["start_sec"])
        is_short = (
            num_words <= cfg.short_utt_max_words
            or duration <= cfg.short_utt_max_duration_sec
        )

        # ── Is this early dialogue? ───────────────────────────────────────────
        is_provisional = seg["start_sec"] < cfg.early_dialogue_window_sec

        # ── Collect evidence ──────────────────────────────────────────────────
        all_evidence: List[str] = []

        acoustic_score, acoustic_ev, best_alt_speaker, best_alt_sim = (
            self._acoustic_scorer.score(
                phrase_idx=phrase_idx,
                speaker_label=original_speaker,
                phrase_embeddings=phrase_embeddings,
                speaker_centroids=speaker_centroids,
                all_labels=all_labels,
            )
        )
        all_evidence.extend(acoustic_ev)

        temporal_score, temporal_ev, turn_expected = (
            self._temporal_scorer.score(
                phrase_idx=phrase_idx,
                speaker_label=original_speaker,
                segments=segments,
            )
        )
        all_evidence.extend(temporal_ev)

        linguistic_score, linguistic_ev, dialogue_suggests_change = (
            self._linguistic_scorer.score(
                phrase_idx=phrase_idx,
                speaker_label=original_speaker,
                segments=segments,
            )
        )
        all_evidence.extend(linguistic_ev)

        # ── Fuse scores with appropriate weights ──────────────────────────────
        if is_short:
            wa, wt, wl = cfg.short_utt_w_acoustic, cfg.short_utt_w_temporal, cfg.short_utt_w_linguistic
        else:
            wa, wt, wl = cfg.w_acoustic, cfg.w_temporal, cfg.w_linguistic

        fused_confidence = wa * acoustic_score + wt * temporal_score + wl * linguistic_score
        fused_confidence = max(0.0, min(1.0, round(fused_confidence, 4)))

        # ── Principled Multi-Candidate Evaluation ─────────────────────────────
        # For every discovered speaker, compute candidate score
        candidate_scores: Dict[str, float] = {}
        if speaker_centroids and len(speaker_centroids) > 1:
            emb = None
            if phrase_embeddings and phrase_idx < len(phrase_embeddings):
                emb = phrase_embeddings[phrase_idx]
            emb_norm = emb / np.linalg.norm(emb) if (emb is not None and np.linalg.norm(emb) > 1e-6) else None

            prev_seg = segments[phrase_idx - 1] if phrase_idx > 0 else None
            prev_speaker = prev_seg.get("speaker_label") if prev_seg else None

            for spk_lbl, c in speaker_centroids.items():
                if emb_norm is not None:
                    c_sim = float(np.dot(emb_norm, c))
                    c_ac = 0.50 + 0.40 * max(-0.2, min(1.0, c_sim))
                else:
                    c_ac = 0.50

                # Candidate temporal continuity
                c_temp = 0.60
                if prev_speaker:
                    if prev_speaker == spk_lbl:
                        c_temp += 0.15
                    else:
                        c_temp += 0.05

                # Candidate linguistic alignment
                c_ling = linguistic_score if spk_lbl == original_speaker else (1.0 - linguistic_score if dialogue_suggests_change else 0.55)

                c_score = wa * c_ac + wt * c_temp + wl * c_ling
                candidate_scores[spk_lbl] = max(0.0, min(1.0, round(c_score, 4)))

        # ── Make attribution decision ─────────────────────────────────────────
        decision, proposed_speaker, alternate_conf = self._decide(
            original_speaker=original_speaker,
            best_alt_speaker=best_alt_speaker,
            best_alt_sim=best_alt_sim,
            fused_confidence=fused_confidence,
            turn_expected=turn_expected,
            dialogue_suggests_change=dialogue_suggests_change,
            speaker_centroids=speaker_centroids,
            candidate_scores=candidate_scores,
            all_evidence=all_evidence,
        )

        return CASAResult(
            phrase_index=phrase_idx,
            proposed_speaker=proposed_speaker,
            original_speaker=original_speaker,
            confidence=fused_confidence,
            decision=decision,
            evidence=all_evidence,
            provisional=is_provisional,
            acoustic_score=round(acoustic_score, 4),
            temporal_score=round(temporal_score, 4),
            linguistic_score=round(linguistic_score, 4),
            alternate_speaker=best_alt_speaker if (decision == "CORRECT" or best_alt_speaker is not None) else None,
            alternate_confidence=round(alternate_conf, 4) if alternate_conf else 0.0,
        )

    def _decide(
        self,
        original_speaker: str,
        best_alt_speaker: Optional[str],
        best_alt_sim: float,
        fused_confidence: float,
        turn_expected: bool,
        dialogue_suggests_change: bool,
        speaker_centroids: Optional[Dict[str, np.ndarray]],
        candidate_scores: Dict[str, float],
        all_evidence: List[str],
    ) -> Tuple[str, str, float]:
        """
        Returns: (decision, proposed_speaker, alternate_confidence)
        CONFIRM  → proposed_speaker == original_speaker
        CORRECT  → proposed_speaker == best_alt_speaker (strict guard)
        UNCERTAIN → proposed_speaker == original_speaker (keep, lower conf)
        """
        cfg = self._cfg

        # ── CONFIRM: high confidence in original ─────────────────────────────
        if fused_confidence >= cfg.confirm_threshold:
            all_evidence.append(f"decision: CONFIRM (confidence={fused_confidence:.2f} ≥ {cfg.confirm_threshold})")
            return "CONFIRM", original_speaker, 0.0

        # ── Candidate Ranking Check ──────────────────────────────────────────
        if (
            fused_confidence < cfg.uncertain_threshold
            and best_alt_speaker is not None
            and (dialogue_suggests_change or turn_expected)
        ):
            # Compute alternate speaker confidence estimate
            if candidate_scores and best_alt_speaker in candidate_scores:
                alt_confidence_estimate = candidate_scores[best_alt_speaker]
            else:
                alt_centroid_sim = best_alt_sim if speaker_centroids else 0.0
                alt_confidence_estimate = 0.50 + 0.40 * max(0.0, alt_centroid_sim)

            confidence_delta = alt_confidence_estimate - fused_confidence
            independent_signals = int(turn_expected) + int(dialogue_suggests_change)

            if (
                confidence_delta >= cfg.min_correction_confidence_delta
                and independent_signals >= cfg.min_correction_signals
            ):
                all_evidence.append(
                    f"decision: CORRECT {original_speaker}→{best_alt_speaker} "
                    f"(delta={confidence_delta:.2f}, signals={independent_signals})"
                )
                return "CORRECT", best_alt_speaker, alt_confidence_estimate

        # Estimate alternate confidence for UNCERTAIN if candidate exists
        alt_conf_val = 0.0
        if best_alt_speaker is not None and candidate_scores and best_alt_speaker in candidate_scores:
            alt_conf_val = candidate_scores[best_alt_speaker]

        # ── UNCERTAIN: keep original label, expose lower confidence ──────────
        all_evidence.append(
            f"decision: UNCERTAIN (confidence={fused_confidence:.2f}; original label preserved)"
        )
        return "UNCERTAIN", original_speaker, alt_conf_val

    # ── Early-dialogue stabilization ──────────────────────────────────────────

    def _stabilize_early_dialogue(
        self,
        results: List[CASAResult],
        segments: List[dict],
        speaker_centroids: Optional[Dict[str, np.ndarray]],
        phrase_embeddings: Optional[List[Optional[np.ndarray]]],
    ) -> List[CASAResult]:
        """
        Re-evaluate provisional (early-dialogue) phrases that are UNCERTAIN using
        global centroid evidence accumulated from all non-provisional phrases.
        """
        cfg = self._cfg

        # Gather global centroid from confident non-provisional phrases
        # (only used when speaker_centroids is already available)
        provisional_uncertain_indices = [
            i for i, r in enumerate(results)
            if r.provisional and r.decision == "UNCERTAIN"
        ]

        if not provisional_uncertain_indices:
            return results

        logger.debug(
            f"[CASA] Early-dialogue stabilization: re-evaluating "
            f"{len(provisional_uncertain_indices)} provisional-uncertain phrase(s)"
        )

        for i in provisional_uncertain_indices:
            seg = segments[i]
            original_speaker = seg.get("speaker_label") or "Speaker 1"

            # Build updated label context from all results so far
            updated_labels = [
                r.proposed_speaker if r.decision in ("CONFIRM", "CORRECT") else seg.get("speaker_label")
                for r, seg in zip(results, segments)
            ]

            # Re-score acoustic with updated centroid info
            acoustic_score, acoustic_ev, best_alt_speaker, best_alt_sim = (
                self._acoustic_scorer.score(
                    phrase_idx=i,
                    speaker_label=original_speaker,
                    phrase_embeddings=phrase_embeddings,
                    speaker_centroids=speaker_centroids,
                    all_labels=updated_labels,
                )
            )

            # Temporal and linguistic scores unchanged (same phrase context)
            temporal_score = results[i].temporal_score
            linguistic_score = results[i].linguistic_score

            wa, wt, wl = cfg.w_acoustic, cfg.w_temporal, cfg.w_linguistic
            new_confidence = wa * acoustic_score + wt * temporal_score + wl * linguistic_score
            new_confidence = max(0.0, min(1.0, round(new_confidence, 4)))

            if new_confidence >= cfg.provisional_reeval_confirm_threshold:
                results[i].confidence = new_confidence
                results[i].decision = "CONFIRM"
                results[i].evidence.append(
                    f"early-dialogue stabilization: re-evaluated confidence={new_confidence:.2f} → CONFIRM"
                )
            else:
                results[i].confidence = new_confidence
                results[i].evidence.append(
                    f"early-dialogue stabilization: re-evaluated confidence={new_confidence:.2f} (still UNCERTAIN)"
                )

        return results

    # ── Pass-through helper ───────────────────────────────────────────────────

    @staticmethod
    def _passthrough(segments: List[dict]) -> List[CASAResult]:
        return [
            CASAResult(
                phrase_index=i,
                proposed_speaker=seg.get("speaker_label") or "Speaker 1",
                original_speaker=seg.get("speaker_label") or "Speaker 1",
                confidence=0.70,
                decision="CONFIRM",
                evidence=["CASA disabled: acoustic label preserved"],
                provisional=False,
                acoustic_score=0.70,
                temporal_score=0.70,
                linguistic_score=0.70,
            )
            for i, seg in enumerate(segments)
        ]
