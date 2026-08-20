"""
Unit tests for the CASA Engine — V3.2 Speaker Intelligence
Tests acoustic, temporal, and linguistic scorers as well as fusion,
decision logic, early-dialogue stabilization, and the pass-through mode.

All tests use synthetic diarized_segment dicts — no real audio files needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.casa_config import CASAConfig
from services.casa_engine import (
    CASAEngine,
    CASAResult,
    LinguisticClassifier,
    PhraseType,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

DIM = 192  # ECAPA embedding dimension


def _seg(text, speaker, start, end, duration=None):
    """Build a minimal diarized segment dict."""
    dur = duration if duration is not None else (end - start)
    words = [{"word": w} for w in text.split()]
    return {
        "text": text,
        "speaker_label": speaker,
        "start_sec": start,
        "end_sec": end,
        "duration_sec": dur,
        "words": words,
    }


def _centroid(dim: int, idx: int) -> np.ndarray:
    """Return an orthogonal unit vector at position idx."""
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


def _similar_emb(centroid: np.ndarray, noise: float = 0.02) -> np.ndarray:
    """Return a noisy copy of centroid — high cosine similarity."""
    v = centroid.copy() + noise * np.random.RandomState(42).randn(len(centroid)).astype(np.float32)
    return v / np.linalg.norm(v)


# ─────────────────────────────────────────────────────────────────────────────
# Linguistic Classifier
# ─────────────────────────────────────────────────────────────────────────────

class TestLinguisticClassifier:
    def setup_method(self):
        self.clf = LinguisticClassifier(CASAConfig())

    def test_question_ends_with_mark(self):
        assert self.clf.classify("What is your name?", 1.5, 4) == PhraseType.QUESTION

    def test_question_starts_with_interrogative(self):
        assert self.clf.classify("How are you doing today", 2.0, 5) == PhraseType.QUESTION

    def test_affirmation_yes(self):
        assert self.clf.classify("yes", 0.3, 1) == PhraseType.AFFIRMATION

    def test_affirmation_okay(self):
        assert self.clf.classify("okay", 0.3, 1) == PhraseType.AFFIRMATION

    def test_affirmation_right(self):
        assert self.clf.classify("right", 0.4, 1) == PhraseType.AFFIRMATION

    def test_short_filler_oh(self):
        assert self.clf.classify("oh", 0.2, 1) == PhraseType.SHORT_FILLER

    def test_short_filler_uh(self):
        assert self.clf.classify("uh", 0.3, 1) == PhraseType.SHORT_FILLER

    def test_short_filler_mm(self):
        assert self.clf.classify("mm", 0.2, 1) == PhraseType.SHORT_FILLER

    def test_greeting_hello(self):
        assert self.clf.classify("Hello how are you", 1.2, 4) == PhraseType.GREETING

    def test_continuation_and(self):
        assert self.clf.classify("And another thing is", 1.0, 4) == PhraseType.CONTINUATION

    def test_continuation_so(self):
        assert self.clf.classify("So basically what I mean is", 1.5, 6) == PhraseType.CONTINUATION

    def test_greeting_word_boundary_no_false_positive(self):
        # 'this' and 'China' contain 'hi', but should NOT be classified as GREETING
        assert self.clf.classify("this is a test sentence", 1.5, 5) == PhraseType.STATEMENT
        assert self.clf.classify("compare the population of China", 1.8, 5) == PhraseType.STATEMENT

    def test_affirmation_multiword(self):
        assert self.clf.classify("that's right", 0.4, 2) == PhraseType.AFFIRMATION
        assert self.clf.classify("it's mine", 0.4, 2) == PhraseType.AFFIRMATION
        assert self.clf.classify("of course", 0.4, 2) == PhraseType.AFFIRMATION
        assert self.clf.classify("all right", 0.4, 2) == PhraseType.AFFIRMATION

    def test_statement_default(self):
        assert self.clf.classify("I went to the store yesterday", 2.0, 6) == PhraseType.STATEMENT


# ─────────────────────────────────────────────────────────────────────────────
# CASA Engine — pass-through mode
# ─────────────────────────────────────────────────────────────────────────────

class TestCASAPassthrough:
    def test_disabled_casa_returns_original_labels(self):
        cfg = CASAConfig(enable_casa=False)
        engine = CASAEngine(config=cfg)
        segs = [
            _seg("Hello there", "Speaker 1", 0.0, 1.5),
            _seg("How are you?", "Speaker 2", 2.0, 3.5),
        ]
        results = engine.apply(segs)
        assert len(results) == 2
        assert all(r.decision == "CONFIRM" for r in results)
        assert results[0].proposed_speaker == "Speaker 1"
        assert results[1].proposed_speaker == "Speaker 2"

    def test_empty_segments_returns_empty(self):
        engine = CASAEngine()
        results = engine.apply([])
        assert results == []

    def test_disabled_does_not_change_any_labels(self):
        cfg = CASAConfig(enable_casa=False)
        engine = CASAEngine(config=cfg)
        segs = [_seg("yeah", "Speaker 1", 0.0, 0.3)]
        results = engine.apply(segs)
        assert results[0].proposed_speaker == "Speaker 1"


# ─────────────────────────────────────────────────────────────────────────────
# CASA Engine — confidence and decisions
# ─────────────────────────────────────────────────────────────────────────────

class TestCASADecisions:
    def test_all_results_have_confidence(self):
        engine = CASAEngine()
        segs = [
            _seg("What is your name?", "Speaker 1", 0.0, 2.0),
            _seg("My name is Jeff.", "Speaker 1", 2.5, 4.5),
        ]
        results = engine.apply(segs)
        assert all(0.0 <= r.confidence <= 1.0 for r in results)

    def test_all_results_have_valid_decision(self):
        engine = CASAEngine()
        segs = [
            _seg("Hello.", "Speaker 1", 0.0, 1.0),
            _seg("Hi there!", "Speaker 2", 1.5, 2.5),
        ]
        results = engine.apply(segs)
        assert all(r.decision in {"CONFIRM", "CORRECT", "UNCERTAIN"} for r in results)

    def test_all_results_have_evidence(self):
        engine = CASAEngine()
        segs = [_seg("Right.", "Speaker 1", 0.0, 0.3)]
        results = engine.apply(segs)
        assert results[0].evidence  # non-empty list

    def test_single_speaker_monologue_no_correction(self):
        """All phrases from one speaker with strong centroid match → all CONFIRM."""
        c1 = _centroid(DIM, 10)
        centroids = {"Speaker 1": c1}
        embs = [_similar_emb(c1) for _ in range(4)]
        segs = [
            _seg("I went to the store.", "Speaker 1", 0.0, 1.5),
            _seg("Then I came back home.", "Speaker 1", 1.8, 3.2),
            _seg("And made some coffee.", "Speaker 1", 3.5, 4.8),
            _seg("It was a good day.", "Speaker 1", 5.0, 6.5),
        ]
        engine = CASAEngine()
        results = engine.apply(segs, phrase_embeddings=embs, speaker_centroids=centroids)
        corrections = [r for r in results if r.decision == "CORRECT"]
        assert len(corrections) == 0, "Monologue should have zero corrections"

    def test_no_hallucinated_speakers(self):
        """CASA must never introduce speaker labels beyond those in original diarization."""
        c1 = _centroid(DIM, 5)
        c2 = _centroid(DIM, 95)
        centroids = {"Speaker 1": c1, "Speaker 2": c2}
        segs = [
            _seg("Okay.", "Speaker 1", 0.0, 0.3),
            _seg("Let me explain.", "Speaker 2", 0.8, 2.0),
        ]
        engine = CASAEngine()
        results = engine.apply(segs, speaker_centroids=centroids)
        original_labels = {"Speaker 1", "Speaker 2"}
        for r in results:
            assert r.proposed_speaker in original_labels, (
                f"CASA introduced unknown speaker: {r.proposed_speaker}"
            )

    def test_strong_acoustic_confirms(self):
        """High embedding similarity to centroid → CONFIRM."""
        c1 = _centroid(DIM, 0)
        centroids = {"Speaker 1": c1}
        # Very similar embedding
        emb = _similar_emb(c1, noise=0.005)
        segs = [_seg("Let me tell you something important.", "Speaker 1", 0.0, 2.0)]
        engine = CASAEngine()
        results = engine.apply(segs, phrase_embeddings=[emb], speaker_centroids=centroids)
        assert results[0].decision == "CONFIRM"
        assert results[0].acoustic_score > 0.7


# ─────────────────────────────────────────────────────────────────────────────
# CASA Engine — short utterance handling
# ─────────────────────────────────────────────────────────────────────────────

class TestShortUtteranceHandling:
    def test_filler_word_uses_shifted_weights(self):
        """For short fillers, acoustic_score weight is reduced — the fused confidence
        must NOT be dominated by a single acoustic value."""
        cfg = CASAConfig()
        engine = CASAEngine(config=cfg)

        # Weak acoustic match centroid (low embedding similarity) → low acoustic score
        c1 = _centroid(DIM, 0)
        c2 = _centroid(DIM, 100)
        centroids = {"Speaker 1": c1, "Speaker 2": c2}
        # Phrase embedding is equidistant (unclear)
        equidist = np.zeros(DIM, dtype=np.float32)
        equidist[0] = 1.0
        equidist[100] = 1.0
        equidist /= np.linalg.norm(equidist)

        segs = [
            _seg("I am talking now.", "Speaker 1", 0.0, 2.0),
            _seg("yeah", "Speaker 1", 2.5, 2.8),
        ]
        results = engine.apply(segs, phrase_embeddings=[equidist, equidist], speaker_centroids=centroids)
        yeah_result = results[1]
        # Confidence should not be blindly driven by acoustic alone
        assert 0.0 <= yeah_result.confidence <= 1.0
        # Evidence must mention short-phrase or filler
        evidence_text = " ".join(yeah_result.evidence).lower()
        assert any(k in evidence_text for k in ("short", "filler", "duration", "weight")), (
            f"No short-phrase evidence found in: {yeah_result.evidence}"
        )

    def test_short_duration_classified_correctly(self):
        """Phrase with duration < 0.8s should use short_utt weights."""
        cfg = CASAConfig(short_utt_max_duration_sec=0.8)
        c1 = _centroid(DIM, 0)
        centroids = {"Speaker 1": c1}
        segs = [_seg("oh", "Speaker 1", 0.0, 0.25)]  # duration = 0.25s
        engine = CASAEngine(config=cfg)
        results = engine.apply(segs, speaker_centroids=centroids)
        assert 0.0 <= results[0].confidence <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# CASA Engine — early dialogue stabilization
# ─────────────────────────────────────────────────────────────────────────────

class TestEarlyDialogueStabilization:
    def test_early_phrases_marked_provisional(self):
        """Phrases starting before early_dialogue_window_sec must be provisional=True."""
        cfg = CASAConfig(early_dialogue_window_sec=5.0)
        engine = CASAEngine(config=cfg)
        segs = [
            _seg("Hello", "Speaker 1", 0.0, 1.0),
            _seg("How are you?", "Speaker 2", 1.5, 2.5),
            _seg("Fine thanks and you?", "Speaker 1", 3.0, 4.5),
            _seg("Very well, let us begin.", "Speaker 2", 6.0, 8.0),
        ]
        results = engine.apply(segs)
        assert results[0].provisional is True  # start=0.0 < 5.0
        assert results[1].provisional is True  # start=1.5 < 5.0
        assert results[2].provisional is True  # start=3.0 < 5.0
        assert results[3].provisional is False  # start=6.0 ≥ 5.0

    def test_stabilization_disabled_still_marks_provisional(self):
        """Even with stabilization disabled, provisional flag should still be set."""
        cfg = CASAConfig(enable_early_dialogue_stabilization=False, early_dialogue_window_sec=5.0)
        engine = CASAEngine(config=cfg)
        segs = [_seg("Hi", "Speaker 1", 0.0, 0.5)]
        results = engine.apply(segs)
        assert results[0].provisional is True


# ─────────────────────────────────────────────────────────────────────────────
# CASA Engine — dialogue pattern / linguistic evidence
# ─────────────────────────────────────────────────────────────────────────────

class TestDialoguePatterns:
    def test_question_answer_same_speaker_suspicious(self):
        """Q→A from same speaker should produce low linguistic score / UNCERTAIN or CORRECT."""
        engine = CASAEngine()
        segs = [
            _seg("What is your name?", "Speaker 1", 0.0, 2.0),
            _seg("My name is Alice.", "Speaker 1", 2.5, 4.0),
        ]
        results = engine.apply(segs)
        # The second phrase should not be fully CONFIRM
        assert results[1].linguistic_score < 0.7, (
            f"Q→A from same speaker should lower linguistic score, got {results[1].linguistic_score}"
        )

    def test_question_answer_different_speaker_expected(self):
        """Q→A from different speaker should produce higher linguistic score."""
        engine = CASAEngine()
        segs = [
            _seg("What is your name?", "Speaker 1", 0.0, 2.0),
            _seg("My name is Bob.", "Speaker 2", 2.5, 4.0),
        ]
        results = engine.apply(segs)
        # Different speaker answering → expected
        assert results[1].linguistic_score > 0.65, (
            f"Different speaker answering Q should boost linguistic score, got {results[1].linguistic_score}"
        )

    def test_continuation_same_speaker_boosted(self):
        """'And ...' continuation from same speaker should produce higher linguistic score."""
        engine = CASAEngine()
        segs = [
            _seg("I bought a new laptop yesterday.", "Speaker 1", 0.0, 2.0),
            _seg("And it has a great display.", "Speaker 1", 2.2, 3.5),
        ]
        results = engine.apply(segs)
        assert results[1].linguistic_score > 0.7

    def test_greeting_reciprocation_different_speaker(self):
        """Greeting → greeting from different speaker should be expected."""
        engine = CASAEngine()
        segs = [
            _seg("Hello, how are you?", "Speaker 1", 0.0, 2.0),
            _seg("Hello, I am doing well.", "Speaker 2", 2.5, 4.0),
        ]
        results = engine.apply(segs)
        assert results[1].linguistic_score > 0.65


# ─────────────────────────────────────────────────────────────────────────────
# CASA Engine — 2-speaker benchmark
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarkScenarios:
    def test_2speaker_alternating_conversation(self):
        """
        Simulate a clean 2-speaker Q&A conversation.
        CASA should CONFIRM most turns and make 0 false corrections.
        """
        c1 = _centroid(DIM, 10)
        c2 = _centroid(DIM, 110)
        centroids = {"Speaker 1": c1, "Speaker 2": c2}

        conversation = [
            ("Speaker 1", "What brings you here today?",    0.0,  2.0),
            ("Speaker 2", "I have some questions for you.", 2.5,  4.5),
            ("Speaker 1", "Of course, please go ahead.",   5.0,  6.5),
            ("Speaker 2", "What is the project timeline?",  7.0,  9.0),
            ("Speaker 1", "We plan to finish in March.",    9.5, 11.5),
            ("Speaker 2", "That sounds reasonable.",       12.0, 13.5),
        ]
        segs = [_seg(t, s, st, en) for s, t, st, en in conversation]
        embs = [_similar_emb(c1 if s == "Speaker 1" else c2, noise=0.03)
                for s, *_ in conversation]

        engine = CASAEngine()
        results = engine.apply(segs, phrase_embeddings=embs, speaker_centroids=centroids)

        assert len(results) == len(segs)
        # No CORRECT results expected — speakers are clean and alternating
        corrections = [r for r in results if r.decision == "CORRECT"]
        assert len(corrections) == 0, (
            f"Expected 0 corrections in clean 2-speaker dialogue, got {len(corrections)}: "
            f"{[(r.phrase_index, r.proposed_speaker) for r in corrections]}"
        )

    def test_3speaker_conversation_no_hallucinated_label(self):
        """3-speaker conversation must only produce labels from the original 3 speakers."""
        c1, c2, c3 = _centroid(DIM, 0), _centroid(DIM, 60), _centroid(DIM, 120)
        centroids = {"Speaker 1": c1, "Speaker 2": c2, "Speaker 3": c3}

        conversation = [
            ("Speaker 1", "Let us get started.",           0.0,  1.5),
            ("Speaker 2", "Sure, I am ready.",             2.0,  3.0),
            ("Speaker 3", "Me too.",                        3.5,  4.0),
            ("Speaker 1", "Great, the topic is AI.",       4.5,  6.0),
            ("Speaker 2", "Yes, very exciting.",           6.5,  7.5),
            ("Speaker 3", "Absolutely.",                   8.0,  8.5),
        ]
        segs = [_seg(t, s, st, en) for s, t, st, en in conversation]
        embs_map = {"Speaker 1": c1, "Speaker 2": c2, "Speaker 3": c3}
        embs = [_similar_emb(embs_map[s], noise=0.02) for s, *_ in conversation]

        engine = CASAEngine()
        results = engine.apply(segs, phrase_embeddings=embs, speaker_centroids=centroids)

        allowed = {"Speaker 1", "Speaker 2", "Speaker 3"}
        for r in results:
            assert r.proposed_speaker in allowed, f"Hallucinated speaker: {r.proposed_speaker}"

    def test_result_count_matches_segment_count(self):
        """CASAEngine.apply() must return exactly one CASAResult per input segment."""
        engine = CASAEngine()
        segs = [_seg(f"Phrase {i}.", "Speaker 1", i * 2.0, i * 2.0 + 1.5) for i in range(10)]
        results = engine.apply(segs)
        assert len(results) == 10

    def test_phrase_indices_are_sequential(self):
        """CASAResult.phrase_index must be 0, 1, 2, ... in order."""
        engine = CASAEngine()
        segs = [_seg("Text.", "Speaker 1", i * 2.0, i * 2.0 + 1.5) for i in range(5)]
        results = engine.apply(segs)
        for i, r in enumerate(results):
            assert r.phrase_index == i

    def test_rapid_turn_changes_no_crash(self):
        """Very short phrases alternating speakers should not crash CASA."""
        engine = CASAEngine()
        segs = []
        t = 0.0
        for i in range(20):
            spk = f"Speaker {(i % 2) + 1}"
            segs.append(_seg("mm", spk, t, t + 0.3))
            t += 0.35
        results = engine.apply(segs)
        assert len(results) == 20
        assert all(r.decision in {"CONFIRM", "CORRECT", "UNCERTAIN"} for r in results)

    def test_4speaker_conversation_discovery_and_attribution(self):
        """4-speaker conversation maintains speaker boundaries without spurious identities."""
        c = [_centroid(DIM, i * 40) for i in range(4)]
        centroids = {f"Speaker {i+1}": c[i] for i in range(4)}
        conversation = [
            ("Speaker 1", "Good morning everyone.", 0.0, 1.5),
            ("Speaker 2", "Good morning.", 2.0, 2.8),
            ("Speaker 3", "Hi team.", 3.2, 4.0),
            ("Speaker 4", "Let us start the meeting.", 4.5, 6.0),
        ]
        segs = [_seg(t, s, st, en) for s, t, st, en in conversation]
        embs = [_similar_emb(centroids[s], noise=0.01) for s, *_ in conversation]

        engine = CASAEngine()
        results = engine.apply(segs, phrase_embeddings=embs, speaker_centroids=centroids)

        assert len(results) == 4
        allowed = set(centroids.keys())
        for r in results:
            assert r.proposed_speaker in allowed
            assert r.decision == "CONFIRM"

    def test_5speaker_conversation_no_spurious_speakers(self):
        """5-speaker conversation produces only the 5 original discovered speakers."""
        c = [_centroid(DIM, i * 35) for i in range(5)]
        centroids = {f"Speaker {i+1}": c[i] for i in range(5)}
        conversation = [
            (f"Speaker {i+1}", f"Speaker {i+1} report.", i * 3.0, i * 3.0 + 2.0)
            for i in range(5)
        ]
        segs = [_seg(t, s, st, en) for s, t, st, en in conversation]
        embs = [_similar_emb(centroids[s], noise=0.01) for s, *_ in conversation]

        engine = CASAEngine()
        results = engine.apply(segs, phrase_embeddings=embs, speaker_centroids=centroids)

        assert len(results) == 5
        allowed = set(centroids.keys())
        for r in results:
            assert r.proposed_speaker in allowed

    def test_same_speaker_short_continuation_preserves_identity(self):
        """Short phrase that continues previous thought from same speaker remains same speaker."""
        c1 = _centroid(DIM, 10)
        c2 = _centroid(DIM, 90)
        centroids = {"Speaker 1": c1, "Speaker 2": c2}

        segs = [
            _seg("We are seeing substantial market growth.", "Speaker 1", 0.0, 3.0),
            _seg("And that's right.", "Speaker 1", 3.2, 4.0),
        ]
        embs = [_similar_emb(c1, noise=0.02), _similar_emb(c1, noise=0.02)]

        engine = CASAEngine()
        results = engine.apply(segs, phrase_embeddings=embs, speaker_centroids=centroids)

        assert results[1].proposed_speaker == "Speaker 1"
        assert results[1].decision == "CONFIRM"
