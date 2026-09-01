"""
V3.2 Attribution Evaluation Benchmark
======================================
Reproducible benchmark comparing V3.1 (CASA disabled) vs V3.2 (CASA enabled)
across synthetic ground-truth conversation scenarios.

Usage:
    python tests/evaluation/benchmark_attribution.py

Output: metrics table printed to stdout.
No GPU / network required — all scenarios are synthetic.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is on path when running directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.casa_config import CASAConfig
from services.casa_engine import CASAEngine

DIM = 192  # ECAPA embedding dimension


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth scenario builder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ConversationTurn:
    speaker: str
    text: str
    start_sec: float
    end_sec: float
    is_short: bool = False
    is_early: bool = False


@dataclass
class BenchmarkScenario:
    name: str
    turns: List[ConversationTurn]
    speaker_centroids: Dict[str, np.ndarray]
    phrase_embeddings: List[Optional[np.ndarray]]

    @property
    def ground_truth_labels(self) -> List[str]:
        return [t.speaker for t in self.turns]


def _unit(dim: int, idx: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


def _noisy(centroid: np.ndarray, rng: np.random.RandomState, noise: float = 0.02) -> np.ndarray:
    v = centroid + noise * rng.randn(len(centroid)).astype(np.float32)
    return v / np.linalg.norm(v)


def _build_2speaker_scenario(rng: np.random.RandomState) -> BenchmarkScenario:
    c1, c2 = _unit(DIM, 10), _unit(DIM, 110)
    turns = [
        ConversationTurn("Speaker 1", "Hello, how are you doing today?", 0.0, 2.0),
        ConversationTurn("Speaker 2", "I am doing well, thank you for asking.", 2.5, 4.5),
        ConversationTurn("Speaker 1", "What brings you here?", 5.0, 6.5),
        ConversationTurn("Speaker 2", "I have a few questions about the project.", 7.0, 9.0),
        ConversationTurn("Speaker 1", "Of course, please ask away.", 9.5, 11.0),
        ConversationTurn("Speaker 2", "When is the expected completion date?", 11.5, 13.5),
        ConversationTurn("Speaker 1", "We aim to finish by the end of March.", 14.0, 16.0),
        ConversationTurn("Speaker 2", "That sounds reasonable.", 16.5, 17.5),
    ]
    embs = [_noisy(c1 if t.speaker == "Speaker 1" else c2, rng) for t in turns]
    return BenchmarkScenario(
        name="2-speaker Q&A dialogue",
        turns=turns,
        speaker_centroids={"Speaker 1": c1, "Speaker 2": c2},
        phrase_embeddings=embs,
    )


def _build_3speaker_scenario(rng: np.random.RandomState) -> BenchmarkScenario:
    c1, c2, c3 = _unit(DIM, 0), _unit(DIM, 60), _unit(DIM, 120)
    centroids = {"Speaker 1": c1, "Speaker 2": c2, "Speaker 3": c3}
    turns = [
        ConversationTurn("Speaker 1", "Let us start the meeting.", 0.0, 1.5),
        ConversationTurn("Speaker 2", "I agree. Let me present the agenda.", 2.0, 4.0),
        ConversationTurn("Speaker 3", "I have a question before we begin.", 4.5, 6.0),
        ConversationTurn("Speaker 1", "Go ahead, please.", 6.5, 7.5),
        ConversationTurn("Speaker 3", "What is our primary goal for this quarter?", 8.0, 10.0),
        ConversationTurn("Speaker 2", "Our goal is to launch the new feature.", 10.5, 12.5),
        ConversationTurn("Speaker 1", "Exactly, and we must hit the deadline.", 13.0, 14.5),
    ]
    embs = [_noisy(centroids[t.speaker], rng) for t in turns]
    return BenchmarkScenario(
        name="3-speaker meeting",
        turns=turns,
        speaker_centroids=centroids,
        phrase_embeddings=embs,
    )


def _build_short_utterance_scenario(rng: np.random.RandomState) -> BenchmarkScenario:
    c1, c2 = _unit(DIM, 15), _unit(DIM, 115)
    turns = [
        ConversationTurn("Speaker 1", "Did you receive the email?", 0.0, 1.5),
        ConversationTurn("Speaker 2", "yes", 2.0, 2.3, is_short=True),
        ConversationTurn("Speaker 1", "And did you read it?", 2.8, 4.0),
        ConversationTurn("Speaker 2", "mm", 4.5, 4.7, is_short=True),
        ConversationTurn("Speaker 1", "What do you think about the proposal?", 5.2, 7.0),
        ConversationTurn("Speaker 2", "okay", 7.5, 7.8, is_short=True),
        ConversationTurn("Speaker 2", "I think it is quite good actually.", 8.0, 9.5),
        ConversationTurn("Speaker 1", "right", 9.8, 10.0, is_short=True),
    ]
    embs = []
    for t in turns:
        if t.is_short:
            # Short utterances: weaker / noisier embeddings
            embs.append(_noisy(c1 if t.speaker == "Speaker 1" else c2, rng, noise=0.20))
        else:
            embs.append(_noisy(c1 if t.speaker == "Speaker 1" else c2, rng, noise=0.03))
    return BenchmarkScenario(
        name="short utterances",
        turns=turns,
        speaker_centroids={"Speaker 1": c1, "Speaker 2": c2},
        phrase_embeddings=embs,
    )


def _build_early_dialogue_scenario(rng: np.random.RandomState) -> BenchmarkScenario:
    c1, c2 = _unit(DIM, 20), _unit(DIM, 120)
    # First 3 turns within 5s; embeddings slightly weaker
    turns = [
        ConversationTurn("Speaker 1", "Hello.", 0.0, 0.5, is_early=True),
        ConversationTurn("Speaker 2", "Hi there.", 0.8, 1.5, is_early=True),
        ConversationTurn("Speaker 1", "Nice to meet you.", 2.0, 3.0, is_early=True),
        ConversationTurn("Speaker 2", "Nice to meet you too.", 3.5, 4.5, is_early=True),
        ConversationTurn("Speaker 1", "What are we working on today?", 6.0, 7.5),
        ConversationTurn("Speaker 2", "We are reviewing the Q3 report.", 8.0, 9.5),
        ConversationTurn("Speaker 1", "Good, I have some notes on that.", 10.0, 11.5),
    ]
    embs = []
    for t in turns:
        noise = 0.15 if t.is_early else 0.03
        embs.append(_noisy(c1 if t.speaker == "Speaker 1" else c2, rng, noise=noise))
    return BenchmarkScenario(
        name="early dialogue stabilization",
        turns=turns,
        speaker_centroids={"Speaker 1": c1, "Speaker 2": c2},
        phrase_embeddings=embs,
    )


def _build_5speaker_scenario(rng: np.random.RandomState) -> BenchmarkScenario:
    centroids = {f"Speaker {i+1}": _unit(DIM, i * 30) for i in range(5)}
    speakers = [f"Speaker {(i % 5) + 1}" for i in range(15)]
    texts = [
        "I would like to introduce myself.",
        "Hello everyone, glad to be here.",
        "Let me start with the overview.",
        "That is a great point, thank you.",
        "Could you clarify that last statement?",
        "Of course, let me explain further.",
        "I agree with the previous speaker.",
        "We should move on to the next item.",
        "I have a question about the timeline.",
        "Sure, we plan to deliver in Q2.",
        "That works for me.",
        "Let me add one more thing.",
        "Please go ahead.",
        "Thank you all for your time today.",
        "Looking forward to the next session.",
    ]
    t = 0.0
    turns = []
    for spk, txt in zip(speakers, texts):
        dur = 1.0 + rng.rand() * 2.0
        turns.append(ConversationTurn(spk, txt, t, t + dur))
        t += dur + 0.5 + rng.rand() * 0.5
    embs = [_noisy(centroids[turn.speaker], rng) for turn in turns]
    return BenchmarkScenario(
        name="5-speaker panel",
        turns=turns,
        speaker_centroids=centroids,
        phrase_embeddings=embs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioMetrics:
    scenario_name: str
    total_turns: int
    correct: int
    incorrect: int
    accuracy: float
    corrections_made: int
    correct_corrections: int
    false_corrections: int
    uncertain_count: int
    short_utt_total: int
    short_utt_correct: int
    early_turns_total: int
    early_turns_correct: int
    runtime_ms: float


def _evaluate(
    scenario: BenchmarkScenario,
    use_casa: bool,
) -> ScenarioMetrics:
    """Run CASA (enabled or disabled) and compare against ground truth."""
    cfg = CASAConfig(enable_casa=use_casa)
    engine = CASAEngine(config=cfg)

    # Build diarized_segments (as diarize_audio would produce)
    diarized = []
    for t in scenario.turns:
        words = [{"word": w} for w in t.text.split()]
        diarized.append({
            "text": t.text,
            "speaker_label": t.speaker,
            "start_sec": t.start_sec,
            "end_sec": t.end_sec,
            "duration_sec": t.end_sec - t.start_sec,
            "words": words,
        })

    t0 = time.perf_counter()
    results = engine.apply(
        diarized_segments=diarized,
        phrase_embeddings=scenario.phrase_embeddings,
        speaker_centroids=scenario.speaker_centroids,
    )
    runtime_ms = (time.perf_counter() - t0) * 1000

    gt = scenario.ground_truth_labels
    correct = incorrect = 0
    corrections_made = correct_corrections = false_corrections = 0
    uncertain_count = 0
    short_total = short_correct = 0
    early_total = early_correct = 0

    cfg_full = CASAConfig()

    for i, (r, turn) in enumerate(zip(results, scenario.turns)):
        is_correct = r.proposed_speaker == gt[i]
        if is_correct:
            correct += 1
        else:
            incorrect += 1

        if r.decision == "CORRECT":
            corrections_made += 1
            if is_correct:
                correct_corrections += 1
            else:
                false_corrections += 1
        elif r.decision == "UNCERTAIN":
            uncertain_count += 1

        if turn.is_short:
            short_total += 1
            if is_correct:
                short_correct += 1

        if turn.is_early:
            early_total += 1
            if is_correct:
                early_correct += 1

    total = len(results)
    accuracy = correct / total if total > 0 else 0.0

    return ScenarioMetrics(
        scenario_name=scenario.name,
        total_turns=total,
        correct=correct,
        incorrect=incorrect,
        accuracy=accuracy,
        corrections_made=corrections_made,
        correct_corrections=correct_corrections,
        false_corrections=false_corrections,
        uncertain_count=uncertain_count,
        short_utt_total=short_total,
        short_utt_correct=short_correct,
        early_turns_total=early_total,
        early_turns_correct=early_correct,
        runtime_ms=runtime_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _pct(num, den):
    return f"{num/den*100:.1f}%" if den > 0 else "N/A"


def run_benchmark(seed: int = 42):
    rng = np.random.RandomState(seed)
    scenarios = [
        _build_2speaker_scenario(rng),
        _build_3speaker_scenario(rng),
        _build_early_dialogue_scenario(rng),
        _build_short_utterance_scenario(rng),
        _build_5speaker_scenario(rng),
    ]

    print("\n" + "=" * 90)
    print("  V3.2 CASA Attribution Benchmark  (seed={})".format(seed))
    print("=" * 90)

    header = f"{'Scenario':<35} {'V3.1 Acc':>9} {'V3.2 Acc':>9} {'Delta':>6} {'Corrections':>12} {'FalseCorr':>10} {'Short Acc':>10} {'Early Acc':>10}"
    print(header)
    print("-" * 90)

    aggregate_v31 = aggregate_v32 = total_turns = 0
    total_corrections = total_false = 0
    short_total_all = short_correct_v32 = 0
    early_total_all = early_correct_v32 = 0

    for scenario in scenarios:
        m31 = _evaluate(scenario, use_casa=False)
        m32 = _evaluate(scenario, use_casa=True)

        delta = (m32.accuracy - m31.accuracy) * 100
        delta_str = f"{delta:+.1f}%"

        short_acc = _pct(m32.short_utt_correct, m32.short_utt_total)
        early_acc = _pct(m32.early_turns_correct, m32.early_turns_total)

        print(
            f"{scenario.name:<35} "
            f"{m31.accuracy*100:>8.1f}% "
            f"{m32.accuracy*100:>8.1f}% "
            f"{delta_str:>7} "
            f"{m32.corrections_made:>5}/{m32.total_turns:<5} "
            f"{m32.false_corrections:>10} "
            f"{short_acc:>10} "
            f"{early_acc:>10}"
        )

        aggregate_v31 += m31.correct
        aggregate_v32 += m32.correct
        total_turns += m31.total_turns
        total_corrections += m32.corrections_made
        total_false += m32.false_corrections
        short_total_all += m32.short_utt_total
        short_correct_v32 += m32.short_utt_correct
        early_total_all += m32.early_turns_total
        early_correct_v32 += m32.early_turns_correct

    print("-" * 90)
    agg31 = aggregate_v31 / total_turns
    agg32 = aggregate_v32 / total_turns
    delta = (agg32 - agg31) * 100
    print(
        f"{'OVERALL':<35} "
        f"{agg31*100:>8.1f}% "
        f"{agg32*100:>8.1f}% "
        f"{delta:>+7.1f}% "
        f"{total_corrections:>5}/{total_turns:<5} "
        f"{total_false:>10} "
        f"{_pct(short_correct_v32, short_total_all):>10} "
        f"{_pct(early_correct_v32, early_total_all):>10}"
    )
    print("=" * 90)
    print()
    print("Columns: V3.1 Acc = baseline (CASA off)  |  V3.2 Acc = CASA on")
    print("         Corrections = CORRECT decisions  |  FalseCorr = CORRECT that were wrong")
    print("         Short Acc = accuracy on short utterances  |  Early Acc = accuracy in first 5s")
    print()


if __name__ == "__main__":
    run_benchmark()
