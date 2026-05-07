"""Example 02 — Eval harness with mock scorer.

Run:
    python examples/02_eval_harness.py

What it shows:
- Building `EvalCase` with question/answer/contexts
- Scoring faithfulness with a custom (deterministic) scorer — no API key needed
- Scoring accuracy with `exact_match` (no LLM judge needed)
- Latency tracker context manager
- Writing results to JSON for CI consumption

For real Ragas/DeepEval scoring, install `lcqa[eval]` and an API key.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from lcqa.eval.accuracy import score_accuracy
from lcqa.eval.faithfulness import score_faithfulness
from lcqa.eval.latency import LatencyTracker
from lcqa.eval.result import EvalCase


def _mock_faithfulness_scorer(case: EvalCase) -> float:
    """Score based on how many words from `answer` appear in `contexts` — toy heuristic."""
    if not case.contexts:
        return 0.0
    answer_words = set(case.answer.lower().split())
    if not answer_words:
        return 0.0
    context_blob = " ".join(case.contexts).lower()
    overlap = sum(1 for w in answer_words if w in context_blob)
    return overlap / len(answer_words)


def main() -> None:
    random.seed(42)

    cases = [
        EvalCase(
            question="What is the capital of France?",
            answer="Paris",
            contexts=["Paris is the capital and largest city of France."],
            expected_answer="Paris",
        ),
        EvalCase(
            question="Who wrote Hitchhiker's Guide?",
            answer="Douglas Adams",
            contexts=["Douglas Adams wrote The Hitchhiker's Guide to the Galaxy in 1979."],
            expected_answer="Douglas Adams",
        ),
        EvalCase(
            question="What year was Linux released?",
            answer="1991",
            contexts=["Linus Torvalds released Linux in 1991."],
            expected_answer="1991",
        ),
        EvalCase(
            question="What is the largest planet?",
            answer="Mars",  # wrong
            contexts=["Jupiter is the largest planet in our solar system."],
            expected_answer="Jupiter",
        ),
    ]

    tracker = LatencyTracker()
    with tracker.measure(tag="faithfulness"):
        time.sleep(0.01)  # simulate scoring latency
        f_result = score_faithfulness(cases, scorer=_mock_faithfulness_scorer)

    with tracker.measure(tag="accuracy"):
        a_result = score_accuracy(cases)  # default exact_match

    print(f"Faithfulness:  score={f_result.score:.3f}  per_item={f_result.per_item_scores}")
    print(f"Accuracy:      score={a_result.score:.3f}  per_item={a_result.per_item_scores}")
    print()
    print("Latency summary:")
    for tag in ("faithfulness", "accuracy"):
        s = tracker.summary(tag=tag)
        print(f"  {tag:>14}  n={s['n']}  p50={s['p50']:.2f}ms  p95={s['p95']:.2f}ms")

    out_dir = Path("eval_results")
    f_result.to_json(out_dir / "example_02_faithfulness.json")
    a_result.to_json(out_dir / "example_02_accuracy.json")
    print(f"\nWrote {out_dir}/example_02_*.json")


if __name__ == "__main__":
    main()
