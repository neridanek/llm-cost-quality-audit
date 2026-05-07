"""Answer accuracy scoring.

Two scorers ship out of the box:

- `exact_match` — strict normalized string equality, no LLM judge. Cheap, useful
  for QA benchmarks with short factual answers (HotpotQA, TriviaQA).
- `llm_judge` — DeepEval `GEval` (or any callable). Better for free-form answers
  where the model's wording differs from gold but the meaning is correct.

You can also pass a custom callable. See ADR-002 for the batch-eval rationale.
"""

from __future__ import annotations

import re
import string
from collections.abc import Callable
from dataclasses import dataclass

from lcqa.eval.result import EvalCase, EvalResult

AccuracyScorer = Callable[[EvalCase], float]

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class AccuracyConfig:
    """Config for the LLM-judge scorer (used when `mode='llm_judge'`)."""

    mode: str = "exact_match"  # 'exact_match' | 'llm_judge'
    judge_model: str = "gpt-4o-mini"
    judge_provider: str = "openai"
    threshold: float = 0.7  # llm_judge: scores >= threshold count as correct


def normalize_answer(s: str) -> str:
    """SQuAD-style answer normalization: lowercase, strip punctuation, articles, whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def exact_match(case: EvalCase) -> float:
    if case.expected_answer is None:
        raise ValueError(f"Case {case.question[:40]!r}... missing `expected_answer`")
    return 1.0 if normalize_answer(case.answer) == normalize_answer(case.expected_answer) else 0.0


def score_accuracy(
    cases: list[EvalCase],
    scorer: AccuracyScorer | None = None,
    config: AccuracyConfig | None = None,
) -> EvalResult:
    """Score answer accuracy over a batch of cases.

    Defaults to SQuAD-style exact match. Pass `config=AccuracyConfig(mode='llm_judge')`
    to route through DeepEval, or pass a custom `scorer=` callable.
    """
    if not cases:
        return EvalResult(
            metric="accuracy",
            score=0.0,
            per_item_scores=[],
            n_items=0,
            metadata={"reason": "empty input"},
        )
    config = config or AccuracyConfig()
    for c in cases:
        if c.expected_answer is None:
            raise ValueError(
                f"Case {c.question[:40]!r}... missing `expected_answer` — "
                "accuracy scoring requires ground truth"
            )
    if scorer is None:
        scorer = _resolve_default_scorer(config)
    per_item = [scorer(c) for c in cases]
    for s, c in zip(per_item, cases, strict=True):
        if not 0.0 <= s <= 1.0:
            raise ValueError(f"Scorer returned {s}; expected [0,1] for case {c.question[:40]!r}")
    return EvalResult(
        metric="accuracy",
        score=sum(per_item) / len(per_item),
        per_item_scores=per_item,
        n_items=len(cases),
        metadata={
            "mode": config.mode,
            "judge_model": config.judge_model if config.mode == "llm_judge" else None,
        },
    )


def _resolve_default_scorer(config: AccuracyConfig) -> AccuracyScorer:
    if config.mode == "exact_match":
        return exact_match
    if config.mode == "llm_judge":
        return _llm_judge_scorer(config)
    raise ValueError(f"Unknown mode {config.mode!r}; expected 'exact_match' or 'llm_judge'")


def _llm_judge_scorer(config: AccuracyConfig) -> AccuracyScorer:
    """DeepEval-backed LLM-as-judge correctness scorer. Imported lazily."""
    try:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    except ImportError as e:
        raise ImportError(
            "DeepEval not installed. Install with `pip install lcqa[eval]` "
            "or pass a custom `scorer=` callable."
        ) from e

    metric = GEval(
        name="AnswerCorrectness",
        criteria=(
            "Determine whether the actual output is factually consistent with "
            "the expected output. Penalize contradictions, partial answers, and "
            "fabricated details."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=config.judge_model,
        threshold=config.threshold,
    )

    def _scorer(case: EvalCase) -> float:
        test_case = LLMTestCase(
            input=case.question,
            actual_output=case.answer,
            expected_output=case.expected_answer or "",
        )
        metric.measure(test_case)
        return float(metric.score or 0.0)

    return _scorer
