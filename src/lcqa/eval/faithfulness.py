"""Faithfulness scoring — Ragas wrapper with mock scorer fallback.

Faithfulness measures whether `answer` is supported by `contexts` (no
hallucinations). We default to Ragas (`Faithfulness` metric) but accept any
callable scorer for testing or alternate frameworks.

See ADR-001 for why Ragas was chosen over TruLens.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from lcqa.eval.result import EvalCase, EvalResult

# Scorer signature: takes (case) and returns a float score in [0, 1].
FaithfulnessScorer = Callable[[EvalCase], float]


@dataclass(frozen=True, slots=True)
class FaithfulnessConfig:
    """Configuration for the Ragas-backed scorer.

    `judge_model` is the LLM used by Ragas to evaluate faithfulness — typically
    a cheaper model than the one being evaluated, to keep eval cost low.
    """

    judge_model: str = "gpt-4o-mini"
    judge_provider: str = "openai"
    seed: int | None = 42


def score_faithfulness(
    cases: list[EvalCase],
    scorer: FaithfulnessScorer | None = None,
    config: FaithfulnessConfig | None = None,
) -> EvalResult:
    """Score a batch of `EvalCase` for faithfulness.

    `scorer=None` triggers the Ragas-backed scorer (requires `lcqa[eval]`).
    Pass a custom scorer for unit tests or alternate frameworks.
    """
    if not cases:
        return EvalResult(
            metric="faithfulness",
            score=0.0,
            per_item_scores=[],
            n_items=0,
            metadata={"reason": "empty input"},
        )
    for c in cases:
        if not c.contexts:
            raise ValueError(
                f"EvalCase {c.question[:40]!r}... missing `contexts` "
                "— faithfulness requires retrieved context"
            )
    config = config or FaithfulnessConfig()
    if scorer is None:
        scorer = _ragas_scorer(config)
    per_item: list[float] = []
    for case in cases:
        s = scorer(case)
        if not 0.0 <= s <= 1.0:
            raise ValueError(f"Scorer returned {s}; expected [0,1] for case {case.question[:40]!r}")
        per_item.append(s)
    mean_score = sum(per_item) / len(per_item)
    return EvalResult(
        metric="faithfulness",
        score=mean_score,
        per_item_scores=per_item,
        n_items=len(cases),
        metadata={
            "judge_model": config.judge_model,
            "judge_provider": config.judge_provider,
            "seed": config.seed,
        },
    )


def _ragas_scorer(config: FaithfulnessConfig) -> FaithfulnessScorer:
    """Build a Ragas-backed faithfulness scorer. Imported lazily."""
    try:
        from ragas import SingleTurnSample
        from ragas.metrics import Faithfulness
    except ImportError as e:
        raise ImportError(
            "Ragas not installed. Install with `pip install lcqa[eval]` "
            "or pass a custom `scorer=` callable."
        ) from e

    metric = _build_ragas_metric(Faithfulness, config)

    def _scorer(case: EvalCase) -> float:
        sample = SingleTurnSample(
            user_input=case.question,
            response=case.answer,
            retrieved_contexts=case.contexts or [],
        )
        return float(metric.single_turn_score(sample))

    return _scorer


def _build_ragas_metric(metric_cls: Any, config: FaithfulnessConfig) -> Any:
    """Wire Ragas metric with judge LLM. Provider-specific glue lives here."""
    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
    except ImportError as e:
        raise ImportError(
            "langchain-openai required for the default Ragas judge. "
            "Install `pip install langchain-openai` or pass a custom scorer."
        ) from e

    if config.judge_provider != "openai":
        raise NotImplementedError(
            f"Default judge wiring only supports OpenAI; got {config.judge_provider!r}. "
            "Pass a custom scorer for other providers."
        )
    llm = LangchainLLMWrapper(ChatOpenAI(model=config.judge_model, seed=config.seed))
    metric = metric_cls(llm=llm)
    return metric
