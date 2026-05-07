"""Regression check helpers — wire eval results into CI.

Usage in CI:

    from lcqa.eval import EvalResult, RegressionCheck, RegressionConfig

    baseline = EvalResult.from_json("eval_results/baseline.json")
    current = score_faithfulness(cases)
    check = RegressionCheck.run(baseline, current, RegressionConfig(threshold=0.05))
    if not check.passed:
        sys.exit(1)
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from lcqa.eval.result import EvalResult


@dataclass(frozen=True, slots=True)
class RegressionConfig:
    """Threshold = max allowed score drop below baseline (default: 0.05 = 5pp)."""

    threshold: float = 0.05


class RegressionCheck(BaseModel):
    """Outcome of comparing a current eval run against a baseline."""

    metric: str
    baseline_score: float
    current_score: float
    delta: float = Field(description="current - baseline (negative = regression)")
    threshold: float
    passed: bool

    @classmethod
    def run(
        cls,
        baseline: EvalResult,
        current: EvalResult,
        config: RegressionConfig | None = None,
    ) -> RegressionCheck:
        if baseline.metric != current.metric:
            raise ValueError(
                f"Metric mismatch: baseline={baseline.metric!r} vs current={current.metric!r}"
            )
        config = config or RegressionConfig()
        if not 0 <= config.threshold <= 1:
            raise ValueError(f"threshold must be in [0,1], got {config.threshold}")
        delta = current.score - baseline.score
        passed = delta >= -config.threshold
        return cls(
            metric=baseline.metric,
            baseline_score=baseline.score,
            current_score=current.score,
            delta=delta,
            threshold=config.threshold,
            passed=passed,
        )
