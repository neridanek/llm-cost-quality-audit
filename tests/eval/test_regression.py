"""Regression check tests."""

from __future__ import annotations

import pytest

from lcqa.eval.regression import RegressionCheck, RegressionConfig
from lcqa.eval.result import EvalResult


def _result(metric: str, score: float) -> EvalResult:
    return EvalResult(metric=metric, score=score, per_item_scores=[score], n_items=1)


def test_passes_when_current_above_baseline() -> None:
    baseline = _result("faithfulness", 0.80)
    current = _result("faithfulness", 0.85)
    check = RegressionCheck.run(baseline, current)
    assert check.passed
    assert check.delta == pytest.approx(0.05)


def test_passes_when_within_threshold() -> None:
    baseline = _result("faithfulness", 0.85)
    current = _result("faithfulness", 0.81)  # 4pp drop, default threshold = 5pp
    check = RegressionCheck.run(baseline, current)
    assert check.passed
    assert check.delta == pytest.approx(-0.04)


def test_fails_beyond_threshold() -> None:
    baseline = _result("faithfulness", 0.85)
    current = _result("faithfulness", 0.78)  # 7pp drop > 5pp default
    check = RegressionCheck.run(baseline, current)
    assert not check.passed


def test_custom_threshold() -> None:
    baseline = _result("accuracy", 0.75)
    current = _result("accuracy", 0.74)
    config = RegressionConfig(threshold=0.005)  # 0.5pp tolerance
    check = RegressionCheck.run(baseline, current, config)
    assert not check.passed


def test_metric_mismatch_raises() -> None:
    baseline = _result("faithfulness", 0.85)
    current = _result("accuracy", 0.85)
    with pytest.raises(ValueError, match="Metric mismatch"):
        RegressionCheck.run(baseline, current)


def test_threshold_out_of_range_raises() -> None:
    baseline = _result("m", 0.5)
    current = _result("m", 0.5)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        RegressionCheck.run(baseline, current, RegressionConfig(threshold=1.5))


def test_check_serializable() -> None:
    baseline = _result("faithfulness", 0.85)
    current = _result("faithfulness", 0.86)
    check = RegressionCheck.run(baseline, current)
    payload = check.model_dump()
    assert payload["passed"] is True
    assert payload["metric"] == "faithfulness"
