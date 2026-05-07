"""EvalResult + EvalCase model tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from lcqa.eval.result import EvalCase, EvalResult


def test_evalcase_minimal() -> None:
    c = EvalCase(question="What?", answer="42")
    assert c.contexts is None
    assert c.expected_answer is None


def test_evalcase_with_contexts_and_expected() -> None:
    c = EvalCase(
        question="What?",
        answer="42",
        contexts=["Douglas Adams wrote 42 is the answer."],
        expected_answer="42",
    )
    assert c.contexts == ["Douglas Adams wrote 42 is the answer."]


def test_evalresult_score_bounds() -> None:
    EvalResult(metric="t", score=0.5, per_item_scores=[0.5], n_items=1)
    with pytest.raises(ValidationError):
        EvalResult(metric="t", score=1.5, per_item_scores=[1.0], n_items=1)
    with pytest.raises(ValidationError):
        EvalResult(metric="t", score=-0.1, per_item_scores=[0.0], n_items=1)


def test_evalresult_json_roundtrip(tmp_path: Path) -> None:
    r = EvalResult(
        metric="faithfulness",
        score=0.87,
        per_item_scores=[1.0, 0.8, 0.81],
        n_items=3,
        metadata={"judge_model": "gpt-4o-mini"},
    )
    out = r.to_json(tmp_path / "result.json")
    loaded = EvalResult.from_json(out)
    assert loaded.metric == r.metric
    assert loaded.score == pytest.approx(r.score)
    assert loaded.per_item_scores == r.per_item_scores
    assert loaded.metadata["judge_model"] == "gpt-4o-mini"
