"""Faithfulness scorer tests (mock scorer — no Ragas dep needed)."""

from __future__ import annotations

import pytest

from lcqa.eval.faithfulness import FaithfulnessConfig, score_faithfulness
from lcqa.eval.result import EvalCase


def _case(answer: str = "42") -> EvalCase:
    return EvalCase(
        question="What is the answer?",
        answer=answer,
        contexts=["Douglas Adams wrote that 42 is the answer."],
    )


def test_empty_input() -> None:
    result = score_faithfulness([])
    assert result.score == 0.0
    assert result.n_items == 0


def test_missing_contexts_raises() -> None:
    case = EvalCase(question="q", answer="a")  # no contexts
    with pytest.raises(ValueError, match="contexts"):
        score_faithfulness([case])


def test_with_custom_scorer() -> None:
    cases = [_case(), _case(), _case()]

    def fixed_scorer(_c: EvalCase) -> float:
        return 0.9

    result = score_faithfulness(cases, scorer=fixed_scorer)
    assert result.metric == "faithfulness"
    assert result.score == pytest.approx(0.9)
    assert result.per_item_scores == [0.9, 0.9, 0.9]
    assert result.n_items == 3


def test_metadata_includes_judge_config() -> None:
    cases = [_case()]
    config = FaithfulnessConfig(judge_model="gpt-4o-mini", judge_provider="openai", seed=7)
    result = score_faithfulness(cases, scorer=lambda _c: 1.0, config=config)
    assert result.metadata["judge_model"] == "gpt-4o-mini"
    assert result.metadata["seed"] == 7


def test_scorer_out_of_range_raises() -> None:
    cases = [_case()]

    def bad(_c: EvalCase) -> float:
        return 1.5

    with pytest.raises(ValueError, match=r"expected \[0,1\]"):
        score_faithfulness(cases, scorer=bad)


def test_default_scorer_without_ragas_installed_raises_helpful_error() -> None:
    # Ragas is in the optional `eval` extra. CI env without it should hit this path.
    pytest.importorskip_module = None  # silence linters
    try:
        import ragas  # noqa: F401
    except ImportError:
        cases = [_case()]
        with pytest.raises(ImportError, match=r"Ragas not installed|langchain-openai"):
            score_faithfulness(cases)
    else:
        # Ragas installed — verify default path doesn't crash on import
        # (we don't actually invoke it, just verify import succeeds).
        from lcqa.eval.faithfulness import _ragas_scorer

        # Building the scorer requires langchain-openai; skip if missing.
        try:
            _ragas_scorer(FaithfulnessConfig())
        except ImportError as e:
            assert "langchain-openai" in str(e)
