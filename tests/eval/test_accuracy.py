"""Accuracy scorer tests (exact match + custom scorer paths)."""

from __future__ import annotations

import pytest

from lcqa.eval.accuracy import AccuracyConfig, exact_match, normalize_answer, score_accuracy
from lcqa.eval.result import EvalCase


def test_normalize_answer_case_punctuation_articles() -> None:
    assert normalize_answer("The Paris.") == "paris"
    assert normalize_answer("a CAT") == "cat"
    assert normalize_answer("  multiple   spaces  ") == "multiple spaces"


def test_exact_match_normalized_match() -> None:
    case = EvalCase(question="capital?", answer="The Paris.", expected_answer="paris")
    assert exact_match(case) == 1.0


def test_exact_match_mismatch() -> None:
    case = EvalCase(question="q", answer="London", expected_answer="Paris")
    assert exact_match(case) == 0.0


def test_exact_match_missing_expected_raises() -> None:
    case = EvalCase(question="q", answer="x")
    with pytest.raises(ValueError, match="expected_answer"):
        exact_match(case)


def test_score_accuracy_aggregates_correctly() -> None:
    cases = [
        EvalCase(question="q1", answer="Paris", expected_answer="Paris"),
        EvalCase(question="q2", answer="London", expected_answer="Paris"),
        EvalCase(question="q3", answer="Berlin", expected_answer="Berlin"),
    ]
    result = score_accuracy(cases)
    assert result.metric == "accuracy"
    assert result.n_items == 3
    assert result.score == pytest.approx(2 / 3)
    assert result.per_item_scores == [1.0, 0.0, 1.0]
    assert result.metadata["mode"] == "exact_match"


def test_score_accuracy_empty_input() -> None:
    result = score_accuracy([])
    assert result.score == 0.0
    assert result.n_items == 0


def test_score_accuracy_with_custom_scorer() -> None:
    cases = [
        EvalCase(question="q", answer="a", expected_answer="b"),
        EvalCase(question="q", answer="a", expected_answer="b"),
    ]

    def always_half(_case: EvalCase) -> float:
        return 0.5

    result = score_accuracy(cases, scorer=always_half)
    assert result.score == 0.5
    assert result.per_item_scores == [0.5, 0.5]


def test_score_accuracy_unknown_mode_raises() -> None:
    cases = [EvalCase(question="q", answer="a", expected_answer="b")]
    with pytest.raises(ValueError, match="Unknown mode"):
        score_accuracy(cases, config=AccuracyConfig(mode="hallucinate"))


def test_score_accuracy_missing_expected_answer_raises() -> None:
    cases = [EvalCase(question="q", answer="a")]
    with pytest.raises(ValueError, match="expected_answer"):
        score_accuracy(cases)


def test_score_accuracy_scorer_out_of_range_raises() -> None:
    cases = [EvalCase(question="q", answer="a", expected_answer="b")]

    def bad_scorer(_c: EvalCase) -> float:
        return 1.5

    with pytest.raises(ValueError, match=r"expected \[0,1\]"):
        score_accuracy(cases, scorer=bad_scorer)
