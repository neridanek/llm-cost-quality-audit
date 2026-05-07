"""CLI smoke tests — direct invocation, no subprocess."""

from __future__ import annotations

from pathlib import Path

import pytest

from lcqa.cli import main
from lcqa.cost.analyzer import CostAnalyzer, RequestRecord
from lcqa.eval.result import EvalResult


def test_version_command_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["version"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "lcqa " in out
    assert "pricing table last updated" in out


def test_list_models_default_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["list-models"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "openai" in out
    assert "anthropic" in out
    assert "gpt-4o" in out


def test_list_models_filtered(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["list-models", "--provider", "anthropic"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "anthropic" in out
    assert "openai" not in out


def test_list_models_unknown_provider_exits_one(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["list-models", "--provider", "cohere"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "No models found" in err


def test_cost_summary_loads_records(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = CostAnalyzer()
    a.add(RequestRecord(provider="openai", model="gpt-4o", input_tokens=1000, output_tokens=500, tag="answer"))
    a.add(RequestRecord(provider="openai", model="gpt-4o-mini", input_tokens=500, output_tokens=100, tag="rerank"))
    out_path = a.to_json(tmp_path / "records.json")
    rc = main(["cost-summary", str(out_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Records: 2" in out
    assert "By model" in out
    assert "By tag" in out


def test_cost_summary_missing_file_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["cost-summary", "/nonexistent/records.json"])
    assert rc == 2
    assert "File not found" in capsys.readouterr().err


def test_eval_regress_pass_returns_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = EvalResult(metric="faithfulness", score=0.85, per_item_scores=[0.85], n_items=1)
    cur = EvalResult(metric="faithfulness", score=0.86, per_item_scores=[0.86], n_items=1)
    bp = base.to_json(tmp_path / "baseline.json")
    cp = cur.to_json(tmp_path / "current.json")
    rc = main(["eval-regress", str(bp), str(cp)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "[PASS]" in out


def test_eval_regress_fail_returns_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = EvalResult(metric="faithfulness", score=0.85, per_item_scores=[0.85], n_items=1)
    cur = EvalResult(metric="faithfulness", score=0.70, per_item_scores=[0.70], n_items=1)
    bp = base.to_json(tmp_path / "baseline.json")
    cp = cur.to_json(tmp_path / "current.json")
    rc = main(["eval-regress", str(bp), str(cp)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "[FAIL]" in out


def test_eval_regress_custom_threshold(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = EvalResult(metric="accuracy", score=0.80, per_item_scores=[0.80], n_items=1)
    cur = EvalResult(metric="accuracy", score=0.79, per_item_scores=[0.79], n_items=1)
    bp = base.to_json(tmp_path / "baseline.json")
    cp = cur.to_json(tmp_path / "current.json")
    # 0.005 threshold means 1pp drop fails
    rc = main(["eval-regress", str(bp), str(cp), "--threshold", "0.005"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "[FAIL]" in out


def test_eval_regress_missing_file_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["eval-regress", "/nope/a.json", "/nope/b.json"])
    assert rc == 2
    assert "File not found" in capsys.readouterr().err
