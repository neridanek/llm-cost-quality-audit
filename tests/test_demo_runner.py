"""Demo runner + mock pipeline tests."""

from __future__ import annotations

from pathlib import Path

from lcqa.demo.pipeline import MockPipeline, OptimizedMockPipeline
from lcqa.demo.run_hotpotqa import main as runner_main
from lcqa.eval.result import EvalCase

FIXTURE = Path(__file__).parent / "fixtures" / "hotpotqa_sample.json"


def _case() -> EvalCase:
    return EvalCase(
        question="When was Linux created?",
        answer="",
        contexts=[
            "Linux is a family of open-source operating systems. The Linux kernel was created in 1991 by Linus Torvalds."
        ],
        expected_answer="1991",
    )


def test_mock_pipeline_returns_output() -> None:
    pipe = MockPipeline()
    out = pipe.run(_case())
    assert out.answer
    assert out.cost_record.input_tokens > 0
    assert out.cost_record.output_tokens > 0
    assert out.cost_record.tag == "mock:rag-answer"


def test_mock_pipeline_deterministic_cost() -> None:
    pipe1 = MockPipeline(seed=7)
    pipe2 = MockPipeline(seed=7)
    out1 = pipe1.run(_case())
    out2 = pipe2.run(_case())
    assert out1.cost_record.input_tokens == out2.cost_record.input_tokens
    assert out1.cost_record.output_tokens == out2.cost_record.output_tokens


def test_optimized_pipeline_cheaper_than_baseline() -> None:
    base = MockPipeline()
    opt = OptimizedMockPipeline()
    case = _case()
    base_out = base.run(case)
    opt_out = opt.run(case)
    # Optimized routes to gpt-4o-mini AND reduces input tokens AND uses cache
    assert opt_out.cost_record.cost() < base_out.cost_record.cost()


def test_optimized_pipeline_uses_mini_model() -> None:
    out = OptimizedMockPipeline().run(_case())
    assert out.cost_record.model == "gpt-4o-mini"
    assert out.cost_record.cached_input_tokens > 0


def test_runner_end_to_end_mock_mode(tmp_path: Path) -> None:
    rc = runner_main(
        [
            "--mode",
            "mock",
            "--limit",
            "3",
            "--data",
            str(FIXTURE),
            "--output",
            str(tmp_path),
            "--regression-threshold",
            "1.0",  # high threshold so test passes regardless of mock heuristic noise
        ]
    )
    assert rc == 0
    expected_files = {
        "baseline_records.json",
        "optimized_records.json",
        "baseline_faithfulness.json",
        "optimized_faithfulness.json",
        "baseline_accuracy.json",
        "optimized_accuracy.json",
        "regression_report.md",
    }
    actual = {p.name for p in tmp_path.iterdir()}
    assert expected_files.issubset(actual)


def test_runner_writes_readable_regression_report(tmp_path: Path) -> None:
    runner_main(
        [
            "--mode",
            "mock",
            "--limit",
            "3",
            "--data",
            str(FIXTURE),
            "--output",
            str(tmp_path),
            "--regression-threshold",
            "1.0",
        ]
    )
    report = (tmp_path / "regression_report.md").read_text()
    assert "Faithfulness" in report
    assert "Accuracy" in report
    assert "Cost" in report
