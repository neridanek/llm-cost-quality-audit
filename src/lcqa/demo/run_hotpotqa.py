"""HotpotQA demo runner — baseline + optimized + regression check.

Default mode is `--mock`, which runs the deterministic mock pipelines so
`make demo` works offline. With `OPENAI_API_KEY` set and `--mode real`,
runs the real OpenAI baseline + optimized stack (Day 5-6 wiring).

Outputs (under `eval_results/`):
    baseline_records.json      — CostAnalyzer dump
    optimized_records.json
    baseline_faithfulness.json — EvalResult
    optimized_faithfulness.json
    baseline_accuracy.json
    optimized_accuracy.json
    regression_report.md       — human-readable summary

Usage:
    python -m lcqa.demo.run_hotpotqa --mock --limit 50
    python -m lcqa.demo.run_hotpotqa --mode real --limit 100  # needs OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from lcqa.cost.analyzer import CostAnalyzer
from lcqa.demo.hotpotqa import load_hotpotqa, parse_hotpotqa
from lcqa.demo.pipeline import MockPipeline, OptimizedMockPipeline, Pipeline
from lcqa.eval.accuracy import score_accuracy
from lcqa.eval.faithfulness import score_faithfulness
from lcqa.eval.regression import RegressionCheck, RegressionConfig
from lcqa.eval.result import EvalCase, EvalResult


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the HotpotQA demo benchmark.")
    p.add_argument("--mode", choices=["mock", "real"], default="mock")
    p.add_argument("--limit", type=int, default=20, help="Number of HotpotQA cases (default: 20)")
    p.add_argument(
        "--levels",
        nargs="+",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Filter by HotpotQA difficulty",
    )
    p.add_argument("--data", type=Path, help="Explicit HotpotQA JSON path (skips download)")
    p.add_argument("--output", type=Path, default=Path("eval_results"))
    p.add_argument("--regression-threshold", type=float, default=0.05)
    return p


def _load_cases(args: argparse.Namespace) -> list[EvalCase]:
    raw = load_hotpotqa(path=args.data)
    return parse_hotpotqa(
        raw,
        limit=args.limit,
        levels=tuple(args.levels) if args.levels else None,
    )


def _build_pipelines(mode: str) -> tuple[Pipeline, Pipeline]:
    if mode == "mock":
        return MockPipeline(name="baseline-mock"), OptimizedMockPipeline()
    if mode == "real":
        from lcqa.demo.pipeline_openai import OpenAIBaseline, OpenAIOptimized
        return OpenAIBaseline(), OpenAIOptimized()
    raise ValueError(f"Unknown mode {mode!r}")


def _run_pipeline(
    pipeline: Pipeline,
    cases: list[EvalCase],
) -> tuple[CostAnalyzer, list[EvalCase]]:
    """Run pipeline over cases. Returns (cost analyzer, cases with `answer` filled)."""
    analyzer = CostAnalyzer()
    answered: list[EvalCase] = []
    for case in cases:
        out = pipeline.run(case)
        analyzer.add(out.cost_record)
        answered.append(
            EvalCase(
                question=case.question,
                answer=out.answer,
                contexts=out.retrieved_contexts or case.contexts,
                expected_answer=case.expected_answer,
                metadata={**case.metadata, "pipeline": pipeline.name},
            )
        )
    return analyzer, answered


def _score(cases: list[EvalCase], *, real_scorer: bool = False) -> tuple[EvalResult, EvalResult]:
    """Score faithfulness + accuracy (exact match). Returns (faithfulness, accuracy).

    `real_scorer=True` uses the Ragas LLM-judge (requires OPENAI_API_KEY + lcqa[eval]).
    `real_scorer=False` uses a fast keyword-overlap proxy for offline mock runs.
    """
    scorer = None if real_scorer else _keyword_overlap_scorer
    f = score_faithfulness(cases, scorer=scorer)
    a = score_accuracy(cases)
    return f, a


def _keyword_overlap_scorer(case: EvalCase) -> float:
    """Toy faithfulness proxy: fraction of answer words appearing in contexts."""
    if not case.contexts:
        return 0.0
    answer_words = {w.lower().strip(".,?!") for w in case.answer.split() if len(w) > 2}
    if not answer_words:
        return 0.0
    blob = " ".join(case.contexts).lower()
    overlap = sum(1 for w in answer_words if w in blob)
    return overlap / len(answer_words)


def _write_regression_report(
    output_dir: Path,
    cost_baseline: float,
    cost_optimized: float,
    f_check: RegressionCheck,
    a_check: RegressionCheck,
) -> Path:
    cost_delta_pct = (
        (cost_optimized - cost_baseline) / cost_baseline * 100 if cost_baseline > 0 else 0.0
    )
    report = output_dir / "regression_report.md"
    report.write_text(
        "# LCQA Demo — Regression Report\n\n"
        "## Cost\n\n"
        f"- Baseline:  ${cost_baseline:.4f}\n"
        f"- Optimized: ${cost_optimized:.4f}\n"
        f"- Delta:     {cost_delta_pct:+.1f}%\n\n"
        "## Quality\n\n"
        f"### Faithfulness\n\n"
        f"- Baseline:  {f_check.baseline_score:.4f}\n"
        f"- Optimized: {f_check.current_score:.4f}\n"
        f"- Delta:     {f_check.delta:+.4f} (threshold: {f_check.threshold})\n"
        f"- Status:    {'PASS' if f_check.passed else 'FAIL'}\n\n"
        f"### Accuracy (exact match)\n\n"
        f"- Baseline:  {a_check.baseline_score:.4f}\n"
        f"- Optimized: {a_check.current_score:.4f}\n"
        f"- Delta:     {a_check.delta:+.4f} (threshold: {a_check.threshold})\n"
        f"- Status:    {'PASS' if a_check.passed else 'FAIL'}\n"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Mode: {args.mode}  |  Limit: {args.limit}  |  Levels: {args.levels or 'all'}")
    cases = _load_cases(args)
    print(f"Loaded {len(cases)} cases.\n")

    baseline_pipe, optimized_pipe = _build_pipelines(args.mode)

    print(f"Running baseline ({baseline_pipe.name})...")
    baseline_costs, baseline_answered = _run_pipeline(baseline_pipe, cases)
    print(f"  total cost: ${baseline_costs.total_cost():.4f}")

    print(f"Running optimized ({optimized_pipe.name})...")
    optimized_costs, optimized_answered = _run_pipeline(optimized_pipe, cases)
    print(f"  total cost: ${optimized_costs.total_cost():.4f}\n")

    baseline_costs.to_json(args.output / "baseline_records.json")
    optimized_costs.to_json(args.output / "optimized_records.json")

    real_scorer = args.mode == "real"
    f_base, a_base = _score(baseline_answered, real_scorer=real_scorer)
    f_opt, a_opt = _score(optimized_answered, real_scorer=real_scorer)

    f_base.to_json(args.output / "baseline_faithfulness.json")
    f_opt.to_json(args.output / "optimized_faithfulness.json")
    a_base.to_json(args.output / "baseline_accuracy.json")
    a_opt.to_json(args.output / "optimized_accuracy.json")

    config = RegressionConfig(threshold=args.regression_threshold)
    f_check = RegressionCheck.run(f_base, f_opt, config)
    a_check = RegressionCheck.run(a_base, a_opt, config)

    report_path = _write_regression_report(
        args.output,
        baseline_costs.total_cost(),
        optimized_costs.total_cost(),
        f_check,
        a_check,
    )

    print("=== Summary ===")
    print(f"Faithfulness: {f_check.baseline_score:.4f} → {f_check.current_score:.4f} "
          f"({f_check.delta:+.4f}) [{('PASS' if f_check.passed else 'FAIL')}]")
    print(f"Accuracy:     {a_check.baseline_score:.4f} → {a_check.current_score:.4f} "
          f"({a_check.delta:+.4f}) [{('PASS' if a_check.passed else 'FAIL')}]")
    print(f"Cost:         ${baseline_costs.total_cost():.4f} → ${optimized_costs.total_cost():.4f}")
    print(f"\nReport: {report_path}")

    return 0 if f_check.passed and a_check.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
