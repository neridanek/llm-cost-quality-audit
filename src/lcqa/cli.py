"""LCQA command-line interface.

Subcommands:

- `lcqa version` — toolkit version + pricing table last-updated date
- `lcqa list-models` — print pricing table
- `lcqa cost-summary <records.json>` — summarize a saved CostAnalyzer JSON dump
- `lcqa eval-regress <baseline.json> <current.json>` — CI-friendly regression check
  (exits 1 if regression beyond threshold, 0 otherwise)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from lcqa import __version__
from lcqa.cost.pricing import PRICING_LAST_UPDATED, list_models
from lcqa.eval.regression import RegressionCheck, RegressionConfig
from lcqa.eval.result import EvalResult


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lcqa", description="LLM Cost & Quality Audit toolkit.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("version", help="Print toolkit + pricing table version")

    list_parser = sub.add_parser("list-models", help="Print known provider:model pricing")
    list_parser.add_argument("--provider", help="Filter by provider (openai/anthropic/mistral)")

    cs_parser = sub.add_parser("cost-summary", help="Summarize a CostAnalyzer JSON dump")
    cs_parser.add_argument("records", type=Path, help="Path to records.json (CostAnalyzer.to_json output)")

    er_parser = sub.add_parser(
        "eval-regress",
        help="Compare two EvalResult JSON files; exit 1 on regression beyond threshold",
    )
    er_parser.add_argument("baseline", type=Path, help="Baseline EvalResult JSON")
    er_parser.add_argument("current", type=Path, help="Current EvalResult JSON")
    er_parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Max allowed score drop (default: 0.05)",
    )

    return parser


def _cmd_version() -> int:
    print(f"lcqa {__version__}")
    print(f"pricing table last updated: {PRICING_LAST_UPDATED}")
    return 0


def _cmd_list_models(provider: str | None) -> int:
    rows = list_models(provider=provider)
    if not rows:
        print(f"No models found for provider={provider!r}", file=sys.stderr)
        return 1
    width_p = max(len(r.provider) for r in rows)
    width_m = max(len(r.model) for r in rows)
    print(f"{'provider':<{width_p}}  {'model':<{width_m}}  input/1M  output/1M  cached/1M")
    print(f"{'-' * width_p}  {'-' * width_m}  --------  ---------  ---------")
    for r in sorted(rows, key=lambda x: (x.provider, x.model)):
        cached = f"${r.cached_input_per_1m:>7.3f}" if r.cached_input_per_1m is not None else "       —"
        print(
            f"{r.provider:<{width_p}}  {r.model:<{width_m}}  "
            f"${r.input_per_1m:>6.3f}  ${r.output_per_1m:>7.3f}  {cached}"
        )
    return 0


def _cmd_cost_summary(records_path: Path) -> int:
    if not records_path.exists():
        print(f"File not found: {records_path}", file=sys.stderr)
        return 2
    payload = json.loads(records_path.read_text())
    summary = payload.get("summary", {})
    print(f"Records: {summary.get('total_requests', 0)}")
    print(f"Total cost: ${summary.get('total_cost_usd', 0):.4f}")
    print()
    by_model = summary.get("by_model", [])
    if by_model:
        print("By model:")
        for row in by_model:
            print(
                f"  {row['provider']}:{row['model']}  "
                f"requests={row['requests']}  "
                f"cost=${row['total_cost_usd']:.4f}  "
                f"in={row['input_tokens']}  out={row['output_tokens']}"
            )
        print()
    by_tag = summary.get("by_tag", [])
    if by_tag:
        print("By tag:")
        for row in by_tag:
            tag = row.get("tag") or "(untagged)"
            print(
                f"  {tag}  requests={row['requests']}  "
                f"cost=${row['total_cost_usd']:.4f}"
            )
    return 0


def _cmd_eval_regress(baseline_path: Path, current_path: Path, threshold: float) -> int:
    for p in (baseline_path, current_path):
        if not p.exists():
            print(f"File not found: {p}", file=sys.stderr)
            return 2
    baseline = EvalResult.from_json(baseline_path)
    current = EvalResult.from_json(current_path)
    check = RegressionCheck.run(baseline, current, RegressionConfig(threshold=threshold))
    status = "PASS" if check.passed else "FAIL"
    sign = "+" if check.delta >= 0 else ""
    print(
        f"[{status}] {check.metric}: "
        f"baseline={check.baseline_score:.4f}  "
        f"current={check.current_score:.4f}  "
        f"delta={sign}{check.delta:.4f}  "
        f"threshold={check.threshold:.4f}"
    )
    return 0 if check.passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    match args.command:
        case "version":
            return _cmd_version()
        case "list-models":
            return _cmd_list_models(args.provider)
        case "cost-summary":
            return _cmd_cost_summary(args.records)
        case "eval-regress":
            return _cmd_eval_regress(args.baseline, args.current, args.threshold)
        case _:  # pragma: no cover — argparse rejects unknown commands first
            parser.print_help()
            return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
