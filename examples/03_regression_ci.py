"""Example 03 — Regression check for CI integration.

Run:
    python examples/03_regression_ci.py
    echo "exit code: $?"

What it shows:
- Loading two `EvalResult` JSON dumps (baseline + current)
- Running `RegressionCheck` with a custom threshold
- Exit code 0 if pass, 1 if regression beyond threshold

Drop this pattern into GitHub Actions to fail PRs that hurt quality. See
`.github/workflows/eval.yml` for a full workflow.
"""

from __future__ import annotations

import sys
from pathlib import Path

from lcqa.eval.regression import RegressionCheck, RegressionConfig
from lcqa.eval.result import EvalResult


def main() -> int:
    # In real CI, baseline lands from a previous run (artifact, S3, git-lfs).
    # Here we synthesize both for the example.
    baseline = EvalResult(
        metric="faithfulness",
        score=0.85,
        per_item_scores=[0.9, 0.8, 0.85],
        n_items=3,
    )
    current = EvalResult(
        metric="faithfulness",
        score=0.83,
        per_item_scores=[0.85, 0.80, 0.84],
        n_items=3,
    )

    out_dir = Path("eval_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline.to_json(out_dir / "example_03_baseline.json")
    current.to_json(out_dir / "example_03_current.json")

    check = RegressionCheck.run(
        baseline,
        current,
        RegressionConfig(threshold=0.05),  # tolerate up to 5pp drop
    )
    print(f"Metric:        {check.metric}")
    print(f"Baseline:      {check.baseline_score:.4f}")
    print(f"Current:       {check.current_score:.4f}")
    print(f"Delta:         {check.delta:+.4f} (threshold: {check.threshold})")
    print(f"Result:        {'PASS' if check.passed else 'FAIL'}")
    return 0 if check.passed else 1


if __name__ == "__main__":
    sys.exit(main())
