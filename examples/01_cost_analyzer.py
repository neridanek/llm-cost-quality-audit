"""Example 01 — Cost analyzer with synthetic RAG traces.

Run:
    python examples/01_cost_analyzer.py

What it shows:
- Building `RequestRecord` per LLM call
- Tagging requests by use case (`rag-answer`, `rerank`, `embed`)
- Aggregating by tag/model
- Writing a JSON dump that the `lcqa cost-summary` CLI can reload

No API keys needed — token counts are simulated.
"""

from __future__ import annotations

import random
from pathlib import Path

from lcqa.cost.analyzer import CostAnalyzer, RequestRecord


def main() -> None:
    random.seed(42)
    analyzer = CostAnalyzer()

    # Simulate 100 RAG-answer queries on GPT-4o
    for _ in range(100):
        analyzer.add(
            RequestRecord(
                provider="openai",
                model="gpt-4o",
                input_tokens=random.randint(2000, 4000),
                output_tokens=random.randint(150, 400),
                cached_input_tokens=random.randint(800, 1500),
                latency_ms=random.uniform(800, 2500),
                tag="rag-answer",
            )
        )

    # Simulate 100 reranking calls on the cheaper mini model
    for _ in range(100):
        analyzer.add(
            RequestRecord(
                provider="openai",
                model="gpt-4o-mini",
                input_tokens=random.randint(800, 1500),
                output_tokens=random.randint(50, 100),
                latency_ms=random.uniform(200, 600),
                tag="rerank",
            )
        )

    # Simulate 1000 embedding calls
    for _ in range(1000):
        analyzer.add(
            RequestRecord(
                provider="openai",
                model="text-embedding-3-small",
                input_tokens=random.randint(50, 300),
                output_tokens=0,
                latency_ms=random.uniform(40, 120),
                tag="embed",
            )
        )

    print(f"Total records: {len(analyzer)}")
    print(f"Total cost: ${analyzer.total_cost():.4f}")
    print()

    print("=== By tag ===")
    print(analyzer.aggregate_by("tag").to_string(index=False))
    print()

    print("=== By model ===")
    print(analyzer.aggregate_by("provider", "model").to_string(index=False))

    out = Path("eval_results/example_01_records.json")
    analyzer.to_json(out)
    print(f"\nWrote {out} — try: lcqa cost-summary {out}")


if __name__ == "__main__":
    main()
