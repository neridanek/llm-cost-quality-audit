# ADR-0002 — Batch eval over streaming for the demo benchmark

**Status:** accepted (2026-05-01)

## Context

Two execution modes for an eval harness:
- **Streaming** — score each request as it lands in the pipeline (online)
- **Batch** — collect a fixed test set, score offline (deterministic)

Production users want both. The Day 4-6 HotpotQA demo needs reproducible numbers in a fixed time budget.

## Decision

Batch mode is the default for `make demo`. Streaming is supported via the same `EvalCase` model — feed cases one at a time into `score_faithfulness([case])` for online scoring.

## Alternatives

- **Streaming-first** — closer to production telemetry, but the demo needs deterministic numbers we can put in a README headline. Streaming makes "we got 0.87 faithfulness" depend on traffic mix and timing.
- **Two separate harnesses** — duplication, divergent code paths, more to maintain.

## Consequences

- README headline numbers reproduce from a fixed dev split + seed.
- Streaming users pay per-call eval LLM cost — documented in cost methodology (ADR-0003).
- `RegressionCheck` is batch-mode only. Streaming regressions need a separate aggregation layer.
