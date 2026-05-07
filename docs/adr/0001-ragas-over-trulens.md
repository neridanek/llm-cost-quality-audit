# ADR-0001 — Ragas over TruLens for faithfulness scoring

**Status:** accepted (2026-05-01)

## Context

Faithfulness is the key hallucination-detection metric for RAG. Two production-grade options:
- **[Ragas](https://github.com/explodinggradients/ragas)** — research-backed metric definitions, batch-friendly, pre-1.0 API
- **[TruLens](https://github.com/truera/trulens)** — full app instrumentation, dashboard-first

## Decision

Default to Ragas `Faithfulness`. Expose a `scorer=` callable so users can swap in TruLens or a custom judge.

## Alternatives

- **TruLens** — better instrumentation, but pulls in heavy dependencies + a dashboard server. Audit work runs offline; users want a number and a CSV.
- **DeepEval `FaithfulnessMetric`** — viable, but less battle-tested than Ragas. We use DeepEval for accuracy@k instead (ADR-0002).
- **Custom LLM-as-judge** — reinventing the wheel; Ragas already publishes faithfulness reproducibility benchmarks.

## Consequences

- `ragas` + `langchain-openai` land in the `[eval]` extra (lazy-imported, optional).
- Anthropic / Mistral judge users need a custom `scorer=`; documented in `faithfulness.py`.
- Ragas pre-1.0 — pin a compatible range in `pyproject.toml`, watch breaking changes per release.
