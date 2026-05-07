# ADR-0003 — Cost breakdown methodology: per-request attribution

**Status:** accepted (2026-05-01)

## Context

Two ways to attribute LLM cost:
- **Aggregate** — total monthly bill ÷ feature mix estimate
- **Per-request** — one record per LLM call, summed

Audit clients want to know which use case is bleeding money — not just the total.

## Decision

Per-request `RequestRecord` is the unit of measurement. Each record carries provider, model, input/output/cached token counts, latency, and an optional `tag` for use-case attribution.

## Alternatives

- **Aggregate-only** — easier first-pass triage, but useless for "should we route FAQ-tier queries to a smaller model?" decisions.
- **Provider-supplied billing** — OpenAI/Anthropic publish usage data, but neither lets you tag a request `tag="rag-answer"` vs `tag="rerank"`. We need our own labels.

## Consequences

- Users instrument their pipeline to emit `RequestRecord` per LLM call. Boilerplate, but unavoidable for tagged attribution.
- Token counting is offline (tiktoken) for OpenAI. Anthropic/Mistral approximated with a documented ~5% error band on English text. See `tokens.py`.
- Pricing tables are versioned in `pricing.py` (ADR-0005).
