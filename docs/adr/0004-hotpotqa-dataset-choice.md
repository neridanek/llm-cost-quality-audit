# ADR-0004 — HotpotQA over MS MARCO for the public demo

**Status:** proposed (validate Day 4-6)

## Context

Need a public RAG benchmark dataset to land headline numbers in the README. Two strong candidates:
- **[HotpotQA](https://hotpotqa.github.io/)** — multi-hop QA over Wikipedia, 113k questions, dev split = 7405
- **[MS MARCO](https://microsoft.github.io/msmarco/)** — web search, 1M passages, 100k+ queries

## Decision

HotpotQA dev distractor split for the demo run.

## Alternatives

- **MS MARCO** — larger and more web-search-realistic, but answers are less canonical and faithfulness scoring is noisier. Better fit for retrieval-only benchmarks (BEIR-style).
- **TriviaQA / Natural Questions** — single-hop, doesn't stress retrieval pipelines enough to show optimization wins.
- **Synthetic dataset** — fast to generate, but no external validity. A `$X` cost reduction on a synthetic test means nothing to a buyer.

## Consequences

- Multi-hop questions stress the retrieval pipeline — closer to real RAG workloads.
- Existing baselines from research papers — sanity check on absolute scores.
- License is permissive (CC BY-SA 4.0) — fine for public demos with attribution.
- Status remains *proposed* until Day 4-6 demo run validates the choice produces meaningful before/after numbers.
