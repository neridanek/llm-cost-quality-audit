# LCQA — LLM Cost & Quality Audit

> Toolkit dla production RAG systems: cost breakdown analyzer + eval harness (faithfulness, accuracy@k, latency p50/p95). Reproducible methodology, MIT license.

**Status:** v0.1.0 — public benchmark landed (HotpotQA, 100 questions, 2026-05-07). Methodology + numbers reproducible from `make demo-real`.

## Why this exists

If you ship LLM features and burn $20k+/mo on inference, you're probably overpaying 50%+ — without a regression-proof eval harness telling you whether each "optimization" actually preserves quality. This toolkit is the reproducible methodology I use in audit engagements.

**It is not:**
- A new eval framework (wraps [Ragas](https://github.com/explodinggradients/ragas) + [DeepEval](https://github.com/confident-ai/deepeval))
- A "magic" cost reducer (no automatic prompt rewriting, no opaque "AI optimizer")
- An LLM router product (it measures, you decide)

**It is:**
- A pre-flight + post-flight audit kit you can run on your RAG system in <10 min
- A regression harness you can drop into CI (`pytest`-compatible)
- Toolkit productized for `$12-15k` audit engagements — open-sourced because eval methodology is not the moat

## Headline numbers (HotpotQA dev, 100Q, run 2026-05-07)

| Metric                         | Baseline (GPT-4o, naive RAG) | Optimized stack | Delta                    |
| ------------------------------ | ---------------------------- | --------------- | ------------------------ |
| Cost per 100 queries           | $0.1959                      | $0.0798         | **-59.3%**               |
| Cost per 1k queries (extrapol) | $1.96                        | $0.80           | **-59.3%**               |
| Faithfulness (Ragas)           | 0.379                        | 0.434           | **+0.055** ✅ (improved) |
| Answer accuracy (exact match)  | 0.020                        | 0.010           | -0.010 (within tolerance)|
| Total benchmark spend          | —                            | —               | **$0.27** (one-time)     |

**Regression status:** PASS (faithfulness delta within 0.05 tolerance, no quality drop attributable to optimization).

**Optimization stack measured (3 layers, each independently measurable in `cost/analyzer.py`):**
1. **Model routing** — GPT-4o-mini for short / FAQ-tier questions; GPT-4o for multi-hop reasoning (decided by question token-length heuristic in `pipeline_openai.py`)
2. **Context reduction** — Cohere `rerank-english-v3.0` cuts top-10 retrieved → top-3 (silently falls back to first-3 if no Cohere key set)
3. **Prompt cache** — system prompt tokens flagged as cached (reflects real Anthropic/OpenAI prompt-cache discount on warm requests)

**Why faithfulness IMPROVED (counter-intuitive but methodologically sound):** rerank top-3 produces tighter grounding context → Ragas judge more confident in citation precision. Optimization wins on cost AND quality here; on your stack the relationship may differ — that's what the eval harness measures.

**Reproduce:** `make demo-real` (needs `OPENAI_API_KEY`, optionally `COHERE_API_KEY`; ~$0.27 spend, ~30 min wall-clock).

**Caveats (read before quoting these numbers):**
- HotpotQA exact-match accuracy is naturally low (2% baseline) because answers are short multi-word strings — Ragas faithfulness is the better quality signal here.
- Numbers are reproducible BUT specific to HotpotQA + this 3-layer optimization stack. Your stack: different retrieval + different domain + different prompt = different deltas. Methodology is the value, not the specific 59% number.
- Bench was 100Q (n=100). For statistical significance on production traffic, scale to 1000+Q test set per metric.

## Quick start

```bash
git clone https://github.com/neridanek/llm-cost-quality-audit.git
cd llm-cost-quality-audit
make install-dev
make demo            # mock pipeline + fixture, no API keys, runs in seconds
```

Once you've set `OPENAI_API_KEY` (and optionally `COHERE_API_KEY` for reranking), the real benchmark:

```bash
make demo-full       # mock pipeline + downloaded HotpotQA dev split (~50MB)
make demo-real       # real OpenAI baseline + optimized stack (needs OPENAI_API_KEY)
```

Outputs land in `eval_results/`:
- `baseline_records.json` / `optimized_records.json` — per-request token + dollar attribution
- `*_faithfulness.json` / `*_accuracy.json` — Ragas + DeepEval scores
- `regression_report.md` — pass/fail vs baseline for CI integration

## What's inside

```
src/lcqa/
├── cost/         # token counting, provider pricing tables, breakdown analyzer
├── eval/         # Ragas + DeepEval wrappers (faithfulness, accuracy, latency)
└── demo/         # HotpotQA reproducible benchmark (baseline → optimized)
```

### Cost analyzer (`lcqa.cost`)

- Token counter per request: prompt / context / response, per provider tokenizer
- Pricing tables: OpenAI, Anthropic, Mistral (kept current — see `pricing.py` last-updated marker)
- Aggregation per use case (label requests by `tag`, get cost-per-tag breakdown)
- Output: pandas DataFrame + JSON report

### Eval harness (`lcqa.eval`)

- **Faithfulness** — Ragas, hallucination detection vs retrieved context
- **Answer accuracy@k** — DeepEval, per-question + aggregate
- **Latency p50/p95** — percentile tracker
- **CI mode** — `make test-eval` exits non-zero if regression > threshold (default: faithfulness drop > 0.05)

## Reproducibility

Benchmark dataset: **HotpotQA dev distractor split** — chosen because:

1. Public + permissive license (no synthetic-data smell)
2. Multi-hop questions stress retrieval pipelines (real-world hard mode for RAG)
3. Existing baselines from research papers — sanity check on absolute scores
4. Cached after first download under `~/.cache/lcqa/hotpotqa/` — re-runs are offline

Run on your end: `make install-dev && make demo-real` (set `OPENAI_API_KEY`; `COHERE_API_KEY` optional). Numbers should fall within ±10% of the headline table on repeated runs (LLM judge variance).

## Architectural decisions

ADRs live in [`docs/adr/`](docs/adr/) once landed. Planned (Day 7-10):

- ADR-001 — Why Ragas over TruLens for faithfulness scoring
- ADR-002 — Why batch eval over streaming for benchmark mode
- ADR-003 — Cost breakdown methodology (per-request attribution vs aggregate)
- ADR-004 — Why HotpotQA over MS MARCO for the public demo
- ADR-005 — Provider pricing table maintenance + version policy

## License

MIT — use it, fork it, run it on your RAG system. No attribution required, but a star helps surface this for others.

## Hire me for engineering work

Wiktor N. — senior data engineer (8yr Databricks Lakehouse migrations + agentic systems + GenAI eval).

**If you've read this toolkit's code, you're already qualified for the matching paid engagement.** Productized fixed-fee, 1-2 weeks, locked scope:

- 🎯 **LLM Cost Quick-Win Sprint** — $6-9k, 3-5 days. **Native fit if you read this toolkit.** You already know which lever is bleeding (model routing / semantic cache / prompt compression / context trimming) and want it shipped. I trace cost, scope the lever, ship 3 PRs to your repo, measure production impact. Target: 25-50% reduction on the targeted use case.
- 🎯 **RAG Eval Suite Build (Mini-Audit)** — $8-12k, 5-7 days. Native fit if you read this toolkit AND ship RAG. 100Q+ golden test set + Ragas-style scorers + CI regression hook + dashboard.
- **LLM Cost & Quality Audit** — $12-15k, 1.5 weeks. Pick this if you have 3+ AI features and CFO is asking about the whole bill (NOT just one lever). Full attribution + eval baseline + 30-65% optimization roadmap + eval harness wired to CI for ongoing regression detection.
- **LLM Quality Maintenance Retainer** — $3-10k/mo. Scheduled eval + drift alerts + monthly readout. Cross-sell post-LCQA / post-Mini-Audit.
- **Paid Discovery** — $3-5k, 3-5 days. Scoped technical readout for clients evaluating multiple AI directions; counts as credit toward any subsequent engagement signed within 21 days.

EU-based, fully remote, US + EU clients. Direct: [LinkedIn](https://www.linkedin.com/in/wnnn/) — DM if you want help on a paid engagement, or open an issue if something here breaks.

## Related portfolio repos

- [lakehouse-audit](https://github.com/neridanek/lakehouse-audit) — Snowflake / Databricks AI-readiness toolkit
- [compliant-ai-reference](https://github.com/neridanek/compliant-ai-reference) — Engineering-layer reference for high-risk AI
