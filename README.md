# LCQA — LLM Cost & Quality Audit

> Toolkit dla production RAG systems: cost breakdown analyzer + eval harness (faithfulness, accuracy@k, latency p50/p95). Reproducible methodology, MIT license.

**Status:** alpha (pre-demo run). Concrete demo numbers below land after Day 4-6 HotpotQA benchmark — see [acceptance criteria](#acceptance-criteria).

## Why this exists

If you ship LLM features and burn $20k+/mo on inference, you're probably overpaying 50%+ — without a regression-proof eval harness telling you whether each "optimization" actually preserves quality. This toolkit is the reproducible methodology I use in audit engagements.

**It is not:**
- A new eval framework (wraps [Ragas](https://github.com/explodinggradients/ragas) + [DeepEval](https://github.com/confident-ai/deepeval))
- A "magic" cost reducer (no automatic prompt rewriting, no opaque "AI optimizer")
- An LLM router product (it measures, you decide)

**It is:**
- A pre-flight + post-flight audit kit you can run on your RAG system in <10 min
- A regression harness you can drop into CI (`pytest`-compatible)
- The same toolkit I run in `$12-15k` audit engagements — open-sourced because eval methodology is not the moat

## Headline numbers (HotpotQA demo, Day 4-6 deliverable)

> Filled after demo run. Placeholder format below shows the contract.

| Metric                | Baseline (GPT-4o, naive RAG) | Optimized | Delta            |
| --------------------- | ---------------------------- | --------- | ---------------- |
| Cost per 1k queries   | `$[X]`                       | `$[Y]`    | `-[Z]%`          |
| Faithfulness (Ragas)  | `[A]`                        | `[B]`     | `+/-[C]`         |
| Answer accuracy@1     | `[D]`                        | `[E]`     | `+/-[F]`         |
| Latency p50           | `[L50_b]s`                   | `[L50_o]s`| —                |
| Latency p95           | `[L95_b]s`                   | `[L95_o]s`| —                |

**Optimization stack measured:** prompt cache, context size reduction (re-ranking), model routing (FAQ-tier → mini model). Each layer measured independently — no hand-waving "we got 65% wins" without showing which knob did what.

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

## Acceptance criteria (reproducible run)

After Day 4-6 demo, this README will surface concrete numbers from a public, reproducible benchmark. Until then, the placeholder table above is a contract for what lands. The benchmark dataset is **HotpotQA dev split** — chosen because:

1. Public + permissive license (no fake "synthetic" numbers)
2. Multi-hop questions stress retrieval pipelines (real-world hard mode)
3. Existing baselines from research papers — sanity check on absolute scores

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

EU-based, fully remote, US + EU clients. Direct: [LinkedIn](https://www.linkedin.com/in/) — DM if you want help on a paid engagement, or open an issue if something here breaks.

## Related portfolio repos

- [lakehouse-audit](https://github.com/neridanek/lakehouse-audit) — Snowflake / Databricks AI-readiness toolkit
- [compliant-ai-reference](https://github.com/neridanek/compliant-ai-reference) — Engineering-layer reference for high-risk AI
