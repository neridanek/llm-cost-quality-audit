# Contributing

Thanks for the interest. Two contribution modes are warmly welcomed; a third is discouraged.

## Welcomed

### 1. Pricing updates

Providers change prices. If you spot stale pricing in `src/lcqa/cost/pricing.py`:

1. Open an issue using the [pricing-update template](.github/ISSUE_TEMPLATE/pricing-update.md), or skip straight to a PR.
2. Update the relevant `ModelPricing(...)` entry.
3. Bump `PRICING_LAST_UPDATED` to today's ISO date.
4. Run `pytest tests/cost/` — pricing tests should still pass.

If you're adding a brand-new model: also add an entry to `_OPENAI_ENCODINGS` in `tokens.py` if it uses a non-default tokenizer.

### 2. New eval metrics

Faithfulness and accuracy cover the bulk of RAG audit work. If you have a metric that materially adds (e.g., context relevance, citation precision):

1. Open an issue describing what it measures and why a real audit needs it.
2. Add a `score_<metric>` function in `src/lcqa/eval/<metric>.py`, mirroring the shape of `score_faithfulness`.
3. Make heavy deps lazy-imported so the metric is opt-in.
4. Land tests using a custom scorer (no API keys required for CI).

## Discouraged

- **New eval framework wrappers** beyond Ragas + DeepEval. The point of this toolkit is methodology, not framework abstraction. If you want a fourth wrapper, write a `scorer=` callable in your own code — that's what the parameter is for.
- **Auto-fetch pricing scrapers**. See [ADR-0005](docs/adr/0005-pricing-table-maintenance.md). PRs to add this will be closed.
- **Adding tutorial-style sections to the README**. The README is for architectural decisions, not LLM 101.

## Local dev setup

```bash
git clone https://github.com/neridanek/llm-cost-quality-audit.git
cd llm-cost-quality-audit
make install-dev
make test
make lint
```

## Architectural decisions

Read [`docs/adr/`](docs/adr/) before sending non-trivial PRs. Decisions captured there are deliberate; if you disagree, propose a superseding ADR.
