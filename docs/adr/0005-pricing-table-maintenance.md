# ADR-0005 — Provider pricing table maintenance + version policy

**Status:** accepted (2026-05-01)

## Context

OpenAI, Anthropic, and Mistral revise pricing every 3-9 months. Stale pricing in an audit toolkit is a credibility hit. We don't want to auto-fetch (network dependency, no stable price APIs from providers, breaks offline use).

## Decision

Pricing is a Python dict in `src/lcqa/cost/pricing.py` with a `PRICING_LAST_UPDATED` constant. Manual review cadence: every quarter, or on provider announcement. Each PR that updates pricing also bumps `LAST_UPDATED`. `lcqa version` surfaces the date so audit clients can tell data freshness.

## Alternatives

- **Auto-fetch from provider docs** — providers don't publish stable machine-readable feeds. HTML scraping is brittle. Debugging price-fetch breakages > doing audit work.
- **YAML/JSON config** — adds parsing dependency, worse IDE support. Code dict gives type-checking via `ModelPricing` dataclass.
- **CDN-hosted JSON we update** — adds infra ownership for a tiny payload.

## Consequences

- Users on an old version don't get new model pricing — but `LAST_UPDATED` makes that visible via `lcqa version`.
- Quarterly maintenance overhead: ~30 min to diff provider price pages and bump.
- Users adding new models contribute via PR using the [pricing-update issue template](../../.github/ISSUE_TEMPLATE/pricing-update.md). Encourages community ownership.
