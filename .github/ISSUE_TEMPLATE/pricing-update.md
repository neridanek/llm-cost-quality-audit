---
name: Pricing update
about: Provider changed pricing — update the table
title: "[pricing] <provider> <model> price change"
labels: pricing
---

## What changed

- **Provider:** <openai / anthropic / mistral / new>
- **Model:** <e.g. gpt-4o>
- **Effective date:** <YYYY-MM-DD>
- **Old:** input=$X/1M, output=$Y/1M, cached=$Z/1M
- **New:** input=$X/1M, output=$Y/1M, cached=$Z/1M

## Source

Link to the provider's official pricing page or announcement post.

## Checklist

- [ ] Updated `src/lcqa/cost/pricing.py`
- [ ] Bumped `PRICING_LAST_UPDATED` constant
- [ ] Tests pass (`pytest tests/cost/`)
- [ ] If the model is new, added a corresponding entry in `_OPENAI_ENCODINGS` (in `tokens.py`) if it uses a non-default tokenizer
