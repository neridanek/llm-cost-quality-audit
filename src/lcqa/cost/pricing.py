"""Provider pricing tables.

Prices are USD per 1M tokens, as published by providers. Update this file when
providers change pricing — there is no auto-fetch, intentionally (pricing is
audit input, not a moving target).

Last manual review: 2026-05. See ADR-005 for maintenance policy.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICING_LAST_UPDATED = "2026-05-03"


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """USD price per 1M tokens. cached_input_per_1m=None means provider has no cache discount."""

    provider: str
    model: str
    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float | None = None

    def cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> float:
        """Compute USD cost for a single request. cached_input_tokens defaults to 0."""
        if cached_input_tokens and self.cached_input_per_1m is None:
            raise ValueError(
                f"{self.provider}:{self.model} has no cached pricing; "
                f"got cached_input_tokens={cached_input_tokens}"
            )
        if cached_input_tokens > input_tokens:
            raise ValueError(
                f"cached_input_tokens ({cached_input_tokens}) exceeds "
                f"input_tokens ({input_tokens})"
            )
        regular_input = input_tokens - cached_input_tokens
        cached_rate = self.cached_input_per_1m or 0.0
        return (
            (regular_input * self.input_per_1m)
            + (cached_input_tokens * cached_rate)
            + (output_tokens * self.output_per_1m)
        ) / 1_000_000


def _key(provider: str, model: str) -> str:
    return f"{provider}:{model}"


PRICING_TABLE: dict[str, ModelPricing] = {
    # --- OpenAI ---
    _key("openai", "gpt-4o"): ModelPricing(
        provider="openai",
        model="gpt-4o",
        input_per_1m=2.50,
        output_per_1m=10.00,
        cached_input_per_1m=1.25,
    ),
    _key("openai", "gpt-4o-mini"): ModelPricing(
        provider="openai",
        model="gpt-4o-mini",
        input_per_1m=0.15,
        output_per_1m=0.60,
        cached_input_per_1m=0.075,
    ),
    _key("openai", "gpt-4-turbo"): ModelPricing(
        provider="openai",
        model="gpt-4-turbo",
        input_per_1m=10.00,
        output_per_1m=30.00,
    ),
    _key("openai", "o1"): ModelPricing(
        provider="openai",
        model="o1",
        input_per_1m=15.00,
        output_per_1m=60.00,
        cached_input_per_1m=7.50,
    ),
    _key("openai", "o1-mini"): ModelPricing(
        provider="openai",
        model="o1-mini",
        input_per_1m=3.00,
        output_per_1m=12.00,
        cached_input_per_1m=1.50,
    ),
    _key("openai", "text-embedding-3-small"): ModelPricing(
        provider="openai",
        model="text-embedding-3-small",
        input_per_1m=0.02,
        output_per_1m=0.0,
    ),
    _key("openai", "text-embedding-3-large"): ModelPricing(
        provider="openai",
        model="text-embedding-3-large",
        input_per_1m=0.13,
        output_per_1m=0.0,
    ),
    # --- Anthropic ---
    # Anthropic prompt-cache write is 1.25x base input; cache read is 0.1x base input.
    # We surface cache read pricing here; cache write priced separately at request site.
    _key("anthropic", "claude-opus-4-7"): ModelPricing(
        provider="anthropic",
        model="claude-opus-4-7",
        input_per_1m=5.00,
        output_per_1m=25.00,
        cached_input_per_1m=0.50,
    ),
    _key("anthropic", "claude-sonnet-4-6"): ModelPricing(
        provider="anthropic",
        model="claude-sonnet-4-6",
        input_per_1m=3.00,
        output_per_1m=15.00,
        cached_input_per_1m=0.30,
    ),
    _key("anthropic", "claude-haiku-4-5"): ModelPricing(
        provider="anthropic",
        model="claude-haiku-4-5",
        input_per_1m=1.00,
        output_per_1m=5.00,
        cached_input_per_1m=0.10,
    ),
    _key("anthropic", "claude-haiku-3-5"): ModelPricing(
        provider="anthropic",
        model="claude-haiku-3-5",
        input_per_1m=0.80,
        output_per_1m=4.00,
        cached_input_per_1m=0.08,
    ),
    # --- Mistral ---
    _key("mistral", "mistral-large-2"): ModelPricing(
        provider="mistral",
        model="mistral-large-2",
        input_per_1m=2.00,
        output_per_1m=6.00,
    ),
    _key("mistral", "mistral-small"): ModelPricing(
        provider="mistral",
        model="mistral-small",
        input_per_1m=0.20,
        output_per_1m=0.60,
    ),
    _key("mistral", "mistral-embed"): ModelPricing(
        provider="mistral",
        model="mistral-embed",
        input_per_1m=0.10,
        output_per_1m=0.0,
    ),
}


def get_pricing(provider: str, model: str) -> ModelPricing:
    """Lookup pricing for a provider/model pair. Raises KeyError if not found."""
    key = _key(provider, model)
    if key not in PRICING_TABLE:
        raise KeyError(
            f"No pricing for {key!r}. Known: {sorted(PRICING_TABLE.keys())}. "
            f"Add to PRICING_TABLE in pricing.py if you need a new model."
        )
    return PRICING_TABLE[key]


def list_models(provider: str | None = None) -> list[ModelPricing]:
    """List all known models, optionally filtered by provider."""
    if provider is None:
        return list(PRICING_TABLE.values())
    return [p for p in PRICING_TABLE.values() if p.provider == provider]
