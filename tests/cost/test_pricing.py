"""Pricing table + cost computation tests."""

from __future__ import annotations

import pytest

from lcqa.cost.pricing import ModelPricing, get_pricing, list_models


def test_get_pricing_known_model() -> None:
    p = get_pricing("openai", "gpt-4o")
    assert p.input_per_1m == 2.50
    assert p.output_per_1m == 10.00
    assert p.cached_input_per_1m == 1.25


def test_get_pricing_unknown_model_raises() -> None:
    with pytest.raises(KeyError, match="No pricing for"):
        get_pricing("openai", "gpt-9000")


def test_list_models_filtered() -> None:
    openai_models = list_models(provider="openai")
    assert all(m.provider == "openai" for m in openai_models)
    assert any(m.model == "gpt-4o" for m in openai_models)


def test_list_models_all() -> None:
    all_models = list_models()
    providers = {m.provider for m in all_models}
    assert {"openai", "anthropic", "mistral"}.issubset(providers)


def test_cost_basic() -> None:
    p = ModelPricing(
        provider="test",
        model="t",
        input_per_1m=2.0,
        output_per_1m=10.0,
    )
    cost = p.cost(input_tokens=1_000_000, output_tokens=500_000)
    assert cost == pytest.approx(2.0 + 5.0)


def test_cost_with_cache() -> None:
    p = ModelPricing(
        provider="test",
        model="t",
        input_per_1m=10.0,
        output_per_1m=20.0,
        cached_input_per_1m=1.0,
    )
    cost = p.cost(input_tokens=1_000_000, output_tokens=500_000, cached_input_tokens=500_000)
    expected = (500_000 * 10.0 + 500_000 * 1.0 + 500_000 * 20.0) / 1_000_000
    assert cost == pytest.approx(expected)


def test_cost_cache_without_cache_pricing_raises() -> None:
    p = ModelPricing(provider="test", model="t", input_per_1m=2.0, output_per_1m=10.0)
    with pytest.raises(ValueError, match="no cached pricing"):
        p.cost(input_tokens=1_000, output_tokens=500, cached_input_tokens=100)


def test_cost_cached_exceeds_input_raises() -> None:
    p = ModelPricing(
        provider="test",
        model="t",
        input_per_1m=2.0,
        output_per_1m=10.0,
        cached_input_per_1m=0.5,
    )
    with pytest.raises(ValueError, match="exceeds"):
        p.cost(input_tokens=100, output_tokens=50, cached_input_tokens=200)


def test_anthropic_sonnet_realistic_request() -> None:
    p = get_pricing("anthropic", "claude-sonnet-4-6")
    cost = p.cost(input_tokens=10_000, output_tokens=2_000)
    # 10k * $3/1M + 2k * $15/1M = $0.03 + $0.03 = $0.06
    assert cost == pytest.approx(0.06)


def test_gpt_4o_mini_cheap_request() -> None:
    p = get_pricing("openai", "gpt-4o-mini")
    cost = p.cost(input_tokens=1_000, output_tokens=500)
    # 1k * $0.15/1M + 0.5k * $0.60/1M
    assert cost == pytest.approx((1_000 * 0.15 + 500 * 0.60) / 1_000_000)
