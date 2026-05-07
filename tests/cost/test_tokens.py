"""Token counting tests (offline-only — no API calls)."""

from __future__ import annotations

import pytest

from lcqa.cost.tokens import count_tokens


def test_empty_string_returns_zero() -> None:
    assert count_tokens("openai", "gpt-4o", "") == 0


def test_openai_gpt4o_counts_consistently() -> None:
    text = "Hello, world!"
    n = count_tokens("openai", "gpt-4o", text)
    assert n > 0
    assert count_tokens("openai", "gpt-4o", text) == n  # deterministic


def test_openai_unknown_model_falls_back() -> None:
    # Unknown models route through o200k_base, so we still get a count.
    n = count_tokens("openai", "gpt-future-9000", "test")
    assert n > 0


def test_mistral_uses_approximation() -> None:
    n = count_tokens("mistral", "mistral-large-2", "Hello world")
    assert n > 0


def test_anthropic_uses_approximation_when_sdk_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    # The anthropic-SDK path is exercised via integration tests with credentials;
    # here we just verify the offline approximation returns a positive count.
    n = count_tokens("anthropic", "claude-sonnet-4-6", "Hello world")
    assert n > 0


def test_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        count_tokens("cohere", "command-r", "test")


def test_longer_text_more_tokens() -> None:
    short = count_tokens("openai", "gpt-4o", "Hi")
    long = count_tokens("openai", "gpt-4o", "Hi " * 1000)
    assert long > short
