"""Token counting per provider.

OpenAI uses tiktoken (exact). Anthropic and Mistral are approximated via
tiktoken `o200k_base` — error band is typically <5% for English text. For
audit-grade accuracy on Anthropic, install the `anthropic` SDK and the
counter routes through `client.messages.count_tokens` automatically.

See ADR-003 for cost breakdown methodology decisions.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import tiktoken

logger = logging.getLogger(__name__)

# Map provider:model → tiktoken encoding. Models not in the map fall back to o200k_base.
_OPENAI_ENCODINGS: dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
}


@lru_cache(maxsize=8)
def _encoding(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def _openai_token_count(model: str, text: str) -> int:
    enc_name = _OPENAI_ENCODINGS.get(model, "o200k_base")
    return len(_encoding(enc_name).encode(text))


def _approximate_token_count(text: str) -> int:
    """Fallback for non-OpenAI providers using o200k_base."""
    return len(_encoding("o200k_base").encode(text))


def _anthropic_token_count(model: str, text: str) -> int:
    """Use anthropic SDK if installed + auth'd, else approximate via o200k_base.

    Approximation error band is ~5% for English text (see ADR-003). Exact
    counts require the anthropic SDK plus a valid API key — the SDK calls
    `client.messages.count_tokens`, which is a network request.

    On any failure (missing SDK, missing key, network/API error) we log a
    warning and fall back to the approximation so audit numbers remain
    well-defined; check logs to confirm whether you got exact or approximate
    counts.
    """
    try:
        import anthropic
    except ImportError:
        logger.warning(
            "anthropic SDK not installed; using o200k_base approximation "
            "(~5%% error). Install `pip install anthropic` for exact counts."
        )
        return _approximate_token_count(text)

    try:
        client = anthropic.Anthropic()
        result = client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": text}],
        )
        return int(result.input_tokens)
    except anthropic.APIError as e:
        logger.warning(
            "anthropic count_tokens failed (%s); using o200k_base approximation "
            "(~5%% error). Verify ANTHROPIC_API_KEY and network access.",
            e,
        )
        return _approximate_token_count(text)


def count_tokens(provider: str, model: str, text: str) -> int:
    """Count tokens for `text` under (provider, model) tokenization.

    Raises ValueError on unknown provider.
    """
    if not text:
        return 0
    match provider:
        case "openai":
            return _openai_token_count(model, text)
        case "anthropic":
            return _anthropic_token_count(model, text)
        case "mistral":
            return _approximate_token_count(text)
        case _:
            raise ValueError(
                f"Unknown provider {provider!r}. Supported: openai, anthropic, mistral."
            )
