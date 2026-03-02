"""Token counting utilities."""

from __future__ import annotations

import json
from typing import Any


def count_tokens(text: str) -> int:
    """Count tokens in a text string.

    Example::

        n = count_tokens("Hello, world!")
    """
    from gauss._native import count_tokens as _native

    return _native(text)  # type: ignore[no-any-return]


def count_tokens_for_model(text: str, model: str) -> int:
    """Count tokens for a specific model.

    Example::

        n = count_tokens_for_model("Hello", "gpt-4o")
    """
    from gauss._native import count_tokens_for_model as _native

    return _native(text, model)  # type: ignore[no-any-return]


def count_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Count tokens in a message list.

    Example::

        n = count_message_tokens([{"role": "user", "content": "Hi"}])
    """
    from gauss._native import count_message_tokens as _native

    return _native(json.dumps(messages))  # type: ignore[no-any-return]


def get_context_window_size(model: str) -> int:
    """Get the context window size for a model.

    Example::

        size = get_context_window_size("gpt-4o")  # 128000
    """
    from gauss._native import get_context_window_size as _native

    return _native(model)  # type: ignore[no-any-return]


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    reasoning_tokens: int | None = None,
    cache_read_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
) -> "CostEstimate":
    """Estimate request cost from token usage for a model."""
    from gauss._native import estimate_cost as _native
    from gauss._types import CostEstimate

    raw = json.loads(
        _native(
            model,
            input_tokens,
            output_tokens,
            reasoning_tokens,
            cache_read_tokens,
            cache_creation_tokens,
        )
    )
    return CostEstimate(
        model=raw.get("model", model),
        normalized_model=raw.get("normalized_model", model),
        currency=raw.get("currency", "USD"),
        input_tokens=raw.get("input_tokens", input_tokens),
        output_tokens=raw.get("output_tokens", output_tokens),
        reasoning_tokens=raw.get("reasoning_tokens", reasoning_tokens or 0),
        cache_read_tokens=raw.get("cache_read_tokens", cache_read_tokens or 0),
        cache_creation_tokens=raw.get("cache_creation_tokens", cache_creation_tokens or 0),
        input_cost_usd=raw.get("input_cost_usd", 0.0),
        output_cost_usd=raw.get("output_cost_usd", 0.0),
        reasoning_cost_usd=raw.get("reasoning_cost_usd", 0.0),
        cache_read_cost_usd=raw.get("cache_read_cost_usd", 0.0),
        cache_creation_cost_usd=raw.get("cache_creation_cost_usd", 0.0),
        total_cost_usd=raw.get("total_cost_usd", 0.0),
    )
