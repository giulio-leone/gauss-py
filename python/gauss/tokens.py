"""Token counting and cost estimation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


# ─── Runtime Pricing Override ────────────────────────────────────────

@dataclass
class ModelPricing:
    """Per-token pricing for a model (USD per token)."""

    input_per_token: float
    output_per_token: float
    reasoning_per_token: float | None = None
    cache_read_per_token: float | None = None
    cache_creation_per_token: float | None = None


_pricing_overrides: dict[str, ModelPricing] = {}


def set_pricing(model: str, pricing: ModelPricing) -> None:
    """Set custom pricing for a model (overrides built-in Rust pricing).

    Example::

        set_pricing("my-custom-model", ModelPricing(
            input_per_token=0.000003,
            output_per_token=0.000015,
        ))
    """
    _pricing_overrides[model] = pricing


def get_pricing(model: str) -> ModelPricing | None:
    """Get custom pricing for a model, if set."""
    return _pricing_overrides.get(model)


def clear_pricing() -> None:
    """Clear all custom pricing overrides."""
    _pricing_overrides.clear()


# ─── Token Counting ──────────────────────────────────────────────────

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


# ─── Cost Estimation ─────────────────────────────────────────────────

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
    from gauss._types import CostEstimate

    # Check for SDK-level pricing override first
    override = _pricing_overrides.get(model)
    if override:
        input_cost = input_tokens * override.input_per_token
        output_cost = output_tokens * override.output_per_token
        reasoning_cost = (reasoning_tokens or 0) * (override.reasoning_per_token or override.output_per_token)
        cache_read_cost = (cache_read_tokens or 0) * (override.cache_read_per_token or override.input_per_token * 0.5)
        cache_creation_cost = (cache_creation_tokens or 0) * (override.cache_creation_per_token or override.input_per_token * 1.25)

        return CostEstimate(
            model=model,
            normalized_model=model,
            currency="USD",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens or 0,
            cache_read_tokens=cache_read_tokens or 0,
            cache_creation_tokens=cache_creation_tokens or 0,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            reasoning_cost_usd=reasoning_cost,
            cache_read_cost_usd=cache_read_cost,
            cache_creation_cost_usd=cache_creation_cost,
            total_cost_usd=input_cost + output_cost + reasoning_cost + cache_read_cost + cache_creation_cost,
        )

    # Fall back to Rust core pricing
    from gauss._native import estimate_cost as _native

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
