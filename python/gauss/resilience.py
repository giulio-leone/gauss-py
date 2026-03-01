"""Resilience patterns: fallback, circuit breaker, resilient provider."""

from __future__ import annotations

import json


def create_fallback_provider(provider_handles: list[int]) -> int:
    """Create a fallback provider that tries providers in order.

    Example::

        primary = Agent(provider=ProviderType.OPENAI).handle
        backup = Agent(provider=ProviderType.ANTHROPIC).handle
        fallback = create_fallback_provider([primary, backup])
    """
    from gauss._native import (
        create_fallback_provider as _native_create,  # type: ignore[import-not-found]
    )

    return _native_create(json.dumps(provider_handles))  # type: ignore[no-any-return]


def create_circuit_breaker(
    provider_handle: int,
    failure_threshold: int = 5,
    recovery_timeout_ms: int = 30000,
) -> int:
    """Create a circuit breaker wrapping a provider.

    Example::

        breaker = create_circuit_breaker(agent.handle, failure_threshold=3)
    """
    from gauss._native import (
        create_circuit_breaker as _native_create,  # type: ignore[import-not-found]
    )

    return _native_create(provider_handle, failure_threshold, recovery_timeout_ms)  # type: ignore[no-any-return]


def create_resilient_provider(
    primary_handle: int,
    fallback_handles: list[int] | None = None,
    circuit_breaker: bool = True,
) -> int:
    """Create a resilient provider with fallback and circuit breaker.

    Example::

        resilient = create_resilient_provider(
            primary.handle,
            [backup1.handle, backup2.handle],
            circuit_breaker=True,
        )
    """
    from gauss._native import (
        create_resilient_provider as _native_create,  # type: ignore[import-not-found]
    )

    return _native_create(  # type: ignore[no-any-return]
        primary_handle,
        json.dumps(fallback_handles or []),
        circuit_breaker,
    )
