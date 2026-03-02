"""Provider routing policy primitives for alias/fallback selection."""

from __future__ import annotations

from dataclasses import dataclass, field

from gauss._types import ProviderType


@dataclass
class RoutingCandidate:
    provider: ProviderType
    model: str
    priority: int = 0
    max_cost_usd: float | None = None


@dataclass
class RoutingPolicy:
    aliases: dict[str, list[RoutingCandidate]] = field(default_factory=dict)
    fallback_order: list[ProviderType] = field(default_factory=list)
    max_total_cost_usd: float | None = None


class RoutingPolicyError(ValueError):
    """Routing policy rejection."""


def resolve_routing_target(
    policy: RoutingPolicy | None,
    provider: ProviderType,
    model: str,
    *,
    available_providers: list[ProviderType] | None = None,
    estimated_cost_usd: float | None = None,
) -> tuple[ProviderType, str]:
    if estimated_cost_usd is not None:
        enforce_routing_cost_limit(policy, estimated_cost_usd)

    if policy is None:
        return provider, model
    candidates = policy.aliases.get(model)
    if candidates:
        selected = max(candidates, key=lambda c: c.priority)
        return selected.provider, selected.model

    fallback = resolve_fallback_provider(policy, available_providers or [])
    if fallback is not None and fallback != provider:
        return fallback, model

    return provider, model


def resolve_fallback_provider(
    policy: RoutingPolicy | None,
    available_providers: list[ProviderType],
) -> ProviderType | None:
    if policy is None or not policy.fallback_order or not available_providers:
        return None
    available = set(available_providers)
    for provider in policy.fallback_order:
        if provider in available:
            return provider
    return None


def enforce_routing_cost_limit(
    policy: RoutingPolicy | None,
    cost_usd: float,
) -> None:
    if policy is None or policy.max_total_cost_usd is None:
        return
    if cost_usd > policy.max_total_cost_usd:
        raise RoutingPolicyError(f"routing policy rejected cost {cost_usd}")
