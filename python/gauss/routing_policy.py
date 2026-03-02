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


def resolve_routing_target(
    policy: RoutingPolicy | None,
    provider: ProviderType,
    model: str,
) -> tuple[ProviderType, str]:
    if policy is None:
        return provider, model
    candidates = policy.aliases.get(model)
    if not candidates:
        return provider, model
    selected = max(candidates, key=lambda c: c.priority)
    return selected.provider, selected.model

