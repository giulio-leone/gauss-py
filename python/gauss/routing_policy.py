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
class GovernanceRule:
    type: str
    tag: str | None = None
    provider: ProviderType | None = None


@dataclass
class GovernancePolicyPack:
    rules: list[GovernanceRule] = field(default_factory=list)


@dataclass
class RoutingPolicy:
    aliases: dict[str, list[RoutingCandidate]] = field(default_factory=dict)
    fallback_order: list[ProviderType] = field(default_factory=list)
    max_total_cost_usd: float | None = None
    max_requests_per_minute: int | None = None
    governance: GovernancePolicyPack | None = None


class RoutingPolicyError(ValueError):
    """Routing policy rejection."""


def governance_policy_pack(name: str) -> GovernancePolicyPack:
    if name == "enterprise_strict":
        return GovernancePolicyPack(
            rules=[
                GovernanceRule(type="allow_provider", provider=ProviderType.OPENAI),
                GovernanceRule(type="allow_provider", provider=ProviderType.ANTHROPIC),
                GovernanceRule(type="require_tag", tag="pci"),
            ]
        )
    if name == "eu_residency":
        return GovernancePolicyPack(
            rules=[
                GovernanceRule(type="deny_provider", provider=ProviderType.XAI),
                GovernanceRule(type="require_tag", tag="eu"),
            ]
        )
    if name == "cost_guarded":
        return GovernancePolicyPack(
            rules=[
                GovernanceRule(type="allow_provider", provider=ProviderType.OPENAI),
                GovernanceRule(type="allow_provider", provider=ProviderType.DEEPSEEK),
                GovernanceRule(type="require_tag", tag="cost-sensitive"),
            ]
        )
    raise RoutingPolicyError(f"unknown governance policy pack {name!r}")


def apply_governance_pack(
    policy: RoutingPolicy | None,
    pack_name: str,
) -> RoutingPolicy:
    pack = governance_policy_pack(pack_name)
    if policy is None:
        return RoutingPolicy(governance=GovernancePolicyPack(rules=list(pack.rules)))
    existing = list(policy.governance.rules) if policy.governance else []
    return RoutingPolicy(
        aliases=dict(policy.aliases),
        fallback_order=list(policy.fallback_order),
        max_total_cost_usd=policy.max_total_cost_usd,
        max_requests_per_minute=policy.max_requests_per_minute,
        governance=GovernancePolicyPack(rules=[*existing, *pack.rules]),
    )


def resolve_routing_target(
    policy: RoutingPolicy | None,
    provider: ProviderType,
    model: str,
    *,
    available_providers: list[ProviderType] | None = None,
    estimated_cost_usd: float | None = None,
    current_requests_per_minute: int | None = None,
    governance_tags: list[str] | None = None,
) -> tuple[ProviderType, str]:
    if estimated_cost_usd is not None:
        enforce_routing_cost_limit(policy, estimated_cost_usd)
    if current_requests_per_minute is not None:
        enforce_routing_rate_limit(policy, current_requests_per_minute)

    if policy is None:
        return provider, model
    candidates = policy.aliases.get(model)
    if candidates:
        if not available_providers:
            selected = max(candidates, key=lambda c: c.priority)
            enforce_routing_governance(policy, selected.provider, governance_tags)
            return selected.provider, selected.model
        available = set(available_providers)
        viable = [candidate for candidate in candidates if candidate.provider in available]
        if viable:
            selected = max(viable, key=lambda c: c.priority)
            enforce_routing_governance(policy, selected.provider, governance_tags)
            return selected.provider, selected.model

    fallback = resolve_fallback_provider(policy, available_providers or [])
    if fallback is not None and fallback != provider:
        enforce_routing_governance(policy, fallback, governance_tags)
        return fallback, model

    enforce_routing_governance(policy, provider, governance_tags)
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


def enforce_routing_rate_limit(
    policy: RoutingPolicy | None,
    requests_per_minute: int,
) -> None:
    if policy is None or policy.max_requests_per_minute is None:
        return
    if requests_per_minute > policy.max_requests_per_minute:
        raise RoutingPolicyError(f"routing policy rejected rate {requests_per_minute}")


def enforce_routing_governance(
    policy: RoutingPolicy | None,
    provider: ProviderType,
    governance_tags: list[str] | None,
) -> None:
    if policy is None or policy.governance is None:
        return

    rules = policy.governance.rules
    allow_providers = [
        rule.provider
        for rule in rules
        if rule.type == "allow_provider" and rule.provider is not None
    ]
    if allow_providers and provider not in allow_providers:
        raise RoutingPolicyError(f"routing policy governance rejected provider {provider.value}")

    for rule in rules:
        if rule.type == "deny_provider" and rule.provider == provider:
            raise RoutingPolicyError(f"routing policy governance rejected provider {provider.value}")
        if (
            rule.type == "require_tag"
            and governance_tags is not None
            and rule.tag is not None
            and rule.tag not in governance_tags
        ):
            raise RoutingPolicyError(f"routing policy governance missing tag {rule.tag}")
