"""Provider routing policy primitives for alias/fallback selection."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

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
    max_total_cost_usd: float | None = None
    max_requests_per_minute: int | None = None
    allowed_hours_utc: list[int] = field(default_factory=list)
    provider_weights: dict[ProviderType, int] = field(default_factory=dict)
    fallback_order: list[ProviderType] = field(default_factory=list)


@dataclass
class RoutingPolicy:
    aliases: dict[str, list[RoutingCandidate]] = field(default_factory=dict)
    fallback_order: list[ProviderType] = field(default_factory=list)
    max_total_cost_usd: float | None = None
    max_requests_per_minute: int | None = None
    allowed_hours_utc: list[int] = field(default_factory=list)
    provider_weights: dict[ProviderType, int] = field(default_factory=dict)
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
            ],
            max_total_cost_usd=0.15,
        )
    if name == "ops_business_hours":
        return GovernancePolicyPack(
            rules=[GovernanceRule(type="require_tag", tag="ops")],
            allowed_hours_utc=list(range(8, 19)),
        )
    if name == "balanced_mix":
        return GovernancePolicyPack(
            rules=[
                GovernanceRule(type="allow_provider", provider=ProviderType.OPENAI),
                GovernanceRule(type="allow_provider", provider=ProviderType.ANTHROPIC),
                GovernanceRule(type="require_tag", tag="balanced"),
            ],
            provider_weights={
                ProviderType.OPENAI: 60,
                ProviderType.ANTHROPIC: 40,
            },
        )
    if name == "rollout_canary":
        return GovernancePolicyPack(
            rules=[
                GovernanceRule(type="allow_provider", provider=ProviderType.OPENAI),
                GovernanceRule(type="allow_provider", provider=ProviderType.ANTHROPIC),
                GovernanceRule(type="require_tag", tag="rollout"),
            ],
            max_total_cost_usd=0.1,
            max_requests_per_minute=30,
            provider_weights={
                ProviderType.OPENAI: 70,
                ProviderType.ANTHROPIC: 30,
            },
            fallback_order=[ProviderType.OPENAI, ProviderType.ANTHROPIC],
        )
    if name == "rollout_strict":
        return GovernancePolicyPack(
            rules=[
                GovernanceRule(type="allow_provider", provider=ProviderType.OPENAI),
                GovernanceRule(type="allow_provider", provider=ProviderType.ANTHROPIC),
                GovernanceRule(type="require_tag", tag="rollout"),
                GovernanceRule(type="require_tag", tag="approved"),
            ],
            max_total_cost_usd=0.08,
            max_requests_per_minute=15,
            allowed_hours_utc=list(range(9, 18)),
            fallback_order=[ProviderType.OPENAI, ProviderType.ANTHROPIC],
        )
    raise RoutingPolicyError(f"unknown governance policy pack {name!r}")


def apply_governance_pack(
    policy: RoutingPolicy | None,
    pack_name: str,
) -> RoutingPolicy:
    pack = governance_policy_pack(pack_name)
    allowed_hours_utc = list(pack.allowed_hours_utc) if pack.allowed_hours_utc else list(policy.allowed_hours_utc) if policy else []
    provider_weights = (
        {**(policy.provider_weights if policy else {}), **pack.provider_weights}
        if pack.provider_weights
        else dict(policy.provider_weights) if policy else {}
    )
    fallback_order = (
        list(dict.fromkeys([*(policy.fallback_order if policy else []), *pack.fallback_order]))
        if pack.fallback_order
        else list(policy.fallback_order) if policy else []
    )
    max_total_cost_usd = pack.max_total_cost_usd if pack.max_total_cost_usd is not None else policy.max_total_cost_usd if policy else None
    max_requests_per_minute = (
        pack.max_requests_per_minute
        if pack.max_requests_per_minute is not None
        else policy.max_requests_per_minute if policy else None
    )
    if policy is None:
        return RoutingPolicy(
            fallback_order=fallback_order,
            max_total_cost_usd=max_total_cost_usd,
            max_requests_per_minute=max_requests_per_minute,
            allowed_hours_utc=allowed_hours_utc,
            provider_weights=provider_weights,
            governance=GovernancePolicyPack(rules=list(pack.rules)),
        )
    existing = list(policy.governance.rules) if policy.governance else []
    return RoutingPolicy(
        aliases=dict(policy.aliases),
        fallback_order=fallback_order,
        max_total_cost_usd=max_total_cost_usd,
        max_requests_per_minute=max_requests_per_minute,
        allowed_hours_utc=allowed_hours_utc,
        provider_weights=provider_weights,
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
    current_hour_utc: int | None = None,
    governance_tags: list[str] | None = None,
) -> tuple[ProviderType, str]:
    selected_provider, selected_model, _selected_by = _resolve_routing_decision(
        policy,
        provider,
        model,
        available_providers=available_providers,
        estimated_cost_usd=estimated_cost_usd,
        current_requests_per_minute=current_requests_per_minute,
        current_hour_utc=current_hour_utc,
        governance_tags=governance_tags,
    )
    return selected_provider, selected_model


def _resolve_routing_decision(
    policy: RoutingPolicy | None,
    provider: ProviderType,
    model: str,
    *,
    available_providers: list[ProviderType] | None = None,
    estimated_cost_usd: float | None = None,
    current_requests_per_minute: int | None = None,
    current_hour_utc: int | None = None,
    governance_tags: list[str] | None = None,
) -> tuple[ProviderType, str, str]:
    enforce_routing_time_window(policy, current_hour_utc if current_hour_utc is not None else dt.datetime.now(dt.UTC).hour)
    if estimated_cost_usd is not None:
        enforce_routing_cost_limit(policy, estimated_cost_usd)
    if current_requests_per_minute is not None:
        enforce_routing_rate_limit(policy, current_requests_per_minute)

    if policy is None:
        return provider, model, "direct"
    candidates = policy.aliases.get(model)
    if candidates:
        if not available_providers:
            selected = _select_weighted_candidate(policy, candidates)
            enforce_routing_governance(policy, selected.provider, governance_tags)
            return selected.provider, selected.model, f"alias:{model}"
        available = set(available_providers)
        viable = [candidate for candidate in candidates if candidate.provider in available]
        if viable:
            selected = _select_weighted_candidate(policy, viable)
            enforce_routing_governance(policy, selected.provider, governance_tags)
            return selected.provider, selected.model, f"alias:{model}"

    fallback = resolve_fallback_provider(policy, available_providers or [])
    if fallback is not None and fallback != provider:
        enforce_routing_governance(policy, fallback, governance_tags)
        return fallback, model, f"fallback:{fallback.value}"

    enforce_routing_governance(policy, provider, governance_tags)
    return provider, model, "direct"


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


def enforce_routing_time_window(
    policy: RoutingPolicy | None,
    hour_utc: int,
) -> None:
    if policy is None or not policy.allowed_hours_utc:
        return
    if hour_utc not in policy.allowed_hours_utc:
        raise RoutingPolicyError(f"routing policy rejected hour {hour_utc}")


def enforce_routing_governance(
    policy: RoutingPolicy | None,
    provider: ProviderType,
    governance_tags: list[str] | None,
) -> None:
    if policy is None or policy.governance is None:
        return

    rules = policy.governance.rules
    allow_providers = {
        rule.provider
        for rule in rules
        if rule.type == "allow_provider" and rule.provider is not None
    }
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


def _select_weighted_candidate(
    policy: RoutingPolicy,
    candidates: list[RoutingCandidate],
) -> RoutingCandidate:
    if len(candidates) == 1:
        return candidates[0]
    return max(
        candidates,
        key=lambda c: (policy.provider_weights.get(c.provider, 0), c.priority),
    )


def explain_routing_target(
    policy: RoutingPolicy | None,
    provider: ProviderType,
    model: str,
    *,
    available_providers: list[ProviderType] | None = None,
    estimated_cost_usd: float | None = None,
    current_requests_per_minute: int | None = None,
    current_hour_utc: int | None = None,
    governance_tags: list[str] | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    hour = current_hour_utc if current_hour_utc is not None else dt.datetime.now(dt.UTC).hour

    def fail(check: str, error: Exception) -> dict[str, Any]:
        message = str(error)
        checks.append({"check": check, "status": "failed", "detail": message})
        return {"ok": False, "checks": checks, "error": message}

    if policy is not None and policy.allowed_hours_utc:
        try:
            enforce_routing_time_window(policy, hour)
            checks.append({"check": "time_window", "status": "passed", "detail": f"hour={hour}"})
        except RoutingPolicyError as exc:
            return fail("time_window", exc)
    else:
        checks.append({"check": "time_window", "status": "skipped", "detail": "not configured"})

    if estimated_cost_usd is not None:
        try:
            enforce_routing_cost_limit(policy, estimated_cost_usd)
            checks.append(
                {
                    "check": "cost_limit",
                    "status": "passed",
                    "detail": f"cost={estimated_cost_usd}",
                }
            )
        except RoutingPolicyError as exc:
            return fail("cost_limit", exc)
    else:
        checks.append({"check": "cost_limit", "status": "skipped", "detail": "no estimate provided"})

    if current_requests_per_minute is not None:
        try:
            enforce_routing_rate_limit(policy, current_requests_per_minute)
            checks.append(
                {
                    "check": "rate_limit",
                    "status": "passed",
                    "detail": f"rpm={current_requests_per_minute}",
                }
            )
        except RoutingPolicyError as exc:
            return fail("rate_limit", exc)
    else:
        checks.append({"check": "rate_limit", "status": "skipped", "detail": "no rpm provided"})

    try:
        selected_provider, selected_model, selected_by = _resolve_routing_decision(
            policy,
            provider,
            model,
            available_providers=available_providers,
            estimated_cost_usd=estimated_cost_usd,
            current_requests_per_minute=current_requests_per_minute,
            current_hour_utc=current_hour_utc,
            governance_tags=governance_tags,
        )
        checks.append({"check": "governance", "status": "passed", "detail": "accepted"})
        checks.append({"check": "selection", "status": "passed", "detail": selected_by})
        return {
            "ok": True,
            "checks": checks,
            "decision": {
                "provider": selected_provider.value,
                "model": selected_model,
                "selected_by": selected_by,
            },
        }
    except RoutingPolicyError as exc:
        message = str(exc)
        if "governance" in message:
            checks.append({"check": "governance", "status": "failed", "detail": message})
            checks.append({"check": "selection", "status": "skipped", "detail": "selection aborted"})
            return {"ok": False, "checks": checks, "error": message}
        return fail("selection", exc)


def _coerce_provider(value: Any) -> ProviderType:
    if isinstance(value, ProviderType):
        return value
    return ProviderType(str(value))


def _normalize_policy_scenario(index: int, scenario: dict[str, Any]) -> tuple[ProviderType, str, dict[str, Any]]:
    provider = _coerce_provider(scenario.get("provider", ProviderType.OPENAI))
    model = str(scenario.get("model", "gpt-5.2"))
    options = scenario.get("options")
    if options is not None and not isinstance(options, dict):
        raise ValueError(f"scenario {index} options must be a dict")
    return provider, model, options or {}


def evaluate_policy_gate(
    policy: RoutingPolicy | None,
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    failed_indexes: list[int] = []
    for index, scenario in enumerate(scenarios):
        provider, model, options = _normalize_policy_scenario(index, scenario)
        explanation = explain_routing_target(policy, provider, model, **options)
        results.append(explanation)
        if not explanation.get("ok"):
            failed_indexes.append(index)
    failed = len(failed_indexes)
    return {
        "total": len(results),
        "passed": len(results) - failed,
        "failed": failed,
        "failed_indexes": failed_indexes,
        "results": results,
    }


def evaluate_policy_diff(
    candidate_policy: RoutingPolicy | None,
    scenarios: list[dict[str, Any]],
    baseline_policy: RoutingPolicy | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        provider, model, options = _normalize_policy_scenario(index, scenario)
        baseline = explain_routing_target(baseline_policy, provider, model, **options)
        candidate = explain_routing_target(candidate_policy, provider, model, **options)
        changed = (
            baseline.get("ok") != candidate.get("ok")
            or (baseline.get("decision") or {}).get("provider") != (candidate.get("decision") or {}).get("provider")
            or (baseline.get("decision") or {}).get("model") != (candidate.get("decision") or {}).get("model")
            or (baseline.get("decision") or {}).get("selected_by") != (candidate.get("decision") or {}).get("selected_by")
            or baseline.get("error") != candidate.get("error")
        )
        results.append(
            {
                "index": index,
                "input": {"provider": provider.value, "model": model},
                "baseline": baseline,
                "candidate": candidate,
                "changed": changed,
            }
        )
    regressions = sum(
        1
        for item in results
        if bool(item["baseline"].get("ok")) and not bool(item["candidate"].get("ok"))
    )
    return {
        "total": len(results),
        "baseline_passed": sum(1 for item in results if item["baseline"].get("ok")),
        "candidate_passed": sum(1 for item in results if item["candidate"].get("ok")),
        "changed": sum(1 for item in results if item["changed"]),
        "regressions": regressions,
        "results": results,
    }


def evaluate_policy_rollout_guardrails(
    diff: dict[str, Any],
    guardrails: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rules = guardrails or {}
    total = int(diff.get("total", 0))
    changed = int(diff.get("changed", 0))
    regressions = int(diff.get("regressions", 0))
    candidate_passed = int(diff.get("candidate_passed", 0))
    max_changed = int(rules.get("max_changed", total))
    max_regressions = int(rules.get("max_regressions", 0))
    min_candidate_pass_rate = float(rules.get("min_candidate_pass_rate", 1.0))
    candidate_pass_rate = 1.0 if total == 0 else candidate_passed / total

    checks = [
        {
            "check": "changed_budget",
            "status": "passed" if changed <= max_changed else "failed",
            "detail": f"changed={changed}, limit={max_changed}",
        },
        {
            "check": "regression_budget",
            "status": "passed" if regressions <= max_regressions else "failed",
            "detail": f"regressions={regressions}, limit={max_regressions}",
        },
        {
            "check": "candidate_pass_rate",
            "status": "passed" if candidate_pass_rate >= min_candidate_pass_rate else "failed",
            "detail": f"candidate_pass_rate={candidate_pass_rate:.3f}, min={min_candidate_pass_rate:.3f}",
        },
    ]
    failed = next((check for check in checks if check["status"] == "failed"), None)
    if failed is None:
        return {"ok": True, "checks": checks}
    return {"ok": False, "checks": checks, "error": failed["detail"]}
