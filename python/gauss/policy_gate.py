"""CLI policy gate helper for CI workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gauss.routing_policy import (
    RoutingPolicy,
    evaluate_policy_diff,
    evaluate_policy_gate,
    evaluate_policy_rollout_guardrails,
)

__all__ = ["main"]

def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_policy(path: str | None) -> RoutingPolicy | None:
    if path is None:
        return None
    raw = _load_json(path)
    if not isinstance(raw, dict):
        raise ValueError("policy file must be a JSON object")
    return RoutingPolicy(**raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate routing policy scenarios for CI gates")
    parser.add_argument("scenarios", help="Path to scenarios JSON array")
    parser.add_argument("policy", nargs="?", default=None, help="Path to policy JSON object")
    parser.add_argument("baseline_policy", nargs="?", default=None, help="Path to baseline policy JSON object")
    parser.add_argument("guardrails", nargs="?", default=None, help="Path to rollout guardrails JSON object")
    args = parser.parse_args(argv)

    scenarios = _load_json(args.scenarios)
    if not isinstance(scenarios, list):
        raise ValueError("scenarios file must be a JSON array")

    policy = _load_policy(args.policy)
    summary = evaluate_policy_gate(policy, scenarios)
    rollout_failed = False
    output: dict[str, Any] | Any = summary
    if args.baseline_policy or args.guardrails:
        baseline_policy = _load_policy(args.baseline_policy)
        guardrails_raw = _load_json(args.guardrails) if args.guardrails else {}
        if not isinstance(guardrails_raw, dict):
            raise ValueError("guardrails file must be a JSON object")
        diff = evaluate_policy_diff(policy, scenarios, baseline_policy)
        rollout = evaluate_policy_rollout_guardrails(diff, guardrails_raw)
        rollout_failed = not bool(rollout.get("ok"))
        output = {"summary": summary, "diff": diff, "rollout": rollout}
    print(json.dumps(output, indent=2))
    return 1 if summary["failed"] > 0 or rollout_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
