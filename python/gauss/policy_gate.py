"""CLI policy gate helper for CI workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from gauss.routing_policy import RoutingPolicy, evaluate_policy_gate


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
    args = parser.parse_args(argv)

    scenarios = _load_json(args.scenarios)
    if not isinstance(scenarios, list):
        raise ValueError("scenarios file must be a JSON array")

    policy = _load_policy(args.policy)
    summary = evaluate_policy_gate(policy, scenarios)
    print(json.dumps(summary, indent=2))
    return 1 if summary["failed"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
