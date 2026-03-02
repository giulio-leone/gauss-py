from __future__ import annotations

import json
from pathlib import Path

from gauss.policy_gate import main


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_policy_gate_main_success(tmp_path, capsys) -> None:
    scenarios = tmp_path / "scenarios.json"
    policy = tmp_path / "policy.json"
    _write_json(
        scenarios,
        [{"provider": "openai", "model": "gpt-5.2", "options": {"current_hour_utc": 10}}],
    )
    _write_json(policy, {"allowed_hours_utc": [10]})

    code = main([str(scenarios), str(policy)])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert code == 0
    assert payload["failed"] == 0
    assert payload["passed"] == 1


def test_policy_gate_main_failure(tmp_path, capsys) -> None:
    scenarios = tmp_path / "scenarios.json"
    policy = tmp_path / "policy.json"
    _write_json(
        scenarios,
        [{"provider": "openai", "model": "gpt-5.2", "options": {"current_hour_utc": 22}}],
    )
    _write_json(policy, {"allowed_hours_utc": [10]})

    code = main([str(scenarios), str(policy)])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert code == 1
    assert payload["failed"] == 1
    assert payload["failed_indexes"] == [0]


def test_policy_gate_main_rollout_guardrails(tmp_path, capsys) -> None:
    scenarios = tmp_path / "scenarios.json"
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    guardrails = tmp_path / "guardrails.json"
    _write_json(
        scenarios,
        [
            {"provider": "openai", "model": "gpt-5.2", "options": {"current_hour_utc": 10}},
            {"provider": "openai", "model": "gpt-5.2", "options": {"current_hour_utc": 22}},
        ],
    )
    _write_json(candidate, {"allowed_hours_utc": [10]})
    _write_json(baseline, {})
    _write_json(guardrails, {"max_regressions": 0, "min_candidate_pass_rate": 0.9})

    code = main([str(scenarios), str(candidate), str(baseline), str(guardrails)])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert code == 1
    assert payload["summary"]["failed"] == 1
    assert payload["diff"]["regressions"] == 1
    assert payload["rollout"]["ok"] is False
