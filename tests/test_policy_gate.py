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
