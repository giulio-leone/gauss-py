"""Tests for unified control plane."""

from __future__ import annotations

import json
import urllib.request

from gauss.control_plane import ControlPlane
from gauss.tokens import ModelPricing, clear_pricing, set_pricing


class _DummyTelemetry:
    def export_spans(self):
        return [{"name": "agent.run", "duration_ms": 12}]

    def export_metrics(self):
        return {"total_spans": 1}


class _DummyApprovals:
    def list_pending(self):
        return [{"id": "req-1", "tool": "delete_user"}]


class TestControlPlane:
    def test_snapshot_contains_sections(self):
        cp = ControlPlane(telemetry=_DummyTelemetry(), approvals=_DummyApprovals())
        snap = cp.snapshot()
        assert snap["spans"][0]["name"] == "agent.run"
        assert snap["metrics"]["total_spans"] == 1
        assert snap["pending_approvals"][0]["id"] == "req-1"

    def test_set_cost_usage(self):
        set_pricing("cp-test-model", ModelPricing(input_per_token=0.001, output_per_token=0.002))
        cp = ControlPlane(model="cp-test-model")
        cp.set_cost_usage(10, 5)
        snap = cp.snapshot()
        assert snap["latest_cost"]["total_cost_usd"] == 0.02
        clear_pricing()

    def test_server_exposes_snapshot_and_html(self):
        cp = ControlPlane(telemetry=_DummyTelemetry(), approvals=_DummyApprovals())
        url = cp.start_server(port=0)
        with urllib.request.urlopen(f"{url}/api/snapshot") as resp:
            assert resp.status == 200
            payload = json.loads(resp.read().decode("utf-8"))
            assert payload["metrics"]["total_spans"] == 1

        with urllib.request.urlopen(f"{url}/") as resp:
            html = resp.read().decode("utf-8")
            assert "Gauss Control Plane" in html

        cp.stop_server()

