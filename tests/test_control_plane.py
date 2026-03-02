"""Tests for unified control plane."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.error
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

    def test_auth_token_protects_api(self):
        cp = ControlPlane(auth_token="secret-token")
        url = cp.start_server(port=0)

        try:
            urllib.request.urlopen(f"{url}/api/snapshot")
            assert False, "Expected unauthorized"
        except urllib.error.HTTPError as exc:
            assert exc.code == 401

        with urllib.request.urlopen(f"{url}/api/snapshot?token=secret-token") as resp:
            assert resp.status == 200

        cp.stop_server()

    def test_filters_history_timeline_dag_and_persistence(self):
        persist_path = os.path.join(tempfile.gettempdir(), f"gauss-cp-{os.getpid()}.jsonl")
        if os.path.exists(persist_path):
            os.remove(persist_path)

        set_pricing("cp-test-model", ModelPricing(input_per_token=0.001, output_per_token=0.001))
        cp = ControlPlane(
            telemetry=_DummyTelemetry(),
            approvals=_DummyApprovals(),
            model="cp-test-model",
            persist_path=persist_path,
        )
        cp.set_cost_usage(2, 3)
        url = cp.start_server(port=0)

        with urllib.request.urlopen(f"{url}/api/snapshot?section=metrics") as resp:
            metrics_only = json.loads(resp.read().decode("utf-8"))
            assert metrics_only["metrics"]["total_spans"] == 1

        with urllib.request.urlopen(f"{url}/api/history") as resp:
            history = json.loads(resp.read().decode("utf-8"))
            assert len(history) >= 1

        with urllib.request.urlopen(f"{url}/api/timeline") as resp:
            timeline = json.loads(resp.read().decode("utf-8"))
            assert timeline[-1]["span_count"] == 1
            assert timeline[-1]["pending_approvals_count"] == 1

        with urllib.request.urlopen(f"{url}/api/dag") as resp:
            dag = json.loads(resp.read().decode("utf-8"))
            assert len(dag["nodes"]) == 1

        cp.stop_server()
        clear_pricing()
        assert os.path.exists(persist_path)
        with open(persist_path, "r", encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if line.strip()]
            assert len(lines) >= 1
        os.remove(persist_path)

    def test_supports_tenant_session_filters(self):
        cp = ControlPlane(
            telemetry=_DummyTelemetry(),
            approvals=_DummyApprovals(),
        )
        cp.with_context({"tenant_id": "t-1", "session_id": "s-1", "run_id": "r-1"}).snapshot()
        cp.with_context({"tenant_id": "t-2", "session_id": "s-2", "run_id": "r-2"}).snapshot()

        url = cp.start_server(port=0)
        with urllib.request.urlopen(f"{url}/api/history?tenant=t-1") as resp:
            history = json.loads(resp.read().decode("utf-8"))
            assert len(history) == 1
            assert history[0]["context"]["tenant_id"] == "t-1"

        with urllib.request.urlopen(f"{url}/api/timeline?session=s-2") as resp:
            timeline = json.loads(resp.read().decode("utf-8"))
            assert len(timeline) == 1
            assert timeline[0]["span_count"] == 1

        with urllib.request.urlopen(f"{url}/api/dag?run=r-1") as resp:
            dag = json.loads(resp.read().decode("utf-8"))
            assert len(dag["nodes"]) == 1
            assert dag["nodes"][0]["label"] == "agent.run"

        cp.stop_server()
