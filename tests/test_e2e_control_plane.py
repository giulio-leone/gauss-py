"""E2E tests for control plane operational flow."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.request

from gauss.control_plane import ControlPlane
from gauss.tokens import ModelPricing, clear_pricing, set_pricing


class _Telemetry:
    def export_spans(self):
        return [{"name": "collect"}, {"name": "verify"}]

    def export_metrics(self):
        return {"total_spans": 2}


class _Approvals:
    def list_pending(self):
        return [{"id": "approval-1", "tool": "delete"}]


class TestControlPlaneE2E:
    def test_secured_end_to_end_flow(self):
        persist_path = os.path.join(tempfile.gettempdir(), f"gauss-cp-e2e-{os.getpid()}.jsonl")
        if os.path.exists(persist_path):
            os.remove(persist_path)

        set_pricing("cp-e2e-model", ModelPricing(input_per_token=0.001, output_per_token=0.001))
        cp = ControlPlane(
            telemetry=_Telemetry(),
            approvals=_Approvals(),
            model="cp-e2e-model",
            auth_token="e2e-token",
            persist_path=persist_path,
        )
        cp.set_cost_usage(10, 5)
        url = cp.start_server(port=0)

        try:
            urllib.request.urlopen(f"{url}/api/snapshot")
            assert False, "Expected unauthorized"
        except urllib.error.HTTPError as exc:
            assert exc.code == 401

        with urllib.request.urlopen(f"{url}/api/snapshot?token=e2e-token") as resp:
            assert resp.status == 200
            snap = json.loads(resp.read().decode("utf-8"))
            assert snap["metrics"]["total_spans"] == 2

        with urllib.request.urlopen(f"{url}/api/timeline?token=e2e-token") as resp:
            timeline = json.loads(resp.read().decode("utf-8"))
            assert timeline[-1]["span_count"] == 2
            assert timeline[-1]["pending_approvals_count"] == 1

        with urllib.request.urlopen(f"{url}/api/dag?token=e2e-token") as resp:
            dag = json.loads(resp.read().decode("utf-8"))
            assert len(dag["nodes"]) == 2
            assert len(dag["edges"]) == 1

        cp.stop_server()
        clear_pricing()
        assert os.path.exists(persist_path)
        with open(persist_path, "r", encoding="utf-8") as f:
            assert len([line for line in f if line.strip()]) > 0
        os.remove(persist_path)

