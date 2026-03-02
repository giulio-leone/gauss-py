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
    def test_secured_end_to_end_flow(self) -> None:
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
            raise AssertionError("Expected unauthorized")
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
        with open(persist_path, encoding="utf-8") as f:
            assert len([line for line in f if line.strip()]) > 0
        os.remove(persist_path)

    def test_stream_snapshot_simple_flow(self) -> None:
        cp = ControlPlane(
            telemetry=_Telemetry(),
            approvals=_Approvals(),
        )
        cp.with_context({"tenant_id": "t-simple", "session_id": "s-simple", "run_id": "r-simple"}).snapshot()
        url = cp.start_server(port=0)
        with urllib.request.urlopen(f"{url}/api/stream?channel=snapshot&once=1") as resp:
            assert resp.status == 200
            body = resp.read().decode("utf-8")
            assert "event: snapshot" in body
            data_line = next((line for line in body.splitlines() if line.startswith("data: ")), None)
            assert data_line is not None
            event = json.loads(data_line[6:])
            assert event["event"] == "snapshot"
            assert event["payload"]["context"]["tenant_id"] == "t-simple"
        cp.stop_server()

    def test_stream_timeline_scoped_flow(self) -> None:
        cp = ControlPlane(
            telemetry=_Telemetry(),
            approvals=_Approvals(),
            auth_token="stream-token",
            auth_claims={
                "tenant_id": "tenant-a",
                "allowed_session_ids": ["session-a"],
            },
        )
        cp.with_context({"tenant_id": "tenant-a", "session_id": "session-a", "run_id": "run-a"}).snapshot()
        url = cp.start_server(port=0)

        with urllib.request.urlopen(f"{url}/api/stream?token=stream-token&channel=timeline&once=1") as resp:
            assert resp.status == 200
            body = resp.read().decode("utf-8")
            data_line = next((line for line in body.splitlines() if line.startswith("data: ")), None)
            assert data_line is not None
            event = json.loads(data_line[6:])
            assert event["event"] == "timeline"
            assert isinstance(event["payload"], list)

        try:
            urllib.request.urlopen(
                f"{url}/api/stream?token=stream-token&channel=timeline&tenant=tenant-b&once=1"
            )
            raise AssertionError("Expected forbidden scope")
        except urllib.error.HTTPError as exc:
            assert exc.code == 403

        cp.stop_server()
