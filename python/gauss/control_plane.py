"""Unified Control Plane — local operational surface for Gauss."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from collections.abc import Callable
from copy import copy, deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from gauss._types import ProviderType
from gauss._utils import _utc_iso
from gauss.errors import ValidationError
from gauss.tokens import estimate_cost

if TYPE_CHECKING:
    from gauss.approval import ApprovalManager
    from gauss.routing_policy import RoutingPolicy
    from gauss.telemetry import Telemetry

__all__ = ["ControlPlane"]

_SECTION_KEYS = {"spans", "metrics", "pending_approvals", "latest_cost"}
_STREAM_CHANNELS = {"snapshot", "timeline", "dag"}
_POLICY_DRIFT_WINDOWS = {"custom", "last_1h", "last_24h", "last_7d"}
_PROVIDERS = {
    "openai",
    "anthropic",
    "google",
    "groq",
    "ollama",
    "deepseek",
    "openrouter",
    "together",
    "fireworks",
    "mistral",
    "perplexity",
    "xai",
}


class _ControlPlaneForbiddenError(Exception):
    """Raised when control-plane scope violates auth claims."""


class ControlPlane:
    """Aggregate telemetry, approvals, and cost snapshots behind a local API/UI."""

    def __init__(
        self,
        telemetry: Telemetry | None = None,
        approvals: ApprovalManager | None = None,
        model: str = "gpt-5.2",
        routing_policy: RoutingPolicy | None = None,
        *,
        auth_token: str | None = None,
        auth_claims: dict[str, Any] | None = None,
        persist_path: str | None = None,
        history_limit: int = 200,
        stream_replay_limit: int = 500,
        context: dict[str, str] | None = None,
    ) -> None:
        self._telemetry = telemetry
        self._approvals = approvals
        self._model = model
        self._routing_policy = routing_policy
        self._auth_token = auth_token
        self._auth_claims: dict[str, Any] = dict(auth_claims or {})
        self._persist_path = persist_path
        self._history_limit = history_limit
        self._stream_replay_limit = stream_replay_limit
        self._context: dict[str, str] = dict(context or {})

        self._latest_cost: Any = None
        self._history: deque[dict[str, Any]] = deque(maxlen=self._history_limit)
        self._next_stream_event_id = 1
        self._stream_events: deque[dict[str, Any]] = deque(maxlen=self._stream_replay_limit)
        self._next_explain_trace_id = 1
        self._explain_traces: deque[dict[str, Any]] = deque(maxlen=self._history_limit)
        self._latest_explain_trace_id: str | None = None
        self._policy_drift_alert_hooks: list[Callable[[dict[str, Any]], None]] = []
        self._policy_drift_sinks: list[str] = []
        self._policy_drift_schedule_config: dict[str, Any] | None = None
        self._policy_drift_runs: deque[dict[str, Any]] = deque(maxlen=self._history_limit)
        self._next_policy_drift_run_id = 1
        self._next_policy_lifecycle_version = 1
        self._policy_lifecycle_versions: list[dict[str, Any]] = []
        self._active_policy_version_id: str | None = None
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: Thread | None = None

    def with_model(self, model: str) -> ControlPlane:
        self._model = model
        return self

    def with_routing_policy(self, routing_policy: RoutingPolicy | None) -> ControlPlane:
        self._routing_policy = routing_policy
        return self

    def with_auth_token(self, token: str | None) -> ControlPlane:
        self._auth_token = token
        return self

    def with_auth_claims(self, claims: dict[str, Any] | None) -> ControlPlane:
        self._auth_claims = dict(claims or {})
        return self

    def with_context(self, context: dict[str, str]) -> ControlPlane:
        self._assert_context_allowed(context)
        self._context = dict(context)
        return self

    def on_policy_drift_alert(self, hook: Callable[[dict[str, Any]], None]) -> ControlPlane:
        self._policy_drift_alert_hooks.append(hook)
        return self

    def register_policy_drift_sink(self, sink_id: str) -> ControlPlane:
        normalized = str(sink_id).strip()
        if not normalized:
            raise ValidationError("sink_id must be a non-empty string", "sink_id")
        if normalized not in self._policy_drift_sinks:
            self._policy_drift_sinks.append(normalized)
        return self

    def set_cost_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        reasoning_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_creation_tokens: int | None = None,
    ) -> ControlPlane:
        self._latest_cost = estimate_cost(
            self._model,
            input_tokens,
            output_tokens,
            reasoning_tokens=reasoning_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )
        return self

    def snapshot(self, section: str | None = None) -> dict[str, Any]:
        full = self._capture_snapshot()
        if section is None:
            return full
        if section not in _SECTION_KEYS:
            raise ValidationError(f'Unknown section "{section}"', "section")
        return {"generated_at": full["generated_at"], "context": full["context"], section: full[section]}

    def history(self, filters: dict[str, str | None] | None = None) -> list[dict[str, Any]]:
        return self._filter_history(filters)

    def timeline(self, filters: dict[str, str | None] | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in self._filter_history(filters):
            spans = item.get("spans")
            pending = item.get("pending_approvals")
            latest_cost = item.get("latest_cost")
            out.append(
                {
                    "generated_at": item.get("generated_at"),
                    "span_count": len(spans) if isinstance(spans, list) else 0,
                    "pending_approvals_count": len(pending) if isinstance(pending, list) else 0,
                    "total_cost_usd": float((latest_cost or {}).get("total_cost_usd", 0.0)),
                    "latest_explain_trace_id": item.get("latest_explain_trace_id"),
                }
            )
        return out

    def dag(self, filters: dict[str, str | None] | None = None) -> dict[str, list[dict[str, str]]]:
        filtered = self._filter_history(filters)
        if not filtered:
            return {"nodes": [], "edges": []}
        latest = filtered[-1]
        spans = latest.get("spans")
        if not isinstance(spans, list):
            return {"nodes": [], "edges": []}

        nodes: list[dict[str, str]] = []
        for i, span in enumerate(spans):
            label = f"span-{i + 1}"
            if isinstance(span, dict):
                if isinstance(span.get("name"), str):
                    label = span["name"]
                elif isinstance(span.get("span_name"), str):
                    label = span["span_name"]
            nodes.append({"id": str(i), "label": label})

        edges = [{"from": str(i), "to": str(i + 1)} for i in range(max(0, len(nodes) - 1))]
        return {"nodes": nodes, "edges": edges}

    def start_server(self, host: str = "127.0.0.1", port: int = 4200) -> str:
        if self._server is not None:
            return f"http://{host}:{self._server.server_port}"

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)

                if parsed.path.startswith("/api/") and not outer._is_authorized(self.headers, params):
                    self.send_response(401)
                    payload = b'{"error":"Unauthorized"}'
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                try:
                    if parsed.path == "/api/snapshot":
                        section = params.get("section", [None])[0]
                        payload = json.dumps(outer.snapshot(section), indent=2).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/history":
                        filters = outer._apply_auth_claims(outer._parse_context_filters(params))
                        payload = json.dumps(
                            outer.history(filters), indent=2
                        ).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/timeline":
                        filters = outer._apply_auth_claims(outer._parse_context_filters(params))
                        payload = json.dumps(
                            outer.timeline(filters), indent=2
                        ).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/dag":
                        filters = outer._apply_auth_claims(outer._parse_context_filters(params))
                        payload = json.dumps(
                            outer.dag(filters), indent=2
                        ).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/ops/capabilities":
                        payload = json.dumps(outer._ops_capabilities(), indent=2).encode(
                            "utf-8"
                        )
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/ops/health":
                        payload = json.dumps(outer._ops_health(), indent=2).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/ops/summary":
                        filters = outer._apply_auth_claims(outer._parse_context_filters(params))
                        payload = json.dumps(outer._ops_summary(filters), indent=2).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/ops/tenants":
                        filters = outer._apply_auth_claims(outer._parse_context_filters(params))
                        payload = json.dumps(outer._ops_tenants(filters), indent=2).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/ops/policy/explain":
                        payload = json.dumps(outer._ops_policy_explain(params), indent=2).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/ops/policy/explain/batch":
                        payload = json.dumps(outer._ops_policy_explain_batch(params), indent=2).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/ops/policy/explain/simulate":
                        payload = json.dumps(outer._ops_policy_explain_simulation(params), indent=2).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/ops/policy/explain/diff":
                        payload = json.dumps(outer._ops_policy_explain_diff(params), indent=2).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/ops/policy/explain/traces":
                        payload = json.dumps(outer._ops_policy_explain_traces(params), indent=2).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/ops/policy/lifecycle/draft":
                        payload = json.dumps(outer._ops_policy_lifecycle_draft(params), indent=2).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/ops/policy/lifecycle/validate":
                        payload = json.dumps(outer._ops_policy_lifecycle_validate(params), indent=2).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/ops/policy/lifecycle/approve":
                        payload = json.dumps(outer._ops_policy_lifecycle_approve(params), indent=2).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/ops/policy/lifecycle/promote":
                        payload = json.dumps(outer._ops_policy_lifecycle_promote(params), indent=2).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/ops/policy/lifecycle/versions":
                        payload = json.dumps(outer._ops_policy_lifecycle_versions(), indent=2).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/ops/policy/drift/schedule/set":
                        payload = json.dumps(outer._ops_policy_drift_schedule_set(params), indent=2).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/ops/policy/drift/schedule/run":
                        payload = json.dumps(outer._ops_policy_drift_schedule_run(params), indent=2).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/ops/policy/drift/schedule":
                        payload = json.dumps(outer._ops_policy_drift_schedule(), indent=2).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/ops/policy/drift":
                        payload = json.dumps(outer._ops_policy_drift(params), indent=2).encode("utf-8")
                        self._send_json(payload)
                        return

                    if parsed.path == "/api/stream":
                        filters = outer._apply_auth_claims(outer._parse_context_filters(params))
                        channels = outer._parse_stream_channels(params)
                        for channel in channels:
                            outer._assert_channel_allowed(channel)
                        once = params.get("once", ["0"])[0] in {"1", "true", "yes"}
                        follow = params.get("follow", ["0"])[0] in {"1", "true", "yes"}
                        last_event_id = outer._parse_last_event_id(self.headers, params)
                        interval_ms = int(params.get("interval_ms", ["1000"])[0] or "1000")
                        interval_ms = max(interval_ms, 100)

                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "keep-alive" if (follow and not once) else "close")
                        self.end_headers()

                        try:
                            replayed = outer._replay_stream_events(channels, filters, last_event_id)
                            for event in replayed:
                                self._send_sse_event(event["event"], event)
                            if once and replayed:
                                self.wfile.flush()
                                self.close_connection = True
                                return

                            for event in outer._emit_stream_batch(channels, filters):
                                self._send_sse_event(event["event"], event)
                            self.wfile.flush()
                            if once or not follow:
                                self.close_connection = True
                                return
                            while True:
                                time.sleep(interval_ms / 1000.0)
                                for event in outer._emit_stream_batch(channels, filters):
                                    self._send_sse_event(event["event"], event)
                                self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            return

                    if parsed.path == "/":
                        html = outer._render_dashboard_html().encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(html)))
                        self.end_headers()
                        self.wfile.write(html)
                        return

                    if parsed.path == "/ops":
                        html = outer._render_hosted_ops_html().encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(html)))
                        self.end_headers()
                        self.wfile.write(html)
                        return

                    if parsed.path == "/ops/tenants":
                        html = outer._render_hosted_tenant_ops_html().encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.send_header("Content-Length", str(len(html)))
                        self.end_headers()
                        self.wfile.write(html)
                        return

                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b"Not found")
                except _ControlPlaneForbiddenError as exc:
                    payload = json.dumps({"error": str(exc)}).encode("utf-8")
                    self.send_response(403)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except Exception as exc:
                    logging.getLogger(__name__).warning("Control plane error: %s", exc)
                    payload = json.dumps({"error": str(exc)}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

            def _send_json(self, payload: bytes) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def _send_sse_event(self, event: str, payload: dict[str, Any]) -> None:
                event_id = payload.get("id")
                id_frame = f"id: {event_id}\n" if isinstance(event_id, int) else ""
                frame = f"{id_frame}event: {event}\ndata: {json.dumps(payload)}\n\n".encode()
                self.wfile.write(frame)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server_thread = Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        return f"http://{host}:{self._server.server_port}"

    def stop_server(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        if self._server_thread is not None:
            self._server_thread.join(timeout=2)
            self._server_thread = None

    def destroy(self) -> None:
        self.stop_server()

    def __enter__(self) -> ControlPlane:
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    def _capture_snapshot(self) -> dict[str, Any]:
        self._assert_context_allowed(self._context)
        item = {
            "generated_at": _utc_iso(),
            "context": dict(self._context),
            "spans": self._telemetry.export_spans() if self._telemetry else [],
            "metrics": self._telemetry.export_metrics() if self._telemetry else {},
            "pending_approvals": self._approvals.list_pending() if self._approvals else [],
            "latest_cost": self._latest_cost.__dict__ if self._latest_cost is not None else None,
            "latest_explain_trace_id": self._latest_explain_trace_id,
        }
        self._history.append(item)

        if self._persist_path:
            os.makedirs(os.path.dirname(self._persist_path) or ".", exist_ok=True)
            with open(self._persist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(item) + "\n")
        return item

    def _is_authorized(self, headers: Any, params: dict[str, list[str]]) -> bool:
        if not self._auth_token:
            return True
        auth = headers.get("Authorization")
        x_token = headers.get("x-gauss-token")
        query_token = params.get("token", [None])[0]
        return (
            auth == f"Bearer {self._auth_token}"
            or self._auth_token in (x_token, query_token)
        )

    def _parse_context_filters(self, params: dict[str, list[str]]) -> dict[str, str | None]:
        return {
            "tenant_id": params.get("tenant", [None])[0],
            "session_id": params.get("session", [None])[0],
            "run_id": params.get("run", [None])[0],
        }

    def _parse_policy_provider(self, value: str) -> ProviderType:
        provider = value.strip().lower()
        if provider not in _PROVIDERS:
            raise ValidationError(f'Unknown provider "{value}"', "provider")
        return ProviderType(provider)

    def _parse_optional_float(
        self,
        params: dict[str, list[str]],
        key: str,
        *,
        field: str,
    ) -> float | None:
        raw = params.get(key, [None])[0]
        return self._parse_optional_float_value(raw, field=field)

    def _parse_optional_float_value(self, raw: Any, *, field: str) -> float | None:
        if raw is None or raw == "":
            return None
        try:
            return float(str(raw))
        except ValueError as exc:
            raise ValidationError(f'Invalid {field} "{raw}"', field) from exc

    def _parse_optional_int(
        self,
        params: dict[str, list[str]],
        key: str,
        *,
        field: str,
    ) -> int | None:
        raw = params.get(key, [None])[0]
        return self._parse_optional_int_value(raw, field=field)

    def _parse_optional_int_value(self, raw: Any, *, field: str) -> int | None:
        if raw is None or raw == "":
            return None
        try:
            return int(str(raw))
        except ValueError as exc:
            raise ValidationError(f'Invalid {field} "{raw}"', field) from exc

    def _require_non_empty_string(self, raw: Any, *, field: str) -> str:
        if not isinstance(raw, str):
            raise ValidationError(f"{field} must be a string", field)
        value = raw.strip()
        if not value:
            raise ValidationError(f"{field} must not be empty", field)
        return value

    def _parse_policy_explain_options(
        self,
        params: dict[str, list[str]],
    ) -> tuple[ProviderType, str, dict[str, Any]]:
        return self._parse_policy_explain_scenario(
            {
                "provider": params.get("provider", ["openai"])[0],
                "model": params.get("model", ["gpt-5.2"])[0],
                "available": params.get("available", [None])[0],
                "cost": params.get("cost", [None])[0],
                "rpm": params.get("rpm", [None])[0],
                "hour": params.get("hour", [None])[0],
                "tags": params.get("tags", [None])[0],
            }
        )

    def _parse_policy_explain_scenario(
        self,
        scenario: dict[str, Any],
    ) -> tuple[ProviderType, str, dict[str, Any]]:
        provider_raw = scenario.get("provider")
        provider = self._parse_policy_provider(
            provider_raw if isinstance(provider_raw, str) and provider_raw else "openai"
        )
        model_raw = scenario.get("model")
        model = model_raw if isinstance(model_raw, str) and model_raw else "gpt-5.2"

        available_raw = scenario.get("available")
        available_values: list[str]
        if isinstance(available_raw, str):
            available_values = [value.strip() for value in available_raw.split(",") if value.strip()]
        elif isinstance(available_raw, list):
            available_values = [str(value).strip() for value in available_raw if str(value).strip()]
        else:
            available_values = []
        available_providers = (
            [self._parse_policy_provider(value) for value in available_values]
            if available_values
            else None
        )

        tags_raw = scenario.get("tags")
        if isinstance(tags_raw, str):
            governance_tags = [value.strip() for value in tags_raw.split(",") if value.strip()]
        elif isinstance(tags_raw, list):
            governance_tags = [str(value).strip() for value in tags_raw if str(value).strip()]
        else:
            governance_tags = None

        options = {
            "available_providers": available_providers,
            "estimated_cost_usd": self._parse_optional_float_value(scenario.get("cost"), field="cost"),
            "current_requests_per_minute": self._parse_optional_int_value(scenario.get("rpm"), field="rpm"),
            "current_hour_utc": self._parse_optional_int_value(scenario.get("hour"), field="hour"),
            "governance_tags": governance_tags,
        }
        return provider, model, options

    def _parse_policy_explain_batch_scenarios(
        self,
        params: dict[str, list[str]],
    ) -> list[tuple[ProviderType, str, dict[str, Any]]]:
        raw = params.get("scenarios", [None])[0]
        if raw is None or raw == "":
            raise ValidationError("Missing scenarios query parameter", "scenarios")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValidationError("Invalid scenarios JSON payload", "scenarios") from exc
        if not isinstance(parsed, list) or not parsed:
            raise ValidationError("scenarios must be a non-empty array", "scenarios")
        out: list[tuple[ProviderType, str, dict[str, Any]]] = []
        for index, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ValidationError(f"Scenario {index} must be an object", "scenarios")
            out.append(self._parse_policy_explain_scenario(item))
        return out

    def _parse_policy_provider_list(self, value: Any, field: str) -> list[ProviderType]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValidationError(f"{field} must be an array", field)
        out: list[ProviderType] = []
        for index, item in enumerate(value):
            out.append(self._parse_policy_provider(self._require_non_empty_string(item, field=f"{field}[{index}]")))
        return out

    def _parse_policy_hour_list(self, value: Any, field: str) -> list[int]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValidationError(f"{field} must be an array", field)
        out: list[int] = []
        for index, item in enumerate(value):
            out.append(self._parse_optional_int_value(item, field=f"{field}[{index}]") or 0)
        return out

    def _parse_policy_provider_weights(self, value: Any, field: str) -> dict[ProviderType, int]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValidationError(f"{field} must be an object", field)
        out: dict[ProviderType, int] = {}
        for provider_raw, weight_raw in value.items():
            provider = self._parse_policy_provider(self._require_non_empty_string(provider_raw, field=field))
            out[provider] = self._parse_optional_int_value(weight_raw, field=f"{field}.{provider.value}") or 0
        return out

    def _parse_policy_lifecycle_policy(self, params: dict[str, list[str]]) -> RoutingPolicy:
        from gauss.routing_policy import (
            GovernancePolicyPack,
            GovernanceRule,
            RoutingCandidate,
            RoutingPolicy,
        )

        raw = params.get("policy", [None])[0]
        if raw is None or raw == "":
            raise ValidationError("Missing policy query parameter", "policy")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValidationError("Invalid policy JSON payload", "policy") from exc
        if not isinstance(parsed, dict):
            raise ValidationError("policy must be a JSON object", "policy")

        aliases_raw = parsed.get("aliases")
        aliases: dict[str, list[RoutingCandidate]] = {}
        if aliases_raw is not None:
            if not isinstance(aliases_raw, dict):
                raise ValidationError("aliases must be an object", "aliases")
            for alias, candidates in aliases_raw.items():
                alias_name = self._require_non_empty_string(alias, field="aliases")
                if not isinstance(candidates, list):
                    raise ValidationError(f"aliases.{alias_name} must be an array", f"aliases.{alias_name}")
                parsed_candidates: list[RoutingCandidate] = []
                for index, candidate in enumerate(candidates):
                    if not isinstance(candidate, dict):
                        raise ValidationError(
                            f"aliases.{alias_name}[{index}] must be an object",
                            f"aliases.{alias_name}[{index}]",
                        )
                    provider = self._parse_policy_provider(
                        self._require_non_empty_string(
                            candidate.get("provider"),
                            field=f"aliases.{alias_name}[{index}].provider",
                        )
                    )
                    model = self._require_non_empty_string(
                        candidate.get("model"),
                        field=f"aliases.{alias_name}[{index}].model",
                    )
                    priority = self._parse_optional_int_value(
                        candidate.get("priority"),
                        field=f"aliases.{alias_name}[{index}].priority",
                    )
                    max_cost_usd = self._parse_optional_float_value(
                        candidate.get("max_cost_usd"),
                        field=f"aliases.{alias_name}[{index}].max_cost_usd",
                    )
                    parsed_candidates.append(
                        RoutingCandidate(
                            provider=provider,
                            model=model,
                            priority=priority or 0,
                            max_cost_usd=max_cost_usd,
                        )
                    )
                aliases[alias_name] = parsed_candidates

        governance_raw = parsed.get("governance")
        governance: GovernancePolicyPack | None = None
        if governance_raw is not None:
            if not isinstance(governance_raw, dict):
                raise ValidationError("governance must be an object", "governance")
            rules_raw = governance_raw.get("rules")
            if rules_raw is None:
                rules_raw = []
            if not isinstance(rules_raw, list):
                raise ValidationError("governance.rules must be an array", "governance.rules")
            rules: list[GovernanceRule] = []
            for index, item in enumerate(rules_raw):
                if not isinstance(item, dict):
                    raise ValidationError(
                        f"governance.rules[{index}] must be an object",
                        f"governance.rules[{index}]",
                    )
                rule_type = self._require_non_empty_string(item.get("type"), field=f"governance.rules[{index}].type")
                tag_raw = item.get("tag")
                if tag_raw is not None and not isinstance(tag_raw, str):
                    raise ValidationError(f"governance.rules[{index}].tag must be a string", "governance.rules")
                provider_raw = item.get("provider")
                provider = (
                    self._parse_policy_provider(
                        self._require_non_empty_string(provider_raw, field=f"governance.rules[{index}].provider")
                    )
                    if provider_raw is not None
                    else None
                )
                rules.append(GovernanceRule(type=rule_type, tag=tag_raw, provider=provider))
            governance = GovernancePolicyPack(
                rules=rules,
                max_total_cost_usd=self._parse_optional_float_value(
                    governance_raw.get("max_total_cost_usd"),
                    field="governance.max_total_cost_usd",
                ),
                max_requests_per_minute=self._parse_optional_int_value(
                    governance_raw.get("max_requests_per_minute"),
                    field="governance.max_requests_per_minute",
                ),
                allowed_hours_utc=self._parse_policy_hour_list(
                    governance_raw.get("allowed_hours_utc"),
                    field="governance.allowed_hours_utc",
                ),
                provider_weights=self._parse_policy_provider_weights(
                    governance_raw.get("provider_weights"),
                    field="governance.provider_weights",
                ),
                fallback_order=self._parse_policy_provider_list(
                    governance_raw.get("fallback_order"),
                    field="governance.fallback_order",
                ),
            )

        return RoutingPolicy(
            aliases=aliases,
            fallback_order=self._parse_policy_provider_list(parsed.get("fallback_order"), field="fallback_order"),
            max_total_cost_usd=self._parse_optional_float_value(
                parsed.get("max_total_cost_usd"),
                field="max_total_cost_usd",
            ),
            max_requests_per_minute=self._parse_optional_int_value(
                parsed.get("max_requests_per_minute"),
                field="max_requests_per_minute",
            ),
            allowed_hours_utc=self._parse_policy_hour_list(parsed.get("allowed_hours_utc"), field="allowed_hours_utc"),
            provider_weights=self._parse_policy_provider_weights(parsed.get("provider_weights"), field="provider_weights"),
            governance=governance,
        )

    def _parse_optional_policy_from_query(self, params: dict[str, list[str]], key: str) -> RoutingPolicy | None:
        raw = params.get(key, [None])[0]
        if raw is None or raw == "":
            return None
        return self._parse_policy_lifecycle_policy({"policy": [raw]})

    def _parse_policy_drift_guardrails(self, params: dict[str, list[str]]) -> dict[str, Any]:
        max_changed = self._parse_optional_int(params, "maxChanged", field="maxChanged")
        max_regressions = self._parse_optional_int(params, "maxRegressions", field="maxRegressions")
        min_candidate_pass_rate = self._parse_optional_float(params, "minCandidatePassRate", field="minCandidatePassRate")
        out: dict[str, Any] = {}
        if max_changed is not None:
            out["max_changed"] = max_changed
        if max_regressions is not None:
            out["max_regressions"] = max_regressions
        if min_candidate_pass_rate is not None:
            out["min_candidate_pass_rate"] = min_candidate_pass_rate
        return out

    def _parse_policy_drift_window(self, params: dict[str, list[str]]) -> str:
        raw = str(params.get("window", ["custom"])[0] or "custom").strip().lower()
        if raw in _POLICY_DRIFT_WINDOWS:
            return raw
        raise ValidationError(f'Invalid policy drift window "{raw}"', "window")

    def _summarize_policy_lifecycle_version(self, version: dict[str, Any]) -> dict[str, Any]:
        return {
            "version_id": version["version_id"],
            "status": version["status"],
            "created_at": version["created_at"],
            "validated_at": version.get("validated_at"),
            "approved_at": version.get("approved_at"),
            "promoted_at": version.get("promoted_at"),
            "has_validation": version.get("validation") is not None,
            "validation": version.get("validation"),
            "audit": version.get("audit"),
        }

    def _find_policy_lifecycle_version(self, version_id: str) -> dict[str, Any]:
        for item in self._policy_lifecycle_versions:
            if item["version_id"] == version_id:
                return item
        raise ValidationError(f'Unknown lifecycle version "{version_id}"', "version")

    def _resolve_lifecycle_actor(
        self,
        params: dict[str, list[str]],
        *,
        action: str,
        allowed_roles: list[str],
    ) -> tuple[str, str | None, str | None]:
        requested_role_raw = params.get("role", [None])[0]
        requested_role = str(requested_role_raw or "").strip().lower() or None

        claim_roles = [
            str(role).strip().lower()
            for role in (self._auth_claims.get("roles") or [])
            if str(role).strip()
        ]
        if claim_roles:
            if requested_role is not None and requested_role not in claim_roles:
                raise _ControlPlaneForbiddenError(f'Forbidden lifecycle role "{requested_role}"')
            if requested_role is not None and requested_role not in allowed_roles:
                raise _ControlPlaneForbiddenError(f'Forbidden lifecycle action "{action}"')
            effective_role = requested_role
            if effective_role is None:
                effective_role = next((role for role in claim_roles if role in allowed_roles), None)
            if effective_role is None:
                raise _ControlPlaneForbiddenError(f'Forbidden lifecycle action "{action}"')
        else:
            if requested_role is not None and requested_role not in allowed_roles:
                raise _ControlPlaneForbiddenError(f'Forbidden lifecycle action "{action}"')
            effective_role = requested_role or allowed_roles[0]

        actor = str(params.get("actor", [None])[0] or "").strip() or None
        comment = str(params.get("comment", [None])[0] or "").strip() or None
        return effective_role, actor, comment

    def _merge_lifecycle_audit(self, version: dict[str, Any], **fields: Any) -> None:
        audit = version.get("audit")
        if not isinstance(audit, dict):
            audit = {}
            version["audit"] = audit
        for key, value in fields.items():
            if value is not None:
                audit[key] = value

    def _ops_policy_lifecycle_draft(self, params: dict[str, list[str]]) -> dict[str, Any]:
        role, actor, comment = self._resolve_lifecycle_actor(
            params,
            action="draft",
            allowed_roles=["author", "operator", "admin"],
        )
        version = {
            "version_id": f"policy-v{self._next_policy_lifecycle_version}",
            "status": "draft",
            "created_at": _utc_iso(),
            "policy": self._parse_policy_lifecycle_policy(params),
            "validation": None,
            "audit": {
                "drafted_by_role": role,
                "drafted_by": actor,
                "draft_comment": comment,
            },
        }
        self._next_policy_lifecycle_version += 1
        self._policy_lifecycle_versions.append(version)
        return {"ok": True, "version": self._summarize_policy_lifecycle_version(version)}

    def _ops_policy_lifecycle_validate(self, params: dict[str, list[str]]) -> dict[str, Any]:
        from gauss.routing_policy import evaluate_policy_gate

        role, actor, comment = self._resolve_lifecycle_actor(
            params,
            action="validate",
            allowed_roles=["author", "reviewer", "operator", "admin"],
        )
        version_id = params.get("version", [None])[0]
        if version_id is None or version_id == "":
            raise ValidationError("Missing version query parameter", "version")
        version = self._find_policy_lifecycle_version(version_id)
        scenarios = self._parse_policy_explain_batch_scenarios(params)
        gate_scenarios = [
            {"provider": provider, "model": model, "options": options}
            for provider, model, options in scenarios
        ]
        validation = evaluate_policy_gate(version["policy"], gate_scenarios)
        version["validation"] = validation
        self._merge_lifecycle_audit(
            version,
            validated_by_role=role,
            validated_by=actor,
            validation_comment=comment,
        )
        if int(validation.get("failed", 0)) == 0:
            version["status"] = "validated"
            version["validated_at"] = _utc_iso()
        return {
            "ok": int(validation.get("failed", 0)) == 0,
            "version": self._summarize_policy_lifecycle_version(version),
            "validation": validation,
        }

    def _ops_policy_lifecycle_approve(self, params: dict[str, list[str]]) -> dict[str, Any]:
        role, actor, comment = self._resolve_lifecycle_actor(
            params,
            action="approve",
            allowed_roles=["reviewer", "operator", "admin"],
        )
        version_id = params.get("version", [None])[0]
        if version_id is None or version_id == "":
            raise ValidationError("Missing version query parameter", "version")
        version = self._find_policy_lifecycle_version(version_id)
        validation = version.get("validation") or {}
        if version.get("status") != "validated" or int(validation.get("failed", 1)) > 0:
            return {
                "ok": False,
                "version": self._summarize_policy_lifecycle_version(version),
                "error": "version must pass validation before approval",
            }
        version["status"] = "approved"
        version["approved_at"] = _utc_iso()
        self._merge_lifecycle_audit(
            version,
            approved_by_role=role,
            approved_by=actor,
            approval_comment=comment,
        )
        return {"ok": True, "version": self._summarize_policy_lifecycle_version(version)}

    def _ops_policy_lifecycle_promote(self, params: dict[str, list[str]]) -> dict[str, Any]:
        role, actor, comment = self._resolve_lifecycle_actor(
            params,
            action="promote",
            allowed_roles=["promoter", "operator", "admin"],
        )
        version_id = params.get("version", [None])[0]
        if version_id is None or version_id == "":
            raise ValidationError("Missing version query parameter", "version")
        version = self._find_policy_lifecycle_version(version_id)
        if version.get("status") != "approved":
            return {
                "ok": False,
                "active_version_id": self._active_policy_version_id,
                "version": self._summarize_policy_lifecycle_version(version),
                "error": "version must be approved before promotion",
            }
        for item in self._policy_lifecycle_versions:
            if item["version_id"] != version_id and item.get("status") == "promoted":
                item["status"] = "approved"
        version["status"] = "promoted"
        version["promoted_at"] = _utc_iso()
        self._merge_lifecycle_audit(
            version,
            promoted_by_role=role,
            promoted_by=actor,
            promotion_comment=comment,
        )
        self._active_policy_version_id = version_id
        # deepcopy: version["policy"] is shared with _policy_lifecycle_versions
        self._routing_policy = deepcopy(version["policy"])
        return {
            "ok": True,
            "active_version_id": self._active_policy_version_id,
            "version": self._summarize_policy_lifecycle_version(version),
        }

    def _ops_policy_lifecycle_versions(self) -> dict[str, Any]:
        return {
            "ok": True,
            "active_version_id": self._active_policy_version_id,
            "versions": [self._summarize_policy_lifecycle_version(item) for item in self._policy_lifecycle_versions],
        }

    def _ops_policy_explain(self, params: dict[str, list[str]]) -> dict[str, Any]:
        from gauss.routing_policy import explain_routing_target

        provider, model, options = self._parse_policy_explain_options(params)
        explanation = explain_routing_target(
            self._routing_policy,
            provider,
            model,
            available_providers=options["available_providers"],
            estimated_cost_usd=options["estimated_cost_usd"],
            current_requests_per_minute=options["current_requests_per_minute"],
            current_hour_utc=options["current_hour_utc"],
            governance_tags=options["governance_tags"],
        )
        trace = self._record_policy_explain_trace(
            "single",
            {
                "input": {"provider": provider.value, "model": model, "options": options},
                "explanation": explanation,
            },
        )
        return {**explanation, "trace_id": trace["trace_id"]}

    def _ops_policy_explain_batch(self, params: dict[str, list[str]]) -> dict[str, Any]:
        scenarios = self._parse_policy_explain_batch_scenarios(params)
        response = self._build_policy_explain_batch_response(scenarios)
        trace = self._record_policy_explain_trace("batch", response)
        return {**response, "trace_id": trace["trace_id"]}

    def _build_policy_explain_batch_response(
        self,
        scenarios: list[tuple[ProviderType, str, dict[str, Any]]],
    ) -> dict[str, Any]:
        from gauss.routing_policy import explain_routing_target

        results: list[dict[str, Any]] = []
        for index, (provider, model, options) in enumerate(scenarios):
            explanation = explain_routing_target(
                self._routing_policy,
                provider,
                model,
                available_providers=options["available_providers"],
                estimated_cost_usd=options["estimated_cost_usd"],
                current_requests_per_minute=options["current_requests_per_minute"],
                current_hour_utc=options["current_hour_utc"],
                governance_tags=options["governance_tags"],
            )
            results.append(
                {
                    "index": index,
                    "input": {"provider": provider.value, "model": model},
                    "explanation": explanation,
                }
            )

        passed = sum(1 for item in results if bool(item["explanation"].get("ok")))
        return {
            "ok": True,
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "results": results,
        }

    def _ops_policy_explain_simulation(self, params: dict[str, list[str]]) -> dict[str, Any]:
        scenarios = self._parse_policy_explain_batch_scenarios(params)
        response = self._build_policy_explain_batch_response(scenarios)
        trace = self._record_policy_explain_trace("simulate", response)
        return {**response, "trace_id": trace["trace_id"]}

    def _ops_policy_explain_diff(self, params: dict[str, list[str]]) -> dict[str, Any]:
        from gauss.routing_policy import evaluate_policy_diff

        scenarios = self._parse_policy_explain_batch_scenarios(params)
        diff = evaluate_policy_diff(
            self._routing_policy,
            [
                {
                    "provider": provider,
                    "model": model,
                    "options": {
                        "available_providers": options["available_providers"],
                        "estimated_cost_usd": options["estimated_cost_usd"],
                        "current_requests_per_minute": options["current_requests_per_minute"],
                        "current_hour_utc": options["current_hour_utc"],
                        "governance_tags": options["governance_tags"],
                    },
                }
                for provider, model, options in scenarios
            ],
            None,
        )

        response = {
            "ok": True,
            "total": diff["total"],
            "baseline_passed": diff["baseline_passed"],
            "candidate_passed": diff["candidate_passed"],
            "changed": diff["changed"],
            "regressions": diff["regressions"],
            "results": diff["results"],
        }
        trace = self._record_policy_explain_trace("diff", response)
        return {**response, "trace_id": trace["trace_id"]}

    def _ops_policy_drift_schedule_set(self, params: dict[str, list[str]]) -> dict[str, Any]:
        scenarios_raw = params.get("scenarios", [None])[0]
        if not scenarios_raw:
            raise ValidationError("Missing scenarios query parameter", "scenarios")
        self._parse_policy_explain_batch_scenarios(params)
        interval_ms = self._parse_optional_int(params, "intervalMs", field="intervalMs")
        interval_ms = interval_ms if interval_ms is not None else 60_000
        if interval_ms <= 0:
            raise ValidationError("intervalMs must be > 0", "intervalMs")
        now = _utc_iso()
        self._policy_drift_schedule_config = {
            "enabled": True,
            "interval_ms": interval_ms,
            "window": self._parse_policy_drift_window(params),
            "scenarios_raw": scenarios_raw,
            "baseline_policy_raw": params.get("baselinePolicy", [None])[0] or None,
            "candidate_policy_raw": params.get("candidatePolicy", [None])[0] or None,
            "baseline_version_id": params.get("baselineVersion", [None])[0] or None,
            "candidate_version_id": params.get("candidateVersion", [None])[0] or None,
            "guardrails": self._parse_policy_drift_guardrails(params),
            "updated_at": now,
            "last_run_at": self._policy_drift_schedule_config.get("last_run_at")
            if self._policy_drift_schedule_config
            else None,
        }
        return {"ok": True, "schedule": self._summarize_policy_drift_schedule(self._policy_drift_schedule_config)}

    def _summarize_policy_drift_schedule(self, schedule: dict[str, Any]) -> dict[str, Any]:
        return {
            "enabled": bool(schedule.get("enabled")),
            "interval_ms": schedule.get("interval_ms"),
            "window": schedule.get("window"),
            "updated_at": schedule.get("updated_at"),
            "last_run_at": schedule.get("last_run_at"),
            "baseline_version_id": schedule.get("baseline_version_id"),
            "candidate_version_id": schedule.get("candidate_version_id"),
            "has_baseline_policy": bool(schedule.get("baseline_policy_raw")),
            "has_candidate_policy": bool(schedule.get("candidate_policy_raw")),
            "guardrails": schedule.get("guardrails") or {},
        }

    def _ops_policy_drift_schedule(self) -> dict[str, Any]:
        if self._policy_drift_schedule_config is None:
            return {"ok": True, "schedule": None, "runs": list(self._policy_drift_runs)}
        return {
            "ok": True,
            "schedule": self._summarize_policy_drift_schedule(self._policy_drift_schedule_config),
            "runs": list(self._policy_drift_runs),
        }

    def _ops_policy_drift_schedule_run(self, params: dict[str, list[str]]) -> dict[str, Any]:
        run_params = {key: list(values) for key, values in params.items()}
        if not str(run_params.get("scenarios", [None])[0] or "").strip():
            if self._policy_drift_schedule_config is None:
                raise ValidationError("Missing scenarios query parameter", "scenarios")
            run_params["scenarios"] = [self._policy_drift_schedule_config["scenarios_raw"]]
            run_params.setdefault("window", [self._policy_drift_schedule_config["window"]])
            if self._policy_drift_schedule_config.get("baseline_version_id"):
                run_params.setdefault("baselineVersion", [self._policy_drift_schedule_config["baseline_version_id"]])
            if self._policy_drift_schedule_config.get("candidate_version_id"):
                run_params.setdefault("candidateVersion", [self._policy_drift_schedule_config["candidate_version_id"]])
            if self._policy_drift_schedule_config.get("baseline_policy_raw"):
                run_params.setdefault("baselinePolicy", [self._policy_drift_schedule_config["baseline_policy_raw"]])
            if self._policy_drift_schedule_config.get("candidate_policy_raw"):
                run_params.setdefault("candidatePolicy", [self._policy_drift_schedule_config["candidate_policy_raw"]])
            guardrails = self._policy_drift_schedule_config.get("guardrails") or {}
            if guardrails.get("max_changed") is not None:
                run_params.setdefault("maxChanged", [str(guardrails["max_changed"])])
            if guardrails.get("max_regressions") is not None:
                run_params.setdefault("maxRegressions", [str(guardrails["max_regressions"])])
            if guardrails.get("min_candidate_pass_rate") is not None:
                run_params.setdefault("minCandidatePassRate", [str(guardrails["min_candidate_pass_rate"])])

        drift = self._ops_policy_drift(run_params)
        run = {
            **drift,
            "run_id": f"drift-run-{self._next_policy_drift_run_id}",
        }
        self._next_policy_drift_run_id += 1
        self._policy_drift_runs.append(run)
        if self._policy_drift_schedule_config is not None:
            now = _utc_iso()
            self._policy_drift_schedule_config["last_run_at"] = run["generated_at"]
            self._policy_drift_schedule_config["updated_at"] = now
        return run

    def _ops_policy_drift(self, params: dict[str, list[str]]) -> dict[str, Any]:
        from gauss.routing_policy import evaluate_policy_diff, evaluate_policy_rollout_guardrails

        scenarios = self._parse_policy_explain_batch_scenarios(params)
        window = self._parse_policy_drift_window(params)
        baseline_version_id = params.get("baselineVersion", [None])[0]
        candidate_version_id = params.get("candidateVersion", [None])[0]

        baseline_from_version = (
            copy(self._find_policy_lifecycle_version(baseline_version_id)["policy"])
            if baseline_version_id
            else copy(self._find_policy_lifecycle_version(self._active_policy_version_id)["policy"])
            if self._active_policy_version_id
            else None
        )
        candidate_from_version = (
            copy(self._find_policy_lifecycle_version(candidate_version_id)["policy"])
            if candidate_version_id
            else None
        )
        baseline_policy = self._parse_optional_policy_from_query(params, "baselinePolicy") or baseline_from_version
        candidate_policy = (
            self._parse_optional_policy_from_query(params, "candidatePolicy")
            or candidate_from_version
            or self._routing_policy
        )

        scenario_payload = [
            {"provider": provider, "model": model, "options": options}
            for provider, model, options in scenarios
        ]
        diff = evaluate_policy_diff(candidate_policy, scenario_payload, baseline_policy)
        guardrails = evaluate_policy_rollout_guardrails(diff, self._parse_policy_drift_guardrails(params))
        response = {
            "ok": bool(guardrails.get("ok")),
            "alert": not bool(guardrails.get("ok")),
            "generated_at": _utc_iso(),
            "window": window,
            "baseline_version_id": baseline_version_id or self._active_policy_version_id,
            "candidate_version_id": candidate_version_id,
            "diff": diff,
            "guardrails": guardrails,
            "sinks_triggered": list(self._policy_drift_sinks) if not bool(guardrails.get("ok")) else [],
        }
        if response["alert"]:
            for hook in self._policy_drift_alert_hooks:
                hook(response)
        trace = self._record_policy_explain_trace("drift", response)
        return {**response, "trace_id": trace["trace_id"]}

    def _record_policy_explain_trace(self, mode: str, payload: dict[str, Any]) -> dict[str, Any]:
        trace = {
            "trace_id": f"trace-{self._next_explain_trace_id}",
            "generated_at": _utc_iso(),
            "mode": mode,
            "payload": payload,
        }
        self._next_explain_trace_id += 1
        self._latest_explain_trace_id = str(trace["trace_id"])
        self._explain_traces.append(trace)
        return trace

    def _ops_policy_explain_traces(self, params: dict[str, list[str]]) -> dict[str, Any]:
        trace_id = params.get("traceId", [None])[0]
        traces = (
            [trace for trace in self._explain_traces if trace.get("trace_id") == trace_id]
            if trace_id
            else list(self._explain_traces)
        )
        return {"total": len(traces), "traces": traces}

    def _parse_stream_channel(self, channel: str | None) -> str:
        if channel is None:
            return "snapshot"
        if channel in _STREAM_CHANNELS:
            return channel
        raise ValidationError(f'Unknown stream channel "{channel}"', "channel")

    def _parse_stream_channels(self, params: dict[str, list[str]]) -> list[str]:
        channels_param = params.get("channels", [None])[0]
        if not channels_param:
            return [self._parse_stream_channel(params.get("channel", [None])[0])]
        channels = [
            self._parse_stream_channel(value.strip())
            for value in channels_param.split(",")
            if value.strip()
        ]
        if not channels:
            return ["snapshot"]
        return list(dict.fromkeys(channels))

    def _parse_last_event_id(self, headers: Any, params: dict[str, list[str]]) -> int | None:
        raw = params.get("lastEventId", [None])[0] or headers.get("Last-Event-ID")
        if raw is None:
            return None
        try:
            parsed = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f'Invalid lastEventId "{raw}"', "lastEventId") from exc
        if parsed < 0:
            raise ValidationError(f'Invalid lastEventId "{raw}"', "lastEventId")
        return parsed

    def _build_stream_event(
        self,
        channel: str,
        filters: dict[str, str | None],
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        if channel == "timeline":
            payload: Any = self.timeline(filters)
        elif channel == "dag":
            payload = self.dag(filters)
        else:
            filtered = self._filter_history(filters)
            payload = filtered[-1] if filtered else snapshot
        event = {
            "id": self._next_stream_event_id,
            "event": channel,
            "generated_at": snapshot["generated_at"],
            "context": dict(snapshot.get("context", {})),
            "payload": payload,
        }
        self._next_stream_event_id += 1
        self._stream_events.append(event)
        return event

    def _emit_stream_batch(
        self,
        channels: list[str],
        filters: dict[str, str | None],
    ) -> list[dict[str, Any]]:
        snapshot = self._capture_snapshot()
        return [self._build_stream_event(channel, filters, snapshot) for channel in channels]

    def _replay_stream_events(
        self,
        channels: list[str],
        filters: dict[str, str | None],
        last_event_id: int | None,
    ) -> list[dict[str, Any]]:
        if last_event_id is None:
            return []
        tenant_id = filters.get("tenant_id")
        session_id = filters.get("session_id")
        run_id = filters.get("run_id")
        out: list[dict[str, Any]] = []
        for event in self._stream_events:
            if int(event.get("id", 0)) <= last_event_id:
                continue
            if event.get("event") not in channels:
                continue
            if not self._matches_context(event.get("context"), tenant_id, session_id, run_id):
                continue
            out.append(event)
        return out

    def _apply_auth_claims(self, filters: dict[str, str | None]) -> dict[str, str | None]:
        if not self._auth_claims:
            return filters
        resolved = dict(filters)

        claim_tenant = self._auth_claims.get("tenant_id")
        if claim_tenant:
            if resolved.get("tenant_id") and resolved["tenant_id"] != claim_tenant:
                raise _ControlPlaneForbiddenError("Forbidden tenant scope")
            resolved["tenant_id"] = resolved.get("tenant_id") or claim_tenant

        allowed_sessions = self._auth_claims.get("allowed_session_ids") or []
        session_id = resolved.get("session_id")
        if session_id and allowed_sessions and session_id not in allowed_sessions:
            raise _ControlPlaneForbiddenError("Forbidden session scope")
        if not session_id and len(allowed_sessions) == 1:
            resolved["session_id"] = allowed_sessions[0]

        allowed_runs = self._auth_claims.get("allowed_run_ids") or []
        run_id = resolved.get("run_id")
        if run_id and allowed_runs and run_id not in allowed_runs:
            raise _ControlPlaneForbiddenError("Forbidden run scope")
        if not run_id and len(allowed_runs) == 1:
            resolved["run_id"] = allowed_runs[0]

        return resolved

    def _filter_history(self, filters: dict[str, str | None] | None) -> list[dict[str, Any]]:
        if not filters:
            return list(self._history)
        tenant_id = filters.get("tenant_id")
        session_id = filters.get("session_id")
        run_id = filters.get("run_id")
        if not tenant_id and not session_id and not run_id:
            return list(self._history)
        return [
            item
            for item in self._history
            if self._matches_context(item.get("context"), tenant_id, session_id, run_id)
        ]

    def _matches_context(
        self,
        context: Any,
        tenant_id: str | None,
        session_id: str | None,
        run_id: str | None,
    ) -> bool:
        if not isinstance(context, dict):
            return tenant_id is None and session_id is None and run_id is None
        if tenant_id and context.get("tenant_id") != tenant_id:
            return False
        if session_id and context.get("session_id") != session_id:
            return False
        return not (run_id and context.get("run_id") != run_id)

    def _assert_context_allowed(self, context: dict[str, str]) -> None:
        self._apply_auth_claims(
            {
                "tenant_id": context.get("tenant_id"),
                "session_id": context.get("session_id"),
                "run_id": context.get("run_id"),
            }
        )

    def _assert_channel_allowed(self, channel: str) -> None:
        roles = {str(role).lower() for role in (self._auth_claims.get("roles") or [])}
        if not roles:
            return
        if "admin" in roles or "operator" in roles:
            return
        if channel in {"snapshot", "timeline"} and (
            "viewer" in roles or "reader" in roles
        ):
            return
        raise _ControlPlaneForbiddenError(f'Forbidden stream channel "{channel}"')

    def _render_dashboard_html(self) -> str:
        return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Gauss Control Plane</title>
  <style>
    body { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; margin: 24px; background: #0b1020; color: #f5f7ff; }
    h1 { margin-top: 0; }
    .muted { color: #a9b4d0; margin-bottom: 12px; }
    pre { background: #111935; border: 1px solid #25315f; padding: 16px; border-radius: 8px; overflow: auto; max-height: 70vh; }
  </style>
</head>
<body>
  <h1>Gauss Control Plane</h1>
  <div class=\"muted\">Live snapshot refreshes every 2s • filter with ?section=metrics • auth with ?token=...</div>
  <pre id=\"out\">loading...</pre>
  <script>
    const params = new URLSearchParams(window.location.search);
    const token = params.get('token');
    const section = params.get('section');
    const qs = new URLSearchParams();
    if (token) qs.set('token', token);
    if (section) qs.set('section', section);
    async function refresh() {
      const url = '/api/snapshot' + (qs.toString() ? ('?' + qs.toString()) : '');
      const r = await fetch(url);
      if (!r.ok) {
        document.getElementById('out').textContent = 'HTTP ' + r.status + ': ' + await r.text();
        return;
      }
      const j = await r.json();
      document.getElementById('out').textContent = JSON.stringify(j, null, 2);
    }
    setInterval(refresh, 2000);
    refresh();
  </script>
</body>
</html>"""

    def _ops_capabilities(self) -> dict[str, Any]:
        return {
            "sections": ["spans", "metrics", "pending_approvals", "latest_cost"],
            "channels": ["snapshot", "timeline", "dag"],
            "supports_multiplex": True,
            "supports_replay_cursor": True,
            "supports_channel_rbac": True,
            "supports_ops_summary": True,
            "supports_ops_tenants": True,
            "supports_policy_explain": True,
            "supports_policy_explain_batch": True,
            "supports_policy_explain_traces": True,
            "supports_policy_explain_diff": True,
            "supports_policy_lifecycle": True,
            "supports_policy_lifecycle_rbac": True,
            "supports_policy_drift_monitoring": True,
            "supports_policy_drift_scheduler": True,
            "supports_policy_drift_windows": True,
            "supports_policy_drift_alert_sinks": True,
            "hosted_dashboard_path": "/ops",
            "hosted_tenant_dashboard_path": "/ops/tenants",
            "policy_explain_path": "/api/ops/policy/explain",
            "policy_explain_batch_path": "/api/ops/policy/explain/batch",
            "policy_explain_simulate_path": "/api/ops/policy/explain/simulate",
            "policy_explain_trace_path": "/api/ops/policy/explain/traces",
            "policy_explain_diff_path": "/api/ops/policy/explain/diff",
            "policy_lifecycle_base_path": "/api/ops/policy/lifecycle",
            "policy_lifecycle_role_param": "role",
            "policy_lifecycle_audit_fields": [
                "drafted_by_role",
                "validated_by_role",
                "approved_by_role",
                "promoted_by_role",
            ],
            "policy_drift_path": "/api/ops/policy/drift",
            "policy_drift_schedule_path": "/api/ops/policy/drift/schedule",
            "policy_drift_schedule_run_path": "/api/ops/policy/drift/schedule/run",
        }

    def _ops_health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "generated_at": _utc_iso(),
            "history_size": len(self._history),
            "stream_buffer_size": len(self._stream_events),
        }

    def _ops_summary(self, filters: dict[str, str | None] | None = None) -> dict[str, Any]:
        history = self._filter_history(filters)
        latest = history[-1] if history else None
        spans = latest.get("spans") if isinstance(latest, dict) else None
        pending_approvals = latest.get("pending_approvals") if isinstance(latest, dict) else None

        if filters and any(filters.get(key) for key in ("tenant_id", "session_id", "run_id")):
            stream_buffer_size = len(
                [
                    event
                    for event in self._stream_events
                    if self._matches_context(
                        event.get("context"),
                        filters.get("tenant_id"),
                        filters.get("session_id"),
                        filters.get("run_id"),
                    )
                ]
            )
        else:
            stream_buffer_size = len(self._stream_events)

        tenants: set[str] = set()
        sessions: set[str] = set()
        runs: set[str] = set()
        for item in history:
            context = item.get("context", {}) if isinstance(item, dict) else {}
            if isinstance(context, dict):
                if context.get("tenant_id"):
                    tenants.add(str(context["tenant_id"]))
                if context.get("session_id"):
                    sessions.add(str(context["session_id"]))
                if context.get("run_id"):
                    runs.add(str(context["run_id"]))

        latest_cost = latest.get("latest_cost") if isinstance(latest, dict) else None
        latest_total_cost = (
            float(latest_cost.get("total_cost_usd", 0.0))
            if isinstance(latest_cost, dict)
            else 0.0
        )

        return {
            "status": "ok",
            "generated_at": _utc_iso(),
            "history_size": len(history),
            "stream_buffer_size": stream_buffer_size,
            "spans_count": len(spans) if isinstance(spans, list) else 0,
            "pending_approvals_count": len(pending_approvals) if isinstance(pending_approvals, list) else 0,
            "latest_total_cost_usd": latest_total_cost,
            "tenant_count": len(tenants),
            "session_count": len(sessions),
            "run_count": len(runs),
        }

    def _ops_tenants(self, filters: dict[str, str | None] | None = None) -> list[dict[str, Any]]:
        history = self._filter_history(filters)
        grouped: dict[str, dict[str, Any]] = {}

        for item in history:
            if not isinstance(item, dict):
                continue
            context = item.get("context", {})
            if not isinstance(context, dict):
                context = {}
            tenant_id = str(context.get("tenant_id") or "_unscoped")
            current = grouped.get(tenant_id)
            if current is None:
                current = {
                    "tenant_id": tenant_id,
                    "snapshot_count": 0,
                    "spans_count": 0,
                    "pending_approvals_count": 0,
                    "latest_total_cost_usd": 0.0,
                    "session_ids": set(),
                    "run_ids": set(),
                    "latest_generated_at": str(item.get("generated_at") or ""),
                }
                grouped[tenant_id] = current

            current["snapshot_count"] += 1
            spans = item.get("spans")
            pending = item.get("pending_approvals")
            current["spans_count"] += len(spans) if isinstance(spans, list) else 0
            current["pending_approvals_count"] += len(pending) if isinstance(pending, list) else 0
            latest_cost = item.get("latest_cost")
            if isinstance(latest_cost, dict):
                current["latest_total_cost_usd"] = float(latest_cost.get("total_cost_usd", 0.0))
            generated_at = str(item.get("generated_at") or "")
            current["latest_generated_at"] = max(generated_at, current["latest_generated_at"])
            if context.get("session_id"):
                current["session_ids"].add(str(context["session_id"]))
            if context.get("run_id"):
                current["run_ids"].add(str(context["run_id"]))

        out: list[dict[str, Any]] = []
        for tenant_id in sorted(grouped):
            current = grouped[tenant_id]
            out.append(
                {
                    "tenant_id": current["tenant_id"],
                    "snapshot_count": current["snapshot_count"],
                    "spans_count": current["spans_count"],
                    "pending_approvals_count": current["pending_approvals_count"],
                    "latest_total_cost_usd": current["latest_total_cost_usd"],
                    "session_count": len(current["session_ids"]),
                    "run_count": len(current["run_ids"]),
                    "latest_generated_at": current["latest_generated_at"],
                }
            )
        return out

    def _render_hosted_ops_html(self) -> str:
        return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Gauss Hosted Ops Console</title>
  <style>
    body { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; margin: 20px; background: #0b1020; color: #f5f7ff; }
    .row { margin-bottom: 12px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    input, button { background: #111935; color: #f5f7ff; border: 1px solid #25315f; border-radius: 6px; padding: 8px; }
    pre { background: #111935; border: 1px solid #25315f; padding: 12px; border-radius: 8px; max-height: 60vh; overflow: auto; }
    .muted { color: #a9b4d0; }
  </style>
</head>
<body>
  <h1>Gauss Hosted Ops Console</h1>
  <div class=\"muted\">Live stream viewer with multiplex channels + replay cursor support.</div>
  <div class=\"row\"><a href=\"/ops/tenants\" style=\"color:#9cc3ff\">Open tenant dashboard →</a></div>
  <div class=\"row\">
    <label>Token <input id=\"token\" placeholder=\"optional\" /></label>
    <label>Last Event ID <input id=\"lastEventId\" placeholder=\"optional\" /></label>
    <button id=\"connect\">Connect</button>
  </div>
  <div class=\"row\">
    <label><input type=\"checkbox\" class=\"ch\" value=\"snapshot\" checked /> snapshot</label>
    <label><input type=\"checkbox\" class=\"ch\" value=\"timeline\" checked /> timeline</label>
    <label><input type=\"checkbox\" class=\"ch\" value=\"dag\" /> dag</label>
  </div>
  <pre id=\"out\">idle</pre>
  <script>
    let source;
    const out = document.getElementById('out');
    function selectedChannels() {
      return [...document.querySelectorAll('.ch:checked')].map((node) => node.value);
    }
    function append(message) {
      out.textContent = message + '\\n' + out.textContent;
    }
    document.getElementById('connect').addEventListener('click', () => {
      if (source) source.close();
      const token = document.getElementById('token').value.trim();
      const lastEventId = document.getElementById('lastEventId').value.trim();
      const channels = selectedChannels();
      const qs = new URLSearchParams();
      if (channels.length > 0) qs.set('channels', channels.join(','));
      if (token) qs.set('token', token);
      if (lastEventId) qs.set('lastEventId', lastEventId);
      source = new EventSource('/api/stream?' + qs.toString());
      source.onmessage = (event) => append(event.data);
      source.onerror = () => append('stream disconnected');
      append('stream connected');
    });
    fetch('/api/ops/capabilities')
      .then((r) => r.json())
      .then((j) => append('capabilities: ' + JSON.stringify(j)));
  </script>
</body>
</html>"""

    def _render_hosted_tenant_ops_html(self) -> str:
        return """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Gauss Hosted Tenant Ops</title>
  <style>
    body { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; margin: 20px; background: #0b1020; color: #f5f7ff; }
    .row { margin-bottom: 12px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    input, button { background: #111935; color: #f5f7ff; border: 1px solid #25315f; border-radius: 6px; padding: 8px; }
    pre { background: #111935; border: 1px solid #25315f; padding: 12px; border-radius: 8px; max-height: 60vh; overflow: auto; }
    a { color: #9cc3ff; }
    .muted { color: #a9b4d0; }
  </style>
</head>
<body>
  <h1>Gauss Hosted Tenant Ops</h1>
  <div class=\"muted\">Tenant-level operational metrics powered by <code>/api/ops/tenants</code>.</div>
  <div class=\"row\"><a href=\"/ops\">← Back to stream console</a></div>
  <div class=\"row\">
    <label>Token <input id=\"token\" placeholder=\"optional\" /></label>
    <label>Tenant <input id=\"tenant\" placeholder=\"optional filter\" /></label>
    <button id=\"refresh\">Refresh</button>
  </div>
  <pre id=\"out\">loading...</pre>
  <script>
    const out = document.getElementById('out');
    async function refresh() {
      const token = document.getElementById('token').value.trim();
      const tenant = document.getElementById('tenant').value.trim();
      const qs = new URLSearchParams();
      if (token) qs.set('token', token);
      if (tenant) qs.set('tenant', tenant);
      const target = '/api/ops/tenants' + (qs.toString() ? ('?' + qs.toString()) : '');
      const r = await fetch(target);
      if (!r.ok) {
        out.textContent = 'HTTP ' + r.status + ': ' + await r.text();
        return;
      }
      const j = await r.json();
      out.textContent = JSON.stringify(j, null, 2);
    }
    document.getElementById('refresh').addEventListener('click', refresh);
    refresh();
  </script>
</body>
</html>"""
