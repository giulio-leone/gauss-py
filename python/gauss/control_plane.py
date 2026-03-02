"""Unified Control Plane — local operational surface for Gauss."""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from gauss.errors import ValidationError
from gauss.tokens import estimate_cost

if TYPE_CHECKING:
    from gauss.approval import ApprovalManager
    from gauss.telemetry import Telemetry


_SECTION_KEYS = {"spans", "metrics", "pending_approvals", "latest_cost"}
_STREAM_CHANNELS = {"snapshot", "timeline", "dag"}


class _ControlPlaneForbiddenError(Exception):
    """Raised when control-plane scope violates auth claims."""


class ControlPlane:
    """Aggregate telemetry, approvals, and cost snapshots behind a local API/UI."""

    def __init__(
        self,
        telemetry: Telemetry | None = None,
        approvals: ApprovalManager | None = None,
        model: str = "gpt-5.2",
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
        self._auth_token = auth_token
        self._auth_claims: dict[str, Any] = dict(auth_claims or {})
        self._persist_path = persist_path
        self._history_limit = history_limit
        self._stream_replay_limit = stream_replay_limit
        self._context: dict[str, str] = dict(context or {})

        self._latest_cost: Any = None
        self._history: list[dict[str, Any]] = []
        self._next_stream_event_id = 1
        self._stream_events: list[dict[str, Any]] = []
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: Thread | None = None

    def with_model(self, model: str) -> "ControlPlane":
        self._model = model
        return self

    def with_auth_token(self, token: str | None) -> "ControlPlane":
        self._auth_token = token
        return self

    def with_auth_claims(self, claims: dict[str, Any] | None) -> "ControlPlane":
        self._auth_claims = dict(claims or {})
        return self

    def with_context(self, context: dict[str, str]) -> "ControlPlane":
        self._assert_context_allowed(context)
        self._context = dict(context)
        return self

    def set_cost_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        *,
        reasoning_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_creation_tokens: int | None = None,
    ) -> "ControlPlane":
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

                    if parsed.path == "/api/stream":
                        filters = outer._apply_auth_claims(outer._parse_context_filters(params))
                        channels = outer._parse_stream_channels(params)
                        for channel in channels:
                            outer._assert_channel_allowed(channel)
                        once = params.get("once", ["0"])[0] in {"1", "true", "yes"}
                        follow = params.get("follow", ["0"])[0] in {"1", "true", "yes"}
                        last_event_id = outer._parse_last_event_id(self.headers, params)
                        interval_ms = int(params.get("interval_ms", ["1000"])[0] or "1000")
                        if interval_ms < 100:
                            interval_ms = 100

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
                frame = f"{id_frame}event: {event}\ndata: {json.dumps(payload)}\n\n".encode(
                    "utf-8"
                )
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

    def __enter__(self) -> "ControlPlane":
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    def _capture_snapshot(self) -> dict[str, Any]:
        self._assert_context_allowed(self._context)
        item = {
            "generated_at": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
            "context": dict(self._context),
            "spans": self._telemetry.export_spans() if self._telemetry else [],
            "metrics": self._telemetry.export_metrics() if self._telemetry else {},
            "pending_approvals": self._approvals.list_pending() if self._approvals else [],
            "latest_cost": self._latest_cost.__dict__ if self._latest_cost is not None else None,
        }
        self._history.append(item)
        if len(self._history) > self._history_limit:
            self._history.pop(0)

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
            or x_token == self._auth_token
            or query_token == self._auth_token
        )

    def _parse_context_filters(self, params: dict[str, list[str]]) -> dict[str, str | None]:
        return {
            "tenant_id": params.get("tenant", [None])[0],
            "session_id": params.get("session", [None])[0],
            "run_id": params.get("run", [None])[0],
        }

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
        if len(self._stream_events) > self._stream_replay_limit:
            self._stream_events.pop(0)
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
        if run_id and context.get("run_id") != run_id:
            return False
        return True

    def _assert_context_allowed(self, context: dict[str, str]) -> None:
        self._apply_auth_claims(
            {
                "tenant_id": context.get("tenant_id"),
                "session_id": context.get("session_id"),
                "run_id": context.get("run_id"),
            }
        )

    def _assert_channel_allowed(self, channel: str) -> None:
        roles = [str(role).lower() for role in (self._auth_claims.get("roles") or [])]
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
            "hosted_dashboard_path": "/ops",
            "hosted_tenant_dashboard_path": "/ops/tenants",
        }

    def _ops_health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "generated_at": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
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
            "generated_at": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
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
            if generated_at >= current["latest_generated_at"]:
                current["latest_generated_at"] = generated_at
            if context.get("session_id"):
                current["session_ids"].add(str(context["session_id"]))
            if context.get("run_id"):
                current["run_ids"].add(str(context["run_id"]))

        out: list[dict[str, Any]] = []
        for tenant_id in sorted(grouped.keys()):
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
