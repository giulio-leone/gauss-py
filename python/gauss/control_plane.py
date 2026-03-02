"""Unified Control Plane — lightweight local ops dashboard for Gauss."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import TYPE_CHECKING, Any

from gauss.tokens import estimate_cost

if TYPE_CHECKING:
    from gauss.approval import ApprovalManager
    from gauss.telemetry import Telemetry


class ControlPlane:
    """Aggregate telemetry, approvals, and cost snapshots in one local UI.

    The control plane can be consumed programmatically through :meth:`snapshot`
    or served via a tiny local HTTP dashboard with :meth:`start_server`.
    """

    def __init__(
        self,
        telemetry: Telemetry | None = None,
        approvals: ApprovalManager | None = None,
        model: str = "gpt-5.2",
    ) -> None:
        self._telemetry = telemetry
        self._approvals = approvals
        self._model = model
        self._latest_cost: Any = None
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: Thread | None = None

    def with_model(self, model: str) -> "ControlPlane":
        self._model = model
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

    def snapshot(self) -> dict[str, Any]:
        return {
            "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "spans": self._telemetry.export_spans() if self._telemetry else [],
            "metrics": self._telemetry.export_metrics() if self._telemetry else {},
            "pending_approvals": self._approvals.list_pending() if self._approvals else [],
            "latest_cost": self._latest_cost.__dict__ if self._latest_cost is not None else None,
        }

    def start_server(self, host: str = "127.0.0.1", port: int = 4200) -> str:
        if self._server is not None:
            return f"http://{host}:{self._server.server_port}"

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/api/snapshot":
                    payload = json.dumps(outer.snapshot(), indent=2).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                if self.path == "/":
                    html = outer._render_dashboard_html().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                    return

                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Not found")

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
  <div class=\"muted\">Live snapshot refreshes every 2s</div>
  <pre id=\"out\">loading...</pre>
  <script>
    async function refresh() {
      const r = await fetch('/api/snapshot');
      const j = await r.json();
      document.getElementById('out').textContent = JSON.stringify(j, null, 2);
    }
    setInterval(refresh, 2000);
    refresh();
  </script>
</body>
</html>"""

