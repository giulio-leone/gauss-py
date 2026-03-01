"""Telemetry and observability."""

from __future__ import annotations

import json
from typing import Any


class Telemetry:
    """Collect and export spans and metrics.

    Example::

        tel = Telemetry()
        tel.record_span({"name": "agent.run", "duration": 150})
        spans = tel.export_spans()
        metrics = tel.export_metrics()
        tel.clear()
    """

    def __init__(self) -> None:
        from gauss._native import create_telemetry  # type: ignore[import-not-found]

        self._handle: int = create_telemetry()
        self._destroyed = False

    def record_span(self, span: dict[str, Any]) -> None:
        """Record a telemetry span."""
        from gauss._native import telemetry_record_span  # type: ignore[import-not-found]

        self._check_alive()
        telemetry_record_span(self._handle, json.dumps(span))

    def export_spans(self) -> list[dict[str, Any]]:
        """Export all recorded spans."""
        from gauss._native import telemetry_export_spans  # type: ignore[import-not-found]

        self._check_alive()
        result_json: str = telemetry_export_spans(self._handle)
        return json.loads(result_json)  # type: ignore[no-any-return]

    def export_metrics(self) -> dict[str, Any]:
        """Export aggregated metrics."""
        from gauss._native import telemetry_export_metrics  # type: ignore[import-not-found]

        self._check_alive()
        result_json: str = telemetry_export_metrics(self._handle)
        return json.loads(result_json)  # type: ignore[no-any-return]

    def clear(self) -> None:
        """Clear all recorded telemetry data."""
        from gauss._native import telemetry_clear  # type: ignore[import-not-found]

        self._check_alive()
        telemetry_clear(self._handle)

    def destroy(self) -> None:
        if not self._destroyed:
            from gauss._native import destroy_telemetry  # type: ignore[import-not-found]

            destroy_telemetry(self._handle)
            self._destroyed = True

    def __enter__(self) -> Telemetry:
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    def __del__(self) -> None:
        self.destroy()

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("Telemetry has been destroyed")
