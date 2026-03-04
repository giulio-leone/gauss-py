"""Telemetry and observability."""

from __future__ import annotations

import functools
import json
from typing import Any

from gauss.base import StatefulResource


class Telemetry(StatefulResource):
    """Collect and export spans and metrics.

    Example::

        tel = Telemetry()
        tel.record_span({"name": "agent.run", "duration": 150})
        spans = tel.export_spans()
        metrics = tel.export_metrics()
        tel.clear()
    """

    def __init__(self) -> None:
        super().__init__()
        from gauss._native import create_telemetry

        self._handle: int = create_telemetry()

    @functools.cached_property
    def _resource_name(self) -> str:
        return "Telemetry"

    def record_span(self, span: dict[str, Any]) -> None:
        """Record a telemetry span."""
        from gauss._native import telemetry_record_span

        self._check_alive()
        telemetry_record_span(self._handle, json.dumps(span))

    def export_spans(self) -> list[dict[str, Any]]:
        """Export all recorded spans."""
        from gauss._native import telemetry_export_spans

        self._check_alive()
        result_json: str = telemetry_export_spans(self._handle)
        return json.loads(result_json)  # type: ignore[no-any-return]

    def export_metrics(self) -> dict[str, Any]:
        """Export aggregated metrics."""
        from gauss._native import telemetry_export_metrics

        self._check_alive()
        result_json: str = telemetry_export_metrics(self._handle)
        return json.loads(result_json)  # type: ignore[no-any-return]

    def clear(self) -> None:
        """Clear all recorded telemetry data."""
        from gauss._native import telemetry_clear

        self._check_alive()
        telemetry_clear(self._handle)

    def destroy(self) -> None:
        if not self._destroyed:
            from gauss._native import destroy_telemetry

            destroy_telemetry(self._handle)
        super().destroy()
