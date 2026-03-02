"""Plugin registry for event-driven extensibility."""

from __future__ import annotations

import json
from typing import Any


class PluginRegistry:
    """Plugin system with event emission and built-in plugins.

    Example::

        registry = (
            PluginRegistry()
            .add_telemetry()
            .add_memory()
        )
        print(registry.list())  # ["telemetry", "memory"]
        registry.emit({"type": "agent.start", "agent_name": "test"})
    """

    def __init__(self) -> None:
        from gauss._native import create_plugin_registry

        self._handle: int = create_plugin_registry()
        self._destroyed = False

    def add_telemetry(self) -> PluginRegistry:
        """Register the built-in telemetry plugin. Returns self."""
        from gauss._native import plugin_registry_add_telemetry

        self._check_alive()
        plugin_registry_add_telemetry(self._handle)
        return self

    def add_memory(self) -> PluginRegistry:
        """Register the built-in memory plugin. Returns self."""
        from gauss._native import plugin_registry_add_memory

        self._check_alive()
        plugin_registry_add_memory(self._handle)
        return self

    def list(self) -> list[str]:
        """List registered plugin names."""
        from gauss._native import plugin_registry_list

        self._check_alive()
        result_json: str = plugin_registry_list(self._handle)
        return json.loads(result_json)  # type: ignore[no-any-return]

    def emit(self, event: dict[str, Any]) -> None:
        """Emit an event to all registered plugins."""
        from gauss._native import plugin_registry_emit

        self._check_alive()
        plugin_registry_emit(self._handle, json.dumps(event))

    def destroy(self) -> None:
        if not self._destroyed:
            from gauss._native import destroy_plugin_registry

            destroy_plugin_registry(self._handle)
            self._destroyed = True

    def __enter__(self) -> PluginRegistry:
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    def __del__(self) -> None:
        self.destroy()

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("PluginRegistry has been destroyed")
