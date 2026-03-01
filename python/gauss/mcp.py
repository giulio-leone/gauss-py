"""MCP (Model Context Protocol) server implementation."""

from __future__ import annotations

import json
from typing import Any

from gauss._types import ToolDef


class McpServer:
    """MCP-compliant tool server.

    Example::

        server = McpServer("my-tools", "1.0.0")
        server.add_tool(ToolDef(name="greet", description="Say hello"))
        response = server.handle_message({"jsonrpc": "2.0", "method": "tools/list"})
    """

    def __init__(self, name: str, version: str = "1.0.0") -> None:
        from gauss._native import create_mcp_server  # type: ignore[import-not-found]

        self._handle: int = create_mcp_server(name, version)
        self._destroyed = False

    def add_tool(self, tool: ToolDef | dict[str, Any]) -> McpServer:
        """Register a tool. Returns self for chaining."""
        from gauss._native import mcp_server_add_tool  # type: ignore[import-not-found]

        self._check_alive()
        tool_dict = tool.to_dict() if isinstance(tool, ToolDef) else tool
        mcp_server_add_tool(self._handle, json.dumps(tool_dict))
        return self

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Handle an incoming MCP JSON-RPC message."""
        from gauss._native import mcp_server_handle  # type: ignore[import-not-found]

        self._check_alive()
        result_json: str = mcp_server_handle(self._handle, json.dumps(message))
        return json.loads(result_json)  # type: ignore[no-any-return]

    def destroy(self) -> None:
        if not self._destroyed:
            from gauss._native import destroy_mcp_server  # type: ignore[import-not-found]

            destroy_mcp_server(self._handle)
            self._destroyed = True

    def __enter__(self) -> McpServer:
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    def __del__(self) -> None:
        self.destroy()

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("McpServer has been destroyed")
