"""
MCP Client — consume tools from external MCP servers.

Quick start::

    from gauss import McpClient

    client = McpClient(command="npx", args=["-y", "@mcp/server-fs"])
    client.connect()
    tools = client.list_tools()
    result = client.call_tool("read_file", {"path": "README.md"})
    client.close()

Supports stdio transport (spawn subprocess) for local MCP servers.

.. versionadded:: 1.2.0
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any

from gauss._types import ToolDef

__all__ = ["McpClientConfig", "McpToolResult", "McpClient"]

@dataclass
class McpClientConfig:
    """Configuration for creating an MCP client."""

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    timeout_ms: int = 10000


@dataclass
class McpToolResult:
    """Result from calling a tool on an MCP server."""

    content: list[dict[str, Any]]
    is_error: bool = False


class McpClient:
    """Client for consuming tools from external MCP servers.

    Connects to an MCP server via stdio transport (subprocess), performs
    the initialization handshake, and provides methods to list and call
    tools.

    Example::

        client = McpClient(command="npx", args=["-y", "@mcp/server-everything"])
        client.connect()

        tools = client.list_tools()
        print(tools)

        result = client.call_tool("echo", {"message": "hello"})
        print(result)

        client.close()

    .. versionadded:: 1.2.0
    """

    def __init__(
        self,
        command: str | None = None,
        args: list[str] | None = None,
        *,
        config: McpClientConfig | None = None,
        env: dict[str, str] | None = None,
        timeout_ms: int = 10000,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = McpClientConfig(
                command=command or "",
                args=args or [],
                env=env,
                timeout_ms=timeout_ms,
            )
        self._process: subprocess.Popen[bytes] | None = None
        self._connected = False
        self._closed = False
        self._next_id = 1
        self._pending: dict[int, threading.Event] = {}
        self._results: dict[int, Any] = {}
        self._errors: dict[int, str] = {}
        self._buffer = ""
        self._reader_thread: threading.Thread | None = None
        self._cached_tools: list[ToolDef] | None = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        """Connect to the MCP server and perform initialization handshake."""
        if self._connected:
            return
        if self._closed:
            raise RuntimeError("McpClient has been closed")

        env = {**os.environ, **(self._config.env or {})}
        self._process = subprocess.Popen(
            [self._config.command, *self._config.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        self._reader_thread = threading.Thread(
            target=self._read_stdout, daemon=True
        )
        self._reader_thread.start()

        # Initialize handshake
        self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "gauss-mcp-client", "version": "1.2.0"},
        })

        # Send initialized notification
        self._notify("notifications/initialized", {})

        self._connected = True

    def list_tools(self) -> list[ToolDef]:
        """List all tools available on the connected MCP server."""
        self._assert_connected()

        if self._cached_tools is not None:
            return self._cached_tools

        result = self._request("tools/list", {})
        tools: list[ToolDef] = [
            ToolDef(
                name=t["name"],
                description=t.get("description", ""),
                parameters=t.get("inputSchema"),
            )
            for t in (result or {}).get("tools", [])
        ]

        self._cached_tools = tools
        return tools

    def call_tool(
        self, tool_name: str, args: dict[str, Any] | None = None
    ) -> McpToolResult:
        """Call a tool on the MCP server."""
        self._assert_connected()

        result = self._request("tools/call", {
            "name": tool_name,
            "arguments": args or {},
        })

        content = result.get("content", []) if result else []
        is_error = result.get("isError", False) if result else False
        return McpToolResult(content=content, is_error=is_error)

    def get_tools_with_executor(
        self,
    ) -> tuple[list[ToolDef], Any]:
        """Get tools and a wired executor for agent integration."""
        tools = self.list_tools()

        def executor(call_json: str) -> str:
            try:
                call = json.loads(call_json)
            except json.JSONDecodeError:
                return json.dumps({"error": "Invalid tool call JSON"})

            tool_name = call.get("tool") or call.get("name", "")
            tool_args = call.get("args") or call.get("arguments") or {}

            try:
                result = self.call_tool(tool_name, tool_args)
                if result.is_error:
                    error_text = " ".join(
                        c.get("text", "") for c in result.content if c.get("text")
                    )
                    return json.dumps({"error": error_text or "Tool error"})
                text = " ".join(
                    c.get("text", "") for c in result.content if c.get("text")
                )
                return json.dumps({"result": text})
            except Exception as exc:
                logging.getLogger(__name__).debug("MCP tool call failed: %s", exc)
                return json.dumps({"error": str(exc)})

        return tools, executor

    def close(self) -> None:
        """Close the connection and terminate the subprocess."""
        if self._closed:
            return
        self._closed = True
        self._connected = False
        self._cached_tools = None

        if self._process:
            try:
                if self._process.stdin:
                    self._process.stdin.close()
                self._process.terminate()
            except OSError:
                pass
            self._process = None

        # Wake up any pending requests
        with self._lock:
            for event in self._pending.values():
                event.set()
            self._pending.clear()

    @property
    def is_connected(self) -> bool:
        """Whether the client is currently connected."""
        return self._connected

    def __enter__(self) -> McpClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # ─── Internal JSON-RPC ────────────────────────────────────────

    def _request(
        self, method: str, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        msg_id = self._next_id
        self._next_id += 1

        event = threading.Event()
        with self._lock:
            self._pending[msg_id] = event

        message = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
            "params": params,
        }
        self._send(message)

        timeout_s = self._config.timeout_ms / 1000.0
        if not event.wait(timeout=timeout_s):
            with self._lock:
                self._pending.pop(msg_id, None)
            raise TimeoutError(
                f'MCP request "{method}" timed out after {self._config.timeout_ms}ms'
            )

        with self._lock:
            if msg_id in self._errors:
                error = self._errors.pop(msg_id)
                raise RuntimeError(f"MCP error: {error}")
            return self._results.pop(msg_id, None)

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        self._send(message)

    def _send(self, message: dict[str, Any]) -> None:
        if not self._process or not self._process.stdin:
            raise RuntimeError("MCP server process not available")
        data = json.dumps(message) + "\n"
        self._process.stdin.write(data.encode())
        self._process.stdin.flush()

    def _read_stdout(self) -> None:
        """Background thread reading JSON-RPC responses from stdout."""
        assert self._process and self._process.stdout
        buffer = ""
        while True:
            try:
                chunk = self._process.stdout.read(4096)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        msg_id = msg.get("id")
                        if msg_id is not None:
                            with self._lock:
                                if msg_id in self._pending:
                                    if "error" in msg:
                                        err = msg["error"]
                                        self._errors[msg_id] = (
                                            f'{err.get("code", 0)}: {err.get("message", "")}'
                                        )
                                    else:
                                        self._results[msg_id] = msg.get("result")
                                    self._pending[msg_id].set()
                    except json.JSONDecodeError:
                        pass
            except (OSError, ValueError):
                break

        # Wake up any remaining pending requests
        with self._lock:
            for event in self._pending.values():
                event.set()

    def _assert_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("McpClient is not connected. Call connect() first.")
