"""
E2E tests for MCP (Model Context Protocol) and A2A (Agent-to-Agent) protocol.

MCP tests exercise the real Rust-backed McpServer in-process.
A2A tests use a mock HTTP server simulating an A2A-compliant agent.

Run:
    python -m pytest tests/e2e/test_mcp_a2a.py -v
"""

from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

import pytest

from gauss import (
    A2aClient,
    A2aMessage,
    McpPrompt,
    McpPromptArgument,
    McpResource,
    McpServer,
)


# ══════════════════════════════════════════════════════════════════════════════
# MCP Server Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestMcpServerE2E:
    """End-to-end tests for McpServer registration and message handling."""

    def test_create_and_register(self) -> None:
        with McpServer("test-mcp", "1.0.0") as server:
            server.add_tool(
                {
                    "name": "calc",
                    "description": "Calculator",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"expr": {"type": "string"}},
                        "required": ["expr"],
                    },
                }
            )
            server.add_resource(McpResource(uri="file:///readme.md", name="README"))
            server.add_prompt(
                McpPrompt(
                    name="summarize",
                    arguments=[McpPromptArgument(name="text", required=True)],
                )
            )
            # No assertion needed — if the native calls didn't throw, registration worked.

    async def test_tools_list(self) -> None:
        with McpServer("tools-e2e", "1.0.0") as server:
            server.add_tool(
                {"name": "search", "description": "Web search", "inputSchema": {"type": "object"}}
            )
            server.add_tool(
                {"name": "weather", "description": "Get weather", "inputSchema": {"type": "object"}}
            )

            resp = await server.handle_message(
                {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
            )

            tools = resp["result"]["tools"]
            assert len(tools) == 2
            names = {t["name"] for t in tools}
            assert names == {"search", "weather"}

    async def test_resources_list(self) -> None:
        with McpServer("res-e2e", "1.0.0") as server:
            server.add_resource(McpResource(uri="file:///a.txt", name="FileA", description="First"))
            server.add_resource(
                McpResource(uri="file:///b.json", name="FileB", mime_type="application/json")
            )

            resp = await server.handle_message(
                {"jsonrpc": "2.0", "id": 2, "method": "resources/list"}
            )

            resources = resp["result"]["resources"]
            assert len(resources) == 2
            uris = {r["uri"] for r in resources}
            assert uris == {"file:///a.txt", "file:///b.json"}

    async def test_prompts_list(self) -> None:
        with McpServer("prompts-e2e", "1.0.0") as server:
            server.add_prompt(
                McpPrompt(
                    name="greet",
                    description="Greeting prompt",
                    arguments=[McpPromptArgument(name="name", required=True)],
                )
            )

            resp = await server.handle_message(
                {"jsonrpc": "2.0", "id": 3, "method": "prompts/list"}
            )

            prompts = resp["result"]["prompts"]
            assert len(prompts) == 1
            assert prompts[0]["name"] == "greet"

    async def test_ping(self) -> None:
        with McpServer("ping-e2e", "1.0.0") as server:
            resp = await server.handle_message(
                {"jsonrpc": "2.0", "id": 4, "method": "ping"}
            )
            assert resp["result"] == {}

    async def test_tools_call_without_handler(self) -> None:
        """tools/call returns an error when no execute function is registered."""
        with McpServer("call-e2e", "1.0.0") as server:
            server.add_tool(
                {
                    "name": "calc",
                    "description": "Calculator",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"expr": {"type": "string"}},
                    },
                }
            )

            resp = await server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {"name": "calc", "arguments": {"expr": "2+2"}},
                }
            )

            assert "error" in resp
            assert resp["error"]["code"] == -32603

    async def test_full_builder_chain(self) -> None:
        with McpServer("chain-e2e", "2.0.0") as server:
            server.add_tool(
                {"name": "t1", "description": "Tool1", "inputSchema": {"type": "object"}}
            ).add_resource(
                McpResource(uri="file:///data.json", name="Data")
            ).add_prompt(
                McpPrompt(name="analyze", arguments=[])
            )

            # Verify all registrations via list methods
            tools_resp = await server.handle_message(
                {"jsonrpc": "2.0", "id": 10, "method": "tools/list"}
            )
            res_resp = await server.handle_message(
                {"jsonrpc": "2.0", "id": 11, "method": "resources/list"}
            )
            prompts_resp = await server.handle_message(
                {"jsonrpc": "2.0", "id": 12, "method": "prompts/list"}
            )

            assert len(tools_resp["result"]["tools"]) == 1
            assert len(res_resp["result"]["resources"]) == 1
            assert len(prompts_resp["result"]["prompts"]) == 1


# ══════════════════════════════════════════════════════════════════════════════
# A2A Client Tests
# ══════════════════════════════════════════════════════════════════════════════


class _MockA2AHandler(BaseHTTPRequestHandler):
    """Mock A2A-compliant agent server."""

    def do_GET(self) -> None:  # noqa: N802
        """Serve /.well-known/agent.json"""
        port = self.server.server_address[1]
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        card = {
            "name": "Mock A2A Agent",
            "url": f"http://127.0.0.1:{port}",
            "description": "A mock A2A agent for E2E tests",
            "version": "1.0.0",
            "capabilities": {
                "streaming": False,
                "pushNotifications": False,
                "stateTransitionHistory": False,
            },
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
            "skills": [
                {
                    "id": "echo",
                    "name": "Echo",
                    "description": "Echoes input back",
                    "tags": ["echo", "test"],
                    "examples": ["Hello"],
                    "inputModes": ["text"],
                    "outputModes": ["text"],
                }
            ],
        }
        self.wfile.write(json.dumps(card).encode())

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        method = body.get("method", "")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        # Extract user text from the request
        user_text = ""
        params = body.get("params", {})
        message = params.get("message", {})
        for part in message.get("parts", []):
            if part.get("type") == "text":
                user_text = part.get("text", "")
                break
        # For a2a_ask, text may come as top-level param
        if not user_text:
            user_text = params.get("text", "echo")

        echo_text = f"Echo: {user_text}"

        if method in ("message/send", "tasks/send"):
            result = {
                "type": "task",
                "id": "task-e2e-1",
                "status": {
                    "state": "completed",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "message": {
                        "role": "agent",
                        "parts": [{"type": "text", "text": echo_text}],
                    },
                },
                "artifacts": [
                    {"parts": [{"type": "text", "text": echo_text}], "name": "reply"}
                ],
                "history": [
                    {"role": "agent", "parts": [{"type": "text", "text": echo_text}]}
                ],
                "messages": [
                    {"role": "agent", "parts": [{"type": "text", "text": echo_text}]}
                ],
            }
        elif method == "tasks/get":
            result = {
                "type": "task",
                "id": params.get("id", "task-e2e-1"),
                "status": {
                    "state": "completed",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "message": {
                        "role": "agent",
                        "parts": [{"type": "text", "text": "task retrieved"}],
                    },
                },
                "artifacts": [],
                "history": [],
                "messages": [],
            }
        else:
            result = {}

        resp = {"jsonrpc": "2.0", "id": body.get("id", 1), "result": result}
        self.wfile.write(json.dumps(resp).encode())

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


@pytest.fixture()
def mock_a2a_server() -> tuple[str, HTTPServer]:
    server = HTTPServer(("127.0.0.1", 0), _MockA2AHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", server
    server.shutdown()


class TestA2aClientE2E:
    async def test_discover(self, mock_a2a_server: tuple[str, HTTPServer]) -> None:
        base_url, _ = mock_a2a_server
        client = A2aClient(base_url)
        card = await client.discover()

        assert card.name == "Mock A2A Agent"
        assert card.version == "1.0.0"
        assert card.url == base_url
        assert card.capabilities is not None
        assert card.capabilities.streaming is False
        assert len(card.skills) == 1
        assert card.skills[0].id == "echo"
        assert card.skills[0].name == "Echo"
        assert "echo" in card.skills[0].tags

    async def test_send_message(self, mock_a2a_server: tuple[str, HTTPServer]) -> None:
        base_url, _ = mock_a2a_server
        client = A2aClient(base_url)
        result = await client.send_message(A2aMessage.user_text("world"))

        # send_message returns a Task (since response has 'status')
        from gauss.a2a import Task

        assert isinstance(result, Task)
        assert result.id == "task-e2e-1"
        assert result.status.state.value == "completed"
        assert result.text == "Echo: world"

    async def test_ask(self, mock_a2a_server: tuple[str, HTTPServer]) -> None:
        base_url, _ = mock_a2a_server
        client = A2aClient(base_url)
        answer = await client.ask("ping")

        assert isinstance(answer, str)
        assert "Echo:" in answer

    async def test_client_with_auth_token(
        self, mock_a2a_server: tuple[str, HTTPServer]
    ) -> None:
        base_url, _ = mock_a2a_server
        client = A2aClient(base_url, auth_token="test-token-123")
        card = await client.discover()
        assert card.name == "Mock A2A Agent"
