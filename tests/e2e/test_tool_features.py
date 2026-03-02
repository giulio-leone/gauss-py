"""
E2E tests for tool-related features: ToolRegistry, ToolValidator, Agent tool-call loop.

ToolRegistry and ToolValidator exercise the real Rust native layer directly.
Agent tool-call tests use a mock HTTP server simulating OpenAI tool_calls.

Run:
    python -m pytest tests/e2e/test_tool_features.py -v
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
from gauss import (
    Agent,
    AgentConfig,
    ProviderType,
    ToolDef,
    ToolRegistry,
    ToolValidator,
)
from gauss.tool_registry import ToolExample, ToolRegistryEntry

# ══════════════════════════════════════════════════════════════════════════════
# ToolRegistry Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestToolRegistryE2E:
    def test_create_and_destroy(self) -> None:
        reg = ToolRegistry()
        assert reg._handle >= 0
        reg.destroy()
        assert reg._destroyed

    def test_add_and_list(self) -> None:
        with ToolRegistry() as reg:
            reg.add(ToolRegistryEntry(name="calc", description="Calculator", tags=["math"]))
            reg.add(ToolRegistryEntry(name="weather", description="Get weather", tags=["api"]))
            reg.add(
                ToolRegistryEntry(
                    name="translate",
                    description="Translate text",
                    tags=["nlp", "api"],
                )
            )

            tools = reg.list()
            assert len(tools) == 3
            names = {t.name for t in tools}
            assert names == {"calc", "weather", "translate"}

    def test_search_by_name(self) -> None:
        with ToolRegistry() as reg:
            reg.add(ToolRegistryEntry(name="calculator", description="Math calculator", tags=["math"]))
            reg.add(ToolRegistryEntry(name="converter", description="Unit converter", tags=["math"]))
            reg.add(ToolRegistryEntry(name="weather", description="Get weather", tags=["api"]))

            results = reg.search("calc")
            assert len(results) == 1
            assert results[0].name == "calculator"

    def test_search_by_description(self) -> None:
        with ToolRegistry() as reg:
            reg.add(ToolRegistryEntry(name="add", description="Add two numbers together"))
            reg.add(ToolRegistryEntry(name="concat", description="Concatenate strings"))

            results = reg.search("numbers")
            assert len(results) == 1
            assert results[0].name == "add"

    def test_search_no_results(self) -> None:
        with ToolRegistry() as reg:
            reg.add(ToolRegistryEntry(name="calc", description="Calculator"))
            assert reg.search("nonexistent") == []

    def test_by_tag(self) -> None:
        with ToolRegistry() as reg:
            reg.add(ToolRegistryEntry(name="add", description="Add", tags=["math", "basic"]))
            reg.add(ToolRegistryEntry(name="plot", description="Plot", tags=["math", "viz"]))
            reg.add(ToolRegistryEntry(name="fetch", description="Fetch", tags=["api"]))

            math_tools = reg.by_tag("math")
            assert len(math_tools) == 2
            assert {t.name for t in math_tools} == {"add", "plot"}

            api_tools = reg.by_tag("api")
            assert len(api_tools) == 1
            assert api_tools[0].name == "fetch"

            assert reg.by_tag("nonexistent") == []

    def test_add_with_examples(self) -> None:
        with ToolRegistry() as reg:
            reg.add(
                ToolRegistryEntry(
                    name="add",
                    description="Add numbers",
                    tags=["math"],
                    examples=[
                        ToolExample(
                            description="Add 2 and 3",
                            input={"a": 2, "b": 3},
                            expected_output=5,
                        ),
                        ToolExample(
                            description="Add negatives",
                            input={"a": -1, "b": -2},
                            expected_output=-3,
                        ),
                    ],
                )
            )

            tools = reg.list()
            assert len(tools) == 1
            assert len(tools[0].examples) == 2
            assert tools[0].examples[0].description == "Add 2 and 3"

    def test_add_dict_format(self) -> None:
        with ToolRegistry() as reg:
            reg.add({"name": "raw-tool", "description": "Added as dict", "tags": ["test"]})
            tools = reg.list()
            assert len(tools) == 1
            assert tools[0].name == "raw-tool"

    def test_chaining(self) -> None:
        with ToolRegistry() as reg:
            result = (
                reg.add(ToolRegistryEntry(name="a", description="A"))
                .add(ToolRegistryEntry(name="b", description="B"))
                .add(ToolRegistryEntry(name="c", description="C"))
            )
            assert result is reg
            assert len(reg.list()) == 3

    def test_throws_after_destroy(self) -> None:
        reg = ToolRegistry()
        reg.destroy()
        with pytest.raises(RuntimeError, match="destroyed"):
            reg.add(ToolRegistryEntry(name="x", description="x"))
        with pytest.raises(RuntimeError, match="destroyed"):
            reg.search("x")
        with pytest.raises(RuntimeError, match="destroyed"):
            reg.list()
        with pytest.raises(RuntimeError, match="destroyed"):
            reg.by_tag("x")


# ══════════════════════════════════════════════════════════════════════════════
# ToolValidator Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestToolValidatorE2E:
    def test_valid_input(self) -> None:
        with ToolValidator() as v:
            result = v.validate(
                {"name": "John", "age": 25},
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
            )
            assert result["name"] == "John"
            assert result["age"] == 25

    def test_type_coercion(self) -> None:
        with ToolValidator(strategies=["type_cast"]) as v:
            result = v.validate(
                {"age": "25"},
                {
                    "type": "object",
                    "properties": {"age": {"type": "integer"}},
                },
            )
            # type_cast coerces "25" to numeric
            assert isinstance(result["age"], (int, float))

    def test_nested_object(self) -> None:
        with ToolValidator() as v:
            result = v.validate(
                {"user": {"name": "Alice", "email": "alice@test.com"}},
                {
                    "type": "object",
                    "properties": {
                        "user": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "email": {"type": "string"},
                            },
                        }
                    },
                },
            )
            assert result["user"]["name"] == "Alice"

    def test_throws_after_destroy(self) -> None:
        v = ToolValidator()
        v.destroy()
        with pytest.raises(RuntimeError, match="destroyed"):
            v.validate({}, {"type": "object"})

    def test_context_manager(self) -> None:
        with ToolValidator() as v:
            result = v.validate({"x": 1}, {"type": "object", "properties": {"x": {"type": "integer"}}})
            assert result["x"] == 1
        assert v._destroyed


# ══════════════════════════════════════════════════════════════════════════════
# Agent Tool-Call Loop Tests
# ══════════════════════════════════════════════════════════════════════════════


class _ToolCallHandler(BaseHTTPRequestHandler):
    """Mock OpenAI server that returns a tool_call on first request,
    then a text response on the follow-up."""

    def do_POST(self) -> None:  # noqa: N802
        server: _ToolCallServer = self.server  # type: ignore[assignment]
        server.call_count += 1

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        server.requests.append(body)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        # Check if this request contains a tool result (follow-up)
        messages = body.get("messages", [])
        has_tool_result = any(m.get("role") == "tool" for m in messages)

        if not has_tool_result and server.call_count == 1:
            # First call: return tool_call
            resp = {
                "id": "chatcmpl-tool-1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-abc123",
                                    "type": "function",
                                    "function": {
                                        "name": "calculate",
                                        "arguments": json.dumps({"expression": "15 * 23"}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
            }
        else:
            # Follow-up: return final text
            resp = {
                "id": "chatcmpl-tool-2",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "The result of 15 * 23 is 345.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
            }

        self.wfile.write(json.dumps(resp).encode())

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


class _ToolCallServer(HTTPServer):
    call_count: int = 0
    requests: list[dict[str, Any]] = []


@pytest.fixture()
def mock_tool_server() -> tuple[str, _ToolCallServer]:
    server = _ToolCallServer(("127.0.0.1", 0), _ToolCallHandler)
    server.call_count = 0
    server.requests = []
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/v1", server
    server.shutdown()


class TestAgentToolCallE2E:
    def test_agent_with_tools(
        self, mock_tool_server: tuple[str, _ToolCallServer]
    ) -> None:
        """Agent receives a tool_call, the native layer sends back a tool result,
        and the final text response includes the answer."""
        base_url, server = mock_tool_server
        agent = Agent(
            AgentConfig(
                name="tool-agent",
                provider=ProviderType.OPENAI,
                model="gpt-4o-mini",
                api_key="sk-fake-e2e",
                base_url=base_url,
                tools=[
                    ToolDef(
                        name="calculate",
                        description="Calculate a math expression",
                        parameters={
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Math expression to evaluate",
                                }
                            },
                            "required": ["expression"],
                        },
                    )
                ],
            )
        )

        result = agent.run("What is 15 * 23?")

        assert "345" in result.text
        # Agent made 2 HTTP calls: initial + tool result follow-up
        assert server.call_count == 2
        # Second request should contain a tool role message
        second_req = server.requests[1]
        tool_msgs = [m for m in second_req["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) >= 1

        agent.destroy()

    def test_agent_no_tools_single_call(
        self, mock_tool_server: tuple[str, _ToolCallServer]
    ) -> None:
        """Agent without tools gets a direct text response (server falls through
        to text response since no tool_call is triggered without tools defined)."""
        base_url, server = mock_tool_server
        agent = Agent(
            AgentConfig(
                name="no-tool-agent",
                provider=ProviderType.OPENAI,
                model="gpt-4o-mini",
                api_key="sk-fake-e2e",
                base_url=base_url,
            )
        )

        # Without tools, the mock still returns a tool_call on first request,
        # but the native layer handles it internally and makes a second call.
        result = agent.run("What is 15 * 23?")
        assert result.text != ""

        agent.destroy()

    def test_agent_tool_definition_visible(self) -> None:
        """Verify tools are attached to the agent config."""
        agent = Agent(
            AgentConfig(
                name="def-check",
                provider=ProviderType.OPENAI,
                model="gpt-4o-mini",
                api_key="sk-fake-e2e",
                base_url="http://127.0.0.1:1/v1",  # won't be called
                tools=[
                    ToolDef(name="search", description="Search the web"),
                    ToolDef(name="calc", description="Calculator"),
                ],
            )
        )
        assert len(agent._tools) == 2
        assert agent._tools[0].name == "search"
        assert agent._tools[1].name == "calc"
        agent.destroy()

    def test_agent_add_tool_chainable(self) -> None:
        agent = Agent(
            AgentConfig(
                name="chain-check",
                provider=ProviderType.OPENAI,
                model="gpt-4o-mini",
                api_key="sk-fake-e2e",
                base_url="http://127.0.0.1:1/v1",
            )
        )
        result = (
            agent.add_tool(ToolDef(name="a", description="A"))
            .add_tool(ToolDef(name="b", description="B"))
        )
        assert result is agent
        assert len(agent._tools) == 2
        agent.destroy()
