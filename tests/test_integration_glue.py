"""Tests for M35 (Agent Integration Glue) and M36 (Typed Tool System) — Python SDK."""
from __future__ import annotations

import json
import sys
import pytest
from unittest.mock import MagicMock, patch

from gauss.tool import tool, TypedToolDef, create_tool_executor
from gauss.mcp_client import McpClient, McpClientConfig

# ── Mock native module before importing Agent ────────────────────────
_mock_native = MagicMock()
_mock_native.create_provider.return_value = 42
_mock_native.destroy_provider.return_value = None
_mock_native.agent_run.return_value = json.dumps({
    "text": "response", "steps": 1, "inputTokens": 10, "outputTokens": 5,
})
_mock_native.agent_run_with_tool_executor.return_value = json.dumps({
    "text": "tool response", "steps": 2, "inputTokens": 15, "outputTokens": 8,
})
_mock_native.generate.return_value = json.dumps({"text": "Hello"})
_mock_native.get_provider_capabilities.return_value = json.dumps({})


@pytest.fixture(autouse=True)
def _patch_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gauss._native", _mock_native)
    _mock_native.reset_mock()
    # Re-set return values after reset
    _mock_native.create_provider.return_value = 42
    _mock_native.destroy_provider.return_value = None
    _mock_native.agent_run.return_value = json.dumps({
        "text": "response", "steps": 1, "inputTokens": 10, "outputTokens": 5,
    })
    _mock_native.agent_run_with_tool_executor.return_value = json.dumps({
        "text": "tool response", "steps": 2, "inputTokens": 15, "outputTokens": 8,
    })
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")


from gauss.agent import Agent


# ─── M36: @tool decorator ────────────────────────────────────────────

class TestToolDecorator:
    def test_creates_typed_tool_from_function(self):
        @tool
        def search(query: str, limit: int = 10) -> dict:
            """Search the web for results."""
            return {"results": [query], "limit": limit}

        assert isinstance(search, TypedToolDef)
        assert search.name == "search"
        assert search.description == "Search the web for results."
        props = search.parameters["properties"]
        assert "query" in props
        assert props["query"]["type"] == "string"
        assert "limit" in props
        assert props["limit"]["type"] == "integer"
        assert props["limit"]["default"] == 10

    def test_extracts_name_and_description(self):
        @tool
        def my_tool():
            """This is the description."""
            pass

        assert my_tool.name == "my_tool"
        assert my_tool.description == "This is the description."

    def test_tool_with_custom_name(self):
        @tool(name="custom_name", description="Custom desc")
        def helper(x: int) -> str:
            return str(x)

        assert helper.name == "custom_name"
        assert helper.description == "Custom desc"

    def test_tool_execute_works(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = add.execute(a=2, b=3)
        assert result == 5

    def test_tool_callable(self):
        @tool
        def multiply(x: int, y: int) -> int:
            """Multiply."""
            return x * y

        # Should still be callable directly
        assert multiply(x=3, y=4) == 12

    def test_type_mapping(self):
        @tool
        def complex_fn(s: str, i: int, f: float, b: bool) -> None:
            """Test types."""
            pass

        props = complex_fn.parameters["properties"]
        assert props["s"]["type"] == "string"
        assert props["i"]["type"] == "integer"
        assert props["f"]["type"] == "number"
        assert props["b"]["type"] == "boolean"

    def test_no_docstring_gives_empty_description(self):
        @tool
        def no_docs(x: int) -> int:
            return x

        assert no_docs.description == ""

    def test_to_dict_excludes_execute(self):
        @tool
        def fn(x: str) -> str:
            """Test."""
            return x

        d = fn.to_dict()
        assert "name" in d
        assert "description" in d
        assert "parameters" in d
        assert "execute" not in d


# ─── M36: create_tool_executor ───────────────────────────────────────

class TestCreateToolExecutor:
    def test_dispatches_to_correct_tool(self):
        @tool
        def add(a: int, b: int) -> dict:
            """Add."""
            return {"sum": a + b}

        @tool
        def sub(a: int, b: int) -> dict:
            """Subtract."""
            return {"diff": a - b}

        executor = create_tool_executor([add, sub])
        result = executor(json.dumps({"tool": "add", "args": {"a": 5, "b": 3}}))
        assert json.loads(result) == {"sum": 8}

        result = executor(json.dumps({"tool": "sub", "args": {"a": 5, "b": 3}}))
        assert json.loads(result) == {"diff": 2}

    def test_unknown_tool_returns_error(self):
        executor = create_tool_executor([])
        result = executor(json.dumps({"tool": "unknown", "args": {}}))
        assert "error" in json.loads(result)

    def test_execute_error_returns_error(self):
        @tool
        def failing(x: int) -> int:
            """Fail."""
            raise ValueError("boom")

        executor = create_tool_executor([failing])
        result = executor(json.dumps({"tool": "failing", "args": {"x": 1}}))
        parsed = json.loads(result)
        assert "error" in parsed
        assert "boom" in parsed["error"]

    def test_string_return_from_execute(self):
        @tool
        def echo(msg: str) -> str:
            """Echo."""
            return msg

        executor = create_tool_executor([echo])
        result = executor(json.dumps({"tool": "echo", "args": {"msg": "hello"}}))
        assert result == "hello"

    def test_fallback_for_unmatched(self):
        @tool
        def t1(x: int) -> int:
            """Test."""
            return x

        fallback = MagicMock(return_value='{"custom":"fallback"}')
        executor = create_tool_executor([t1], fallback=fallback)
        result = executor(json.dumps({"tool": "other", "args": {}}))
        assert result == '{"custom":"fallback"}'
        fallback.assert_called_once()


# ─── M35: Agent integration methods ─────────────────────────────────

class TestAgentWithMiddleware:
    def test_with_middleware_returns_self(self):
        agent = Agent(api_key="sk-test-key")
        middleware = MagicMock()

        result = agent.with_middleware(middleware)

        assert result is agent  # chainable

    def test_with_guardrails_returns_self(self):
        agent = Agent(api_key="sk-test-key")
        guardrails = MagicMock()

        result = agent.with_guardrails(guardrails)

        assert result is agent

    def test_with_memory_returns_self(self):
        agent = Agent(api_key="sk-test-key")
        memory = MagicMock()

        result = agent.with_memory(memory, session_id="s1")

        assert result is agent


class TestAgentWithTypedTools:
    def test_typed_tools_use_tool_executor_run(self):
        @tool
        def calc(expr: str) -> dict:
            """Calculate."""
            return {"result": 42}

        agent = Agent(api_key="sk-test-key")
        agent.add_tool(calc)

        result = agent.run("Calculate 6*7")

        _mock_native.agent_run_with_tool_executor.assert_called_once()

    def test_tool_defs_sent_without_execute(self):
        @tool
        def greet(name: str) -> str:
            """Greet."""
            return f"Hello {name}"

        agent = Agent(api_key="sk-test-key")
        agent.add_tool(greet)
        agent.run("test")

        call_args = _mock_native.agent_run_with_tool_executor.call_args
        tool_defs = call_args[0][2] if len(call_args[0]) > 2 else []
        for td in tool_defs:
            assert "execute" not in td


class TestAgentWithMemory:
    def test_memory_recall_before_run(self):
        memory = MagicMock()
        memory.recall.return_value = [
            {"id": "1", "content": "User likes cats", "entryType": "fact", "timestamp": "2024-01-01"}
        ]
        memory._handle = 99

        agent = Agent(api_key="sk-test-key")
        agent.with_memory(memory, session_id="test")

        agent.run("What do I like?")

        memory.recall.assert_called_once()

    def test_memory_store_after_run(self):
        memory = MagicMock()
        memory.recall.return_value = []
        memory._handle = 99

        agent = Agent(api_key="sk-test-key")
        agent.with_memory(memory)

        agent.run("Hello!")

        # Should store user message + assistant response
        assert memory.store_sync.call_count == 2


# ─── M37: McpClient config ──────────────────────────────────────────

class TestMcpClientConfig:
    def test_config_creation(self):
        config = McpClientConfig(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
        assert config.command == "npx"
        assert config.args == ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]

    def test_agent_use_mcp_server(self):
        agent = Agent(api_key="sk-test-key")

        config = McpClientConfig(command="echo", args=["test"])
        result = agent.use_mcp_server(config)

        assert result is agent  # chainable
