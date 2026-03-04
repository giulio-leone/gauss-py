"""Tests for gauss.decorators and Agent DX features (M90)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Mock native layer
# ---------------------------------------------------------------------------
_mock_native = MagicMock()
_mock_native.create_provider.return_value = 1
_mock_native.destroy_provider.return_value = None
_mock_native.agent_run.return_value = json.dumps({
    "text": "hello",
    "messages": [],
    "toolCalls": [],
    "usage": {"input_tokens": 5, "output_tokens": 10, "total_usd": 0.001},
})
_mock_native.get_provider_capabilities.return_value = json.dumps({
    "streaming": True,
    "tool_use": True,
})


# ---------------------------------------------------------------------------
# Decorator tests
# ---------------------------------------------------------------------------


class TestToolDecorator:
    """Tests for @tool decorator from gauss.decorators."""

    def test_basic_tool(self):
        from gauss.decorators import tool

        @tool
        def greet(name: str) -> str:
            """Say hello."""
            return f"Hello, {name}"

        assert greet.is_gauss_tool is True
        assert greet.tool_def["name"] == "greet"

    def test_tool_with_defaults(self):
        from gauss.decorators import tool

        @tool
        def search(query: str, limit: int = 10) -> list:
            """Search items."""
            return []

        schema = search.tool_def["parameters"]
        assert "query" in schema["required"]
        assert "limit" not in schema["required"]

    def test_tool_with_custom_name(self):
        from gauss.decorators import tool

        @tool(name="custom_search", description="My custom tool")
        def search(query: str) -> list:
            return []

        assert search.tool_def["name"] == "custom_search"
        assert search.tool_def["description"] == "My custom tool"

    def test_tool_schema_generation(self):
        from gauss.decorators import tool

        @tool
        def compute(x: int, y: float, flag: bool, items: list) -> dict:
            """Do computation."""
            return {}

        props = compute.tool_def["parameters"]["properties"]
        assert props["x"]["type"] == "integer"
        assert props["y"]["type"] == "number"
        assert props["flag"]["type"] == "boolean"
        assert props["items"]["type"] == "array"

    def test_tool_execution(self):
        from gauss.decorators import tool

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert add(2, 3) == 5
        assert add.tool_def["execute"](2, 3) == 5


class TestAgentDecorator:
    """Tests for @agent decorator from gauss.decorators."""

    def test_agent_creation(self):
        from gauss.decorators import agent

        @agent
        def helper(prompt: str) -> str:
            """A helper."""
            return prompt

        assert helper.is_gauss_agent is True
        assert helper.agent_def["name"] == "helper"
        assert helper.agent_def["model"] == "gpt-4o"

    def test_agent_with_tools(self):
        from gauss.decorators import agent, tool

        @tool
        def my_tool(x: str) -> str:
            return x

        @agent(model="claude-sonnet-4-20250514", tools=[my_tool])
        def assistant(prompt: str) -> str:
            """Smart assistant."""
            return prompt

        assert assistant.agent_def["model"] == "claude-sonnet-4-20250514"
        assert len(assistant.agent_def["tools"]) == 1

    def test_agent_with_instructions(self):
        from gauss.decorators import agent

        @agent(instructions="Be helpful")
        def helper(prompt: str) -> str:
            """Docstring."""
            return prompt

        assert helper.agent_def["instructions"] == "Be helpful"

    def test_agent_max_steps(self):
        from gauss.decorators import agent

        @agent(max_steps=5)
        def helper(prompt: str) -> str:
            return prompt

        assert helper.agent_def["max_steps"] == 5


class TestGuardrailDecorator:
    """Tests for @guardrail decorator from gauss.decorators."""

    def test_guardrail_creation(self):
        from gauss.decorators import guardrail

        @guardrail
        def no_pii(text: str) -> bool:
            """Block PII."""
            return "ssn" not in text.lower()

        assert no_pii.is_gauss_guardrail is True
        assert no_pii.guardrail_def["name"] == "no_pii"

    def test_guardrail_execution(self):
        from gauss.decorators import guardrail

        @guardrail
        def safe_check(text: str) -> bool:
            return len(text) < 100

        assert safe_check("short") is True
        assert safe_check("x" * 200) is False

    def test_guardrail_with_custom_name(self):
        from gauss.decorators import guardrail

        @guardrail(name="pii_filter")
        def check(text: str) -> bool:
            return True

        assert check.guardrail_def["name"] == "pii_filter"


# ---------------------------------------------------------------------------
# Agent.quick / Agent.from_config / cost / trace tests
# ---------------------------------------------------------------------------


_AGENT_PATCHES = {
    "sys.modules": {"gauss._native": _mock_native},
    "os.environ": {"OPENAI_API_KEY": "sk-test-key"},
}


def _mock_routing(policy, provider, model):
    """Bypass routing policy to avoid datetime.UTC on Python <3.11."""
    return provider, model


class TestAgentQuick:
    """Tests for Agent convenience class methods."""

    @patch("gauss.agent.resolve_routing_target", side_effect=_mock_routing)
    @patch.dict("sys.modules", {"gauss._native": _mock_native})
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"})
    def test_quick_creation(self, _rrt):
        from gauss.agent import Agent

        agent = Agent.quick("gpt-4o", "You are helpful")
        assert agent._config.model == "gpt-4o"
        assert agent._config.system_prompt == "You are helpful"

    @patch("gauss.agent.resolve_routing_target", side_effect=_mock_routing)
    @patch.dict("sys.modules", {"gauss._native": _mock_native})
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"})
    def test_from_config(self, _rrt):
        from gauss.agent import Agent

        agent = Agent.from_config({"model": "gpt-4o", "system_prompt": "Hi"})
        assert agent._config.model == "gpt-4o"
        assert agent._config.system_prompt == "Hi"

    @patch("gauss.agent.resolve_routing_target", side_effect=_mock_routing)
    @patch.dict("sys.modules", {"gauss._native": _mock_native})
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"})
    def test_cost_property(self, _rrt):
        from gauss.agent import Agent

        agent = Agent.quick("gpt-4o", "test")
        # Before any run, cost is None
        assert agent.last_run_cost is None

        # After a run, cost is populated from usage
        agent.run("hello")
        cost = agent.last_run_cost
        assert cost is not None
        assert cost["input_tokens"] == 5
        assert cost["output_tokens"] == 10
        assert cost["total_usd"] == 0.001

    @patch("gauss.agent.resolve_routing_target", side_effect=_mock_routing)
    @patch.dict("sys.modules", {"gauss._native": _mock_native})
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-key"})
    def test_trace_property(self, _rrt):
        from gauss.agent import Agent

        agent = Agent.quick("gpt-4o", "test")
        assert agent.last_run_trace is None

        agent.run("hello")
        trace = agent.last_run_trace
        assert trace is not None
        assert trace["model"] == "gpt-4o"
        assert "usage" in trace
