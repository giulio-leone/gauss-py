"""Tests for Gauss error hierarchy."""

from __future__ import annotations

import json
import sys
from typing import NoReturn
from unittest.mock import MagicMock

import pytest
from gauss.errors import (
    DisposedError,
    GaussError,
    ProviderError,
    ToolExecutionError,
    ValidationError,
)

# ---------------------------------------------------------------------------
# Mock the native module
# ---------------------------------------------------------------------------
_mock_native = MagicMock()
_mock_native.create_provider.return_value = 1
_mock_native.destroy_provider.return_value = None
_mock_native.agent_run.return_value = json.dumps({
    "text": "ok",
    "messages": [],
    "toolCalls": [],
    "usage": {"total_tokens": 1},
})


@pytest.fixture(autouse=True)
def _mock_gauss_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gauss._native", _mock_native)
    _mock_native.reset_mock()
    _mock_native.create_provider.return_value = 1
    _mock_native.destroy_provider.return_value = None
    _mock_native.agent_run.return_value = json.dumps({
        "text": "ok",
        "messages": [],
        "toolCalls": [],
        "usage": {"total_tokens": 1},
    })
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")


class TestErrorHierarchy:
    def test_gauss_error_is_exception(self) -> None:
        assert issubclass(GaussError, Exception)

    def test_disposed_error_inherits_gauss(self) -> None:
        assert issubclass(DisposedError, GaussError)

    def test_provider_error_inherits_gauss(self) -> None:
        assert issubclass(ProviderError, GaussError)

    def test_tool_error_inherits_gauss(self) -> None:
        assert issubclass(ToolExecutionError, GaussError)

    def test_validation_error_inherits_gauss(self) -> None:
        assert issubclass(ValidationError, GaussError)

    def test_disposed_error_attributes(self) -> None:
        err = DisposedError("Agent", "my-agent")
        assert err.code == "RESOURCE_DISPOSED"
        assert err.resource_type == "Agent"
        assert err.resource_name == "my-agent"
        assert "my-agent" in str(err)

    def test_provider_error_attributes(self) -> None:
        err = ProviderError("openai", "rate limit exceeded")
        assert err.code == "PROVIDER_ERROR"
        assert err.provider == "openai"
        assert "[openai]" in str(err)

    def test_tool_error_attributes(self) -> None:
        err = ToolExecutionError("search", "timeout")
        assert err.code == "TOOL_EXECUTION_ERROR"
        assert err.tool_name == "search"
        assert "search" in str(err)

    def test_validation_error_with_field(self) -> None:
        err = ValidationError("must be positive", "temperature")
        assert err.code == "VALIDATION_ERROR"
        assert err.field == "temperature"
        assert "temperature" in str(err)

    def test_validation_error_without_field(self) -> None:
        err = ValidationError("config is invalid")
        assert err.field is None

    def test_catch_gauss_error_catches_all(self) -> NoReturn:
        for err_class in [DisposedError, ProviderError, ToolExecutionError, ValidationError]:
            with pytest.raises(GaussError):
                if err_class == DisposedError:
                    raise err_class("Agent", "test")
                elif err_class == ProviderError:
                    raise err_class("openai", "fail")
                elif err_class == ToolExecutionError:
                    raise err_class("tool", "fail")
                else:
                    raise err_class("fail")


class TestAgentDisposedError:
    def test_destroyed_agent_raises_disposed_error(self) -> None:
        from gauss import Agent

        agent = Agent()
        agent.destroy()
        with pytest.raises(DisposedError) as exc_info:
            agent.run("hello")
        assert exc_info.value.resource_type == "Agent"


class TestWithTool:
    def test_with_tool_returns_self(self) -> None:
        from gauss import Agent

        agent = Agent()
        result = agent.with_tool("test", "A test tool", {"x": {"type": "string"}})
        assert result is agent

    def test_with_tool_adds_tool(self) -> None:
        from gauss import Agent

        agent = Agent()
        agent.with_tool("calc", "Calculator", {"expr": {"type": "string"}}, execute=lambda expr: "42")
        assert any(t.name == "calc" for t in agent._tools if hasattr(t, "name"))

    def test_with_tool_chaining(self) -> None:
        from gauss import Agent

        agent = Agent()
        agent.with_tool("a", "Tool A").with_tool("b", "Tool B")
        tool_names = [t.name for t in agent._tools if hasattr(t, "name")]
        assert "a" in tool_names
        assert "b" in tool_names
