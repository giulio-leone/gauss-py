"""Unit tests for async Agent methods (arun, agenerate, arun_with_tools)."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock the native module globally
# ---------------------------------------------------------------------------
_mock_native = MagicMock()
_mock_native.create_provider.return_value = 1
_mock_native.destroy_provider.return_value = None
_mock_native.agent_run.return_value = json.dumps({
    "text": "The answer is 42",
    "messages": [{"role": "assistant", "content": "The answer is 42"}],
    "toolCalls": [],
    "usage": {"total_tokens": 10},
})
_mock_native.agent_run_with_tool_executor.return_value = json.dumps({
    "text": "The answer is 42",
    "messages": [{"role": "assistant", "content": "The answer is 42"}],
    "toolCalls": [],
    "usage": {"total_tokens": 10},
})
_mock_native.generate.return_value = json.dumps({"text": "Hello from Gauss!"})


@pytest.fixture(autouse=True)
def _mock_gauss_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch gauss._native with mock for all tests."""
    monkeypatch.setitem(sys.modules, "gauss._native", _mock_native)
    _mock_native.reset_mock()
    # Set env for auto-detection
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")


# ===========================================================================
# Async Agent Tests
# ===========================================================================

class TestAsyncAgent:
    @pytest.mark.asyncio
    async def test_arun_returns_agent_result(self) -> None:
        """Test that arun returns an AgentResult."""
        from gauss.agent import Agent
        from gauss._types import AgentResult
        
        agent = Agent()
        result = await agent.arun("test")
        assert isinstance(result, AgentResult)
        assert result.text == "The answer is 42"
        _mock_native.agent_run.assert_called_once()
        agent.destroy()

    @pytest.mark.asyncio
    async def test_agenerate_returns_string(self) -> None:
        """Test that agenerate returns a string."""
        from gauss.agent import Agent
        
        agent = Agent()
        text = await agent.agenerate("test")
        assert isinstance(text, str)
        assert text == "Hello from Gauss!"
        _mock_native.generate.assert_called_once()
        agent.destroy()

    @pytest.mark.asyncio
    async def test_arun_with_tools_calls_executor(self) -> None:
        """Test that arun_with_tools invokes the tool executor."""
        import json
        from gauss.agent import Agent
        from gauss._types import AgentResult
        
        def mock_executor(call_json: str) -> str:
            """Tool executor that accepts JSON string and returns JSON string."""
            data = json.loads(call_json)
            return json.dumps({"result": f"Executed {data.get('tool', 'unknown')}"})
        
        agent = Agent()
        result = await agent.arun_with_tools("test", mock_executor)
        assert isinstance(result, AgentResult)
        assert result.text == "The answer is 42"
        _mock_native.agent_run_with_tool_executor.assert_called_once()
        agent.destroy()
