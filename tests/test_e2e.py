"""E2E tests for the Gauss Python SDK (M43)."""
from __future__ import annotations

import json
import sys
import pytest
from unittest.mock import MagicMock

from gauss.tool import tool, TypedToolDef, create_tool_executor
from gauss.mcp_client import McpClientConfig
from gauss.tokens import set_pricing, get_pricing, clear_pricing, ModelPricing

# ── Mock native module ───────────────────────────────────────────────
_mock = MagicMock()
_mock.create_provider.return_value = 1
_mock.destroy_provider.return_value = None
_mock.agent_run.return_value = json.dumps({
    "text": "response", "steps": 1, "inputTokens": 10, "outputTokens": 5,
})
_mock.agent_run_with_tool_executor.return_value = json.dumps({
    "text": "tool response", "steps": 2, "inputTokens": 15, "outputTokens": 8,
})
_mock.generate.return_value = json.dumps({"text": "Hello"})
_mock.get_provider_capabilities.return_value = json.dumps({})
_mock.create_memory.return_value = 2
_mock.memory_store.return_value = None
_mock.memory_recall.return_value = json.dumps([])
_mock.memory_clear.return_value = None
_mock.destroy_memory.return_value = None
_mock.create_middleware_chain.return_value = 3
_mock.middleware_use_logging.return_value = None
_mock.middleware_use_caching.return_value = None
_mock.middleware_use_rate_limit.return_value = None
_mock.destroy_middleware_chain.return_value = None
_mock.create_guardrail_chain.return_value = 4
_mock.guardrail_chain_add_pii_detection.return_value = None
_mock.guardrail_chain_add_content_moderation.return_value = None
_mock.guardrail_chain_add_token_limit.return_value = None
_mock.guardrail_chain_list.return_value = json.dumps([])
_mock.destroy_guardrail_chain.return_value = None
_mock.create_team.return_value = 5
_mock.team_add_agent.return_value = None
_mock.team_set_strategy.return_value = None
_mock.team_run.return_value = json.dumps({"text": "team result"})
_mock.destroy_team.return_value = None
_mock.create_graph.return_value = 6
_mock.graph_add_node.return_value = None
_mock.graph_add_edge.return_value = None
_mock.graph_run.return_value = json.dumps({"node1": {"text": "result"}})
_mock.destroy_graph.return_value = None
_mock.estimate_cost.return_value = json.dumps({
    "model": "gpt-4o", "input_cost_usd": 0.001,
    "output_cost_usd": 0.002, "total_cost_usd": 0.003,
})
_mock.count_tokens.return_value = 5
_mock.count_tokens_for_model.return_value = 5
_mock.get_context_window_size.return_value = 128000
_mock.create_network.return_value = 7
_mock.network_add_agent.return_value = None
_mock.network_set_supervisor.return_value = None
_mock.network_delegate.return_value = json.dumps({"text": "delegated"})
_mock.destroy_network.return_value = None


@pytest.fixture(autouse=True)
def _patch_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gauss._native", _mock)
    _mock.reset_mock()
    # Reset return values
    _mock.create_provider.return_value = 1
    _mock.destroy_provider.return_value = None
    _mock.agent_run.return_value = json.dumps({
        "text": "response", "steps": 1, "inputTokens": 10, "outputTokens": 5,
    })
    _mock.agent_run_with_tool_executor.return_value = json.dumps({
        "text": "tool response", "steps": 2, "inputTokens": 15, "outputTokens": 8,
    })
    _mock.generate.return_value = json.dumps({"text": "Hello"})
    _mock.create_memory.return_value = 2
    _mock.memory_store.return_value = None
    _mock.memory_recall.return_value = json.dumps([])
    _mock.destroy_memory.return_value = None
    _mock.create_team.return_value = 5
    _mock.team_run.return_value = json.dumps({"text": "team result"})
    _mock.create_graph.return_value = 6
    _mock.graph_run.return_value = json.dumps({"node1": {"text": "result"}})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    clear_pricing()


from gauss.agent import Agent
from gauss.memory import Memory
from gauss.middleware import MiddlewareChain
from gauss.guardrail import GuardrailChain
from gauss.team import Team
from gauss.graph import Graph
from gauss.tokens import estimate_cost, count_tokens


# ─── E2E 1: Basic Agent lifecycle ────────────────────────────────────

class TestE2EBasicAgent:
    def test_create_run_destroy(self):
        agent = Agent(api_key="sk-test")
        result = agent.run("Hello!")

        assert result.text == "response"
        assert isinstance(result.usage, dict)

        _mock.agent_run.assert_called_once()
        agent.destroy()
        _mock.destroy_provider.assert_called_once()

    def test_agent_generate(self):
        agent = Agent(api_key="sk-test")
        result = agent.generate([{"role": "user", "content": "Hi"}])

        assert result is not None
        _mock.generate.assert_called_once()
        agent.destroy()


# ─── E2E 2: Typed tools ──────────────────────────────────────────────

class TestE2ETypedTools:
    def test_tool_decorator_and_agent_execution(self):
        @tool
        def calculator(expression: str) -> dict:
            """Evaluate a math expression."""
            return {"result": eval(expression)}

        assert isinstance(calculator, TypedToolDef)
        assert calculator.name == "calculator"

        agent = Agent(api_key="sk-test")
        agent.add_tool(calculator)
        result = agent.run("What is 2+2?")

        assert result.text == "tool response"
        _mock.agent_run_with_tool_executor.assert_called_once()
        agent.destroy()

    def test_multiple_typed_tools(self):
        @tool
        def search(query: str) -> dict:
            """Search."""
            return {"results": [query]}

        @tool
        def fetch(url: str) -> str:
            """Fetch URL."""
            return f"Content from {url}"

        agent = Agent(api_key="sk-test")
        agent.add_tools([search, fetch])
        result = agent.run("Search and fetch")

        _mock.agent_run_with_tool_executor.assert_called_once()
        agent.destroy()


# ─── E2E 3: Middleware chain ──────────────────────────────────────────

class TestE2EMiddleware:
    def test_full_middleware_chain(self):
        chain = MiddlewareChain()
        chain.use_logging()
        chain.use_caching(ttl_ms=5000)
        chain.use_rate_limit(requests_per_minute=100)

        _mock.middleware_use_logging.assert_called_once()
        _mock.middleware_use_caching.assert_called_once()
        _mock.middleware_use_rate_limit.assert_called_once()

        agent = Agent(api_key="sk-test")
        result = agent.with_middleware(chain)

        assert result is agent  # chainable
        agent.run("test")
        agent.destroy()
        chain.destroy()


# ─── E2E 4: Guardrail chain ──────────────────────────────────────────

class TestE2EGuardrails:
    def test_full_guardrail_chain(self):
        chain = GuardrailChain()
        chain.add_pii_detection("redact")
        chain.add_content_moderation(blocked_categories=["bad"], warned_categories=["risky"])
        chain.add_token_limit(max_input=1000, max_output=500)

        _mock.guardrail_chain_add_pii_detection.assert_called_once()
        _mock.guardrail_chain_add_content_moderation.assert_called_once()
        _mock.guardrail_chain_add_token_limit.assert_called_once()

        agent = Agent(api_key="sk-test")
        agent.with_guardrails(chain)
        agent.run("test")
        agent.destroy()
        chain.destroy()


# ─── E2E 5: Memory integration ───────────────────────────────────────

class TestE2EMemory:
    def test_memory_multi_turn(self):
        memory = MagicMock()
        memory.recall.return_value = []
        memory._handle = 2

        agent = Agent(api_key="sk-test")
        agent.with_memory(memory, session_id="e2e-test")

        # Turn 1
        result1 = agent.run("What is AI?")
        assert result1.text == "response"
        memory.recall.assert_called()

        # Turn 2 — simulate recalled context
        memory.recall.return_value = [
            {"id": "1", "content": "AI is artificial intelligence", "entryType": "fact"}
        ]
        result2 = agent.run("Tell me more")
        assert result2.text == "response"
        assert memory.recall.call_count == 2
        assert memory.store_sync.call_count >= 2

        agent.destroy()


# ─── E2E 6: Multi-agent team ─────────────────────────────────────────

class TestE2ETeam:
    def test_sequential_team(self):
        team = Team("research-team")

        agent1 = Agent(api_key="sk-test", name="researcher")
        agent2 = Agent(api_key="sk-test", name="writer")

        team.add(agent1, instructions="Research the topic")
        team.add(agent2, instructions="Write a summary")
        team.strategy("sequential")

        _mock.team_add_agent.assert_called()
        _mock.team_set_strategy.assert_called_once()

        team.destroy()
        agent1.destroy()
        agent2.destroy()


# ─── E2E 7: Graph DAG ────────────────────────────────────────────────

class TestE2EGraph:
    def test_dag_execution(self):
        agent1 = Agent(api_key="sk-test", name="analyzer")
        agent2 = Agent(api_key="sk-test", name="summarizer")

        graph = Graph()
        graph.add_node("analyze", agent1, instructions="Analyze the input")
        graph.add_node("summarize", agent2, instructions="Summarize results")
        graph.add_edge("analyze", "summarize")

        _mock.graph_add_node.assert_called()
        _mock.graph_add_edge.assert_called_once()

        graph.destroy()
        agent1.destroy()
        agent2.destroy()


# ─── E2E 8: Cost tracking ────────────────────────────────────────────

class TestE2ECostTracking:
    def test_estimate_cost_for_run(self):
        agent = Agent(api_key="sk-test")
        result = agent.run("Hello")

        # Use known token counts from the run
        cost = estimate_cost("gpt-4o", 10, 5)

        assert cost.model == "gpt-4o"
        assert cost.total_cost_usd == 0.003
        agent.destroy()

    def test_count_tokens(self):
        n = count_tokens("Hello, world!")
        assert n == 5


# ─── E2E 9: Enterprise preset ────────────────────────────────────────

class TestE2EEnterprisePreset:
    def test_enterprise_agent(self):
        from gauss.agent import enterprise_preset

        agent = enterprise_preset(api_key="sk-test")
        assert agent is not None
        result = agent.run("Enterprise query")
        assert result.text == "response"
        agent.destroy()


# ─── E2E 10: Pricing override ────────────────────────────────────────

class TestE2EPricingOverride:
    def test_set_and_get_pricing(self):
        pricing = ModelPricing(
            input_per_token=0.000003,
            output_per_token=0.000015,
        )
        set_pricing("my-model", pricing)

        retrieved = get_pricing("my-model")
        assert retrieved is not None
        assert retrieved.input_per_token == 0.000003
        assert retrieved.output_per_token == 0.000015

    def test_custom_pricing_in_estimate(self):
        set_pricing("custom-model", ModelPricing(
            input_per_token=0.00001,
            output_per_token=0.00003,
        ))

        cost = estimate_cost("custom-model", 1000, 500)

        assert cost.model == "custom-model"
        assert cost.input_cost_usd == pytest.approx(0.01)
        assert cost.output_cost_usd == pytest.approx(0.015)
        assert cost.total_cost_usd == pytest.approx(0.025)

    def test_clear_pricing(self):
        set_pricing("temp-model", ModelPricing(input_per_token=0.001, output_per_token=0.002))
        assert get_pricing("temp-model") is not None

        clear_pricing()
        assert get_pricing("temp-model") is None
