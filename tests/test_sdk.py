"""Unit tests for gauss-py SDK (mocked native layer)."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

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
_mock_native.generate.return_value = json.dumps({"text": "Hello from Gauss!"})
_mock_native.create_memory.return_value = 2
_mock_native.memory_store.return_value = None
_mock_native.memory_recall.return_value = json.dumps([
    {"role": "user", "content": "Hi", "sessionId": "s1"}
])
_mock_native.memory_clear.return_value = None
_mock_native.destroy_memory.return_value = None
_mock_native.create_vector_store.return_value = 3
_mock_native.vector_store_upsert.return_value = None
_mock_native.vector_store_search.return_value = json.dumps([
    {"id": "c1", "text": "hello", "score": 0.95, "metadata": {}}
])
_mock_native.destroy_vector_store.return_value = None
_mock_native.cosine_similarity.return_value = 0.98
_mock_native.create_graph.return_value = 10
_mock_native.graph_add_node.return_value = None
_mock_native.graph_add_edge.return_value = None
_mock_native.graph_run.return_value = json.dumps({"research": "done", "write": "done"})
_mock_native.destroy_graph.return_value = None
_mock_native.create_workflow.return_value = 20
_mock_native.workflow_add_step.return_value = None
_mock_native.workflow_add_dependency.return_value = None
_mock_native.workflow_run.return_value = json.dumps({"s1": "ok", "s2": "ok"})
_mock_native.destroy_workflow.return_value = None
_mock_native.create_network.return_value = 30
_mock_native.network_add_agent.return_value = None
_mock_native.network_set_supervisor.return_value = None
_mock_native.network_delegate.return_value = json.dumps({"result": "delegated"})
_mock_native.network_agent_cards.return_value = json.dumps([{"name": "a1"}])
_mock_native.destroy_network.return_value = None
_mock_native.create_middleware_chain.return_value = 40
_mock_native.middleware_use_logging.return_value = None
_mock_native.middleware_use_caching.return_value = None
_mock_native.destroy_middleware_chain.return_value = None
_mock_native.create_plugin_registry.return_value = 50
_mock_native.plugin_registry_add_telemetry.return_value = None
_mock_native.plugin_registry_add_memory.return_value = None
_mock_native.plugin_registry_list.return_value = json.dumps(["telemetry", "memory"])
_mock_native.plugin_registry_emit.return_value = None
_mock_native.destroy_plugin_registry.return_value = None
_mock_native.create_guardrail_chain.return_value = 60
_mock_native.guardrail_chain_add_content_moderation.return_value = None
_mock_native.guardrail_chain_add_pii_detection.return_value = None
_mock_native.guardrail_chain_add_token_limit.return_value = None
_mock_native.guardrail_chain_add_regex_filter.return_value = None
_mock_native.guardrail_chain_add_schema.return_value = None
_mock_native.guardrail_chain_list.return_value = json.dumps(["content_moderation", "pii"])
_mock_native.destroy_guardrail_chain.return_value = None
_mock_native.create_eval_runner.return_value = 70
_mock_native.eval_add_scorer.return_value = None
_mock_native.load_dataset_jsonl.return_value = json.dumps([{"input": "q"}])
_mock_native.load_dataset_json.return_value = json.dumps([{"input": "q"}])
_mock_native.destroy_eval_runner.return_value = None
_mock_native.create_telemetry.return_value = 80
_mock_native.telemetry_record_span.return_value = None
_mock_native.telemetry_export_spans.return_value = json.dumps([{"name": "test", "duration": 100}])
_mock_native.telemetry_export_metrics.return_value = json.dumps({"totalSpans": 1})
_mock_native.telemetry_clear.return_value = None
_mock_native.destroy_telemetry.return_value = None
_mock_native.create_approval_manager.return_value = 90
_mock_native.approval_request.return_value = "req-123"
_mock_native.approval_approve.return_value = None
_mock_native.approval_deny.return_value = None
_mock_native.approval_list_pending.return_value = json.dumps([])
_mock_native.destroy_approval_manager.return_value = None
_mock_native.create_checkpoint_store.return_value = 91
_mock_native.checkpoint_save.return_value = None
_mock_native.checkpoint_load.return_value = json.dumps({"id": "cp1", "data": {}})
_mock_native.destroy_checkpoint_store.return_value = None
_mock_native.create_mcp_server.return_value = 92
_mock_native.mcp_server_add_tool.return_value = None
_mock_native.mcp_server_handle.return_value = json.dumps({"jsonrpc": "2.0", "result": {}})
_mock_native.destroy_mcp_server.return_value = None
_mock_native.create_fallback_provider.return_value = 100
_mock_native.create_circuit_breaker.return_value = 101
_mock_native.create_resilient_provider.return_value = 102
_mock_native.count_tokens.return_value = 42
_mock_native.count_tokens_for_model.return_value = 45
_mock_native.count_message_tokens.return_value = 100
_mock_native.get_context_window_size.return_value = 128000
_mock_native.agent_config_from_json.return_value = '{"name":"test"}'
_mock_native.agent_config_resolve_env.return_value = "resolved-value"
_mock_native.create_tool_validator.return_value = 110
_mock_native.tool_validator_validate.return_value = json.dumps({"valid": True})
_mock_native.destroy_tool_validator.return_value = None
_mock_native.py_parse_partial_json.return_value = json.dumps({"partial": True})
_mock_native.version.return_value = "2.0.0-test"

# Team
_mock_native.create_team.return_value = 200
_mock_native.team_add_agent.return_value = None
_mock_native.team_set_strategy.return_value = None
_mock_native.team_run.return_value = json.dumps({
    "finalText": "team output",
    "results": [{"text": "agent1 result", "steps": 1, "inputTokens": 10, "outputTokens": 20}],
})
_mock_native.destroy_team.return_value = None

# stream_generate returns a JSON array of event objects (adjacently tagged)
import asyncio

async def _mock_stream_generate(provider_handle, messages_json, temperature=None, max_tokens=None):
    return json.dumps([
        {"type": "text_delta", "data": "Hello"},
        {"type": "text_delta", "data": " World"},
        {"type": "done"},
    ])

_mock_native.stream_generate = _mock_stream_generate


@pytest.fixture(autouse=True)
def _mock_gauss_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch gauss._native with mock for all tests."""
    import sys
    monkeypatch.setitem(sys.modules, "gauss._native", _mock_native)
    _mock_native.reset_mock()
    # Set env for auto-detection
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")


# ===========================================================================
# Agent Tests
# ===========================================================================

class TestAgent:
    def test_create_and_run_string(self) -> None:
        from gauss.agent import Agent
        agent = Agent()
        result = agent.run("Hello!")
        assert result.text == "The answer is 42"
        assert str(result) == "The answer is 42"
        _mock_native.agent_run.assert_called_once()
        agent.destroy()

    def test_create_and_run_messages(self) -> None:
        from gauss._types import Message
        from gauss.agent import Agent
        agent = Agent()
        result = agent.run([Message("user", "Hi")])
        assert result.text == "The answer is 42"
        agent.destroy()

    def test_generate(self) -> None:
        from gauss.agent import Agent
        agent = Agent()
        text = agent.generate("Hello")
        assert text == "Hello from Gauss!"
        agent.destroy()

    def test_context_manager(self) -> None:
        from gauss.agent import Agent
        with Agent() as agent:
            result = agent.run("Test")
            assert result.text == "The answer is 42"
        _mock_native.destroy_provider.assert_called_once()

    def test_throws_after_destroy(self) -> None:
        from gauss.agent import Agent
        from gauss.errors import DisposedError
        agent = Agent()
        agent.destroy()
        with pytest.raises(DisposedError, match="destroyed"):
            agent.run("Test")

    def test_add_tool_chainable(self) -> None:
        from gauss._types import ToolDef
        from gauss.agent import Agent
        agent = Agent()
        result = agent.add_tool(ToolDef("search", "Search")).add_tool(ToolDef("calc", "Calculate"))
        assert result is agent
        assert len(agent._tools) == 2
        agent.destroy()

    def test_gauss_one_liner(self) -> None:
        from gauss.agent import gauss
        answer = gauss("What is life?")
        assert answer == "The answer is 42"

    def test_enterprise_run_one_liner(self) -> None:
        from gauss.agent import enterprise_run

        answer = enterprise_run("What is life?")
        assert answer == "The answer is 42"

    def test_from_env_helper(self) -> None:
        from gauss.agent import Agent
        agent = Agent.from_env(name="env-agent")
        assert agent is not None
        assert agent._config.name == "env-agent"
        agent.destroy()

    def test_with_model_clones_agent(self) -> None:
        from gauss.agent import Agent
        agent = Agent(model="gpt-4.1")
        clone = agent.with_model("gpt-5.2")
        assert clone is not agent
        assert clone._model == "gpt-5.2"
        assert agent._model == "gpt-4.1"
        agent.destroy()
        clone.destroy()

    def test_routing_policy_resolves_alias_target(self) -> None:
        from gauss._types import ProviderType
        from gauss.agent import Agent
        from gauss.routing_policy import RoutingCandidate, RoutingPolicy

        agent = Agent(
            provider=ProviderType.OPENAI,
            model="fast-chat",
            routing_policy=RoutingPolicy(
                aliases={
                    "fast-chat": [
                        RoutingCandidate(
                            provider=ProviderType.OPENAI,
                            model="gpt-4o-mini",
                            priority=1,
                        ),
                        RoutingCandidate(
                            provider=ProviderType.ANTHROPIC,
                            model="claude-3-5-haiku-latest",
                            priority=10,
                        ),
                    ]
                }
            ),
        )

        assert agent._model == "claude-3-5-haiku-latest"
        assert _mock_native.create_provider.call_args[0][0] == "anthropic"
        assert _mock_native.create_provider.call_args[0][1] == "claude-3-5-haiku-latest"
        agent.destroy()

    def test_with_routing_policy_clones_agent(self) -> None:
        from gauss._types import ProviderType
        from gauss.agent import Agent
        from gauss.routing_policy import RoutingCandidate, RoutingPolicy

        agent = Agent(provider=ProviderType.OPENAI, model="fast-chat")
        clone = agent.with_routing_policy(
            RoutingPolicy(
                aliases={
                    "fast-chat": [
                        RoutingCandidate(
                            provider=ProviderType.ANTHROPIC,
                            model="claude-3-5-haiku-latest",
                            priority=10,
                        )
                    ]
                }
            )
        )
        assert clone is not agent
        assert clone._model == "claude-3-5-haiku-latest"
        agent.destroy()
        clone.destroy()

    def test_with_routing_context_applies_runtime_decision(self) -> None:
        from gauss._types import ProviderType
        from gauss.agent import Agent
        from gauss.routing_policy import RoutingCandidate, RoutingPolicy

        agent = Agent(
            provider=ProviderType.OPENAI,
            model="fast-chat",
            routing_policy=RoutingPolicy(
                aliases={
                    "fast-chat": [
                        RoutingCandidate(
                            provider=ProviderType.OPENAI,
                            model="gpt-4o-mini",
                            priority=1,
                        ),
                        RoutingCandidate(
                            provider=ProviderType.ANTHROPIC,
                            model="claude-3-5-haiku-latest",
                            priority=10,
                        ),
                    ]
                },
                fallback_order=[ProviderType.OPENAI],
            ),
        )
        clone = agent.with_routing_context(available_providers=[ProviderType.OPENAI])
        assert clone is not agent
        assert clone._model == "gpt-4o-mini"
        assert _mock_native.create_provider.call_args[0][0] == "openai"
        assert _mock_native.create_provider.call_args[0][1] == "gpt-4o-mini"
        agent.destroy()
        clone.destroy()

    def test_routing_policy_fallback_helpers(self) -> None:
        from gauss._types import ProviderType
        from gauss.routing_policy import (
            RoutingCandidate,
            RoutingPolicy,
            resolve_fallback_provider,
            resolve_routing_target,
        )

        policy = RoutingPolicy(fallback_order=[ProviderType.ANTHROPIC, ProviderType.OPENAI])
        fallback = resolve_fallback_provider(policy, [ProviderType.OPENAI])
        assert fallback == ProviderType.OPENAI

        provider, model = resolve_routing_target(
            policy,
            ProviderType.GOOGLE,
            "gpt-5.2",
            available_providers=[ProviderType.OPENAI],
        )
        assert provider == ProviderType.OPENAI
        assert model == "gpt-5.2"

        alias_policy = RoutingPolicy(
            aliases={
                "fast-chat": [
                    RoutingCandidate(
                        provider=ProviderType.OPENAI,
                        model="gpt-4o-mini",
                        priority=1,
                    ),
                    RoutingCandidate(
                        provider=ProviderType.ANTHROPIC,
                        model="claude-3-5-haiku-latest",
                        priority=10,
                    ),
                ]
            }
        )
        provider, model = resolve_routing_target(
            alias_policy,
            ProviderType.OPENAI,
            "fast-chat",
            available_providers=[ProviderType.OPENAI],
        )
        assert provider == ProviderType.OPENAI
        assert model == "gpt-4o-mini"

    def test_routing_policy_cost_limit_helper(self) -> None:
        from gauss.routing_policy import (
            resolve_routing_target,
            RoutingPolicy,
            RoutingPolicyError,
            enforce_routing_cost_limit,
            enforce_routing_governance,
            enforce_routing_rate_limit,
            enforce_routing_time_window,
        )
        from gauss.routing_policy import GovernancePolicyPack, GovernanceRule
        from gauss._types import ProviderType

        policy = RoutingPolicy(max_total_cost_usd=1.0)
        enforce_routing_cost_limit(policy, 0.5)
        with pytest.raises(RoutingPolicyError, match="routing policy rejected cost 1.5"):
            enforce_routing_cost_limit(policy, 1.5)

        policy.max_requests_per_minute = 10
        enforce_routing_rate_limit(policy, 10)
        with pytest.raises(RoutingPolicyError, match="routing policy rejected rate 11"):
            enforce_routing_rate_limit(policy, 11)

        policy.allowed_hours_utc = [9, 10, 11]
        enforce_routing_time_window(policy, 10)
        with pytest.raises(RoutingPolicyError, match="routing policy rejected hour 20"):
            enforce_routing_time_window(policy, 20)

        governance = RoutingPolicy(
            governance=GovernancePolicyPack(
                rules=[
                    GovernanceRule(type="allow_provider", provider=ProviderType.OPENAI),
                    GovernanceRule(type="require_tag", tag="pci"),
                ]
            )
        )
        enforce_routing_governance(governance, ProviderType.OPENAI, ["pci"])
        with pytest.raises(
            RoutingPolicyError,
            match="routing policy governance rejected provider anthropic",
        ):
            enforce_routing_governance(governance, ProviderType.ANTHROPIC, ["pci"])
        with pytest.raises(
            RoutingPolicyError,
            match="routing policy governance missing tag pci",
        ):
            enforce_routing_governance(governance, ProviderType.OPENAI, [])
        with pytest.raises(
            RoutingPolicyError,
            match="routing policy governance missing tag pci",
        ):
            resolve_routing_target(
                governance,
                ProviderType.OPENAI,
                "gpt-5.2",
                governance_tags=[],
            )

    def test_governance_policy_pack_helpers(self) -> None:
        from gauss._types import ProviderType
        from gauss.routing_policy import (
            RoutingPolicyError,
            apply_governance_pack,
            governance_policy_pack,
            resolve_routing_target,
        )

        pack = governance_policy_pack("enterprise_strict")
        assert any(rule.type == "require_tag" and rule.tag == "pci" for rule in pack.rules)

        merged = apply_governance_pack(None, "cost_guarded")
        provider, model = resolve_routing_target(
            merged,
            ProviderType.OPENAI,
            "gpt-5.2",
            governance_tags=["cost-sensitive"],
        )
        assert provider == ProviderType.OPENAI
        assert model == "gpt-5.2"

        with pytest.raises(RoutingPolicyError, match="unknown governance policy pack"):
            governance_policy_pack("unknown-pack")

    def test_routing_policy_weighted_mix_and_business_hours_pack(self) -> None:
        from gauss._types import ProviderType
        from gauss.routing_policy import (
            RoutingCandidate,
            RoutingPolicy,
            apply_governance_pack,
            resolve_routing_target,
        )

        policy = RoutingPolicy(
            aliases={
                "fast-chat": [
                    RoutingCandidate(
                        provider=ProviderType.OPENAI,
                        model="gpt-4o-mini",
                        priority=1,
                    ),
                    RoutingCandidate(
                        provider=ProviderType.ANTHROPIC,
                        model="claude-3-5-haiku-latest",
                        priority=10,
                    ),
                ]
            },
            provider_weights={
                ProviderType.OPENAI: 100,
                ProviderType.ANTHROPIC: 10,
            },
        )
        provider, model = resolve_routing_target(
            policy,
            ProviderType.OPENAI,
            "fast-chat",
            available_providers=[ProviderType.OPENAI, ProviderType.ANTHROPIC],
            current_hour_utc=12,
        )
        assert provider == ProviderType.OPENAI
        assert model == "gpt-4o-mini"

        business_hours = apply_governance_pack(None, "ops_business_hours")
        assert 8 in business_hours.allowed_hours_utc
        assert 18 in business_hours.allowed_hours_utc

        balanced = apply_governance_pack(None, "balanced_mix")
        assert balanced.provider_weights.get(ProviderType.OPENAI) == 60
        assert balanced.provider_weights.get(ProviderType.ANTHROPIC) == 40

    def test_explain_routing_target(self) -> None:
        from gauss._types import ProviderType
        from gauss.routing_policy import RoutingPolicy, explain_routing_target

        explained_ok = explain_routing_target(
            RoutingPolicy(fallback_order=[ProviderType.OPENAI]),
            ProviderType.OPENAI,
            "gpt-5.2",
            current_hour_utc=12,
        )
        assert explained_ok["ok"] is True
        assert explained_ok["decision"]["provider"] == "openai"
        assert explained_ok["decision"]["selected_by"] == "direct"
        assert any(check["check"] == "selection" and check["status"] == "passed" for check in explained_ok["checks"])

        explained_fail = explain_routing_target(
            RoutingPolicy(allowed_hours_utc=[9, 10, 11]),
            ProviderType.OPENAI,
            "gpt-5.2",
            current_hour_utc=22,
        )
        assert explained_fail["ok"] is False
        assert "routing policy rejected hour 22" in explained_fail["error"]

    def test_evaluate_policy_gate(self) -> None:
        from gauss._types import ProviderType
        from gauss.routing_policy import RoutingPolicy, evaluate_policy_gate

        summary = evaluate_policy_gate(
            RoutingPolicy(allowed_hours_utc=[9, 10, 11]),
            [
                {"provider": ProviderType.OPENAI, "model": "gpt-5.2", "options": {"current_hour_utc": 10}},
                {"provider": ProviderType.OPENAI, "model": "gpt-5.2", "options": {"current_hour_utc": 22}},
            ],
        )
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["failed_indexes"] == [1]

    def test_stream_text_aggregates_deltas(self) -> None:
        from gauss._types import AgentResult
        from gauss.agent import Agent

        agent = Agent()

        def fake_stream(_prompt, callback):
            callback('{"type":"text_delta","text":"Hello"}')
            callback('{"type":"text_delta","delta":" World"}')
            return AgentResult(text="Hello World", messages=[], tool_calls=[], usage={})

        chunks: list[str] = []
        with patch.object(agent, "stream", side_effect=fake_stream):
            text = agent.stream_text("Hi", on_delta=chunks.append)
        assert text == "Hello World"
        assert chunks == ["Hello", " World"]
        agent.destroy()

    def test_top_level_parity_exports(self) -> None:
        from gauss import (
            McpClient,
            McpClientConfig,
            McpToolResult,
            ModelPricing,
            ProviderType,
            TypedToolDef,
            clear_pricing,
            create_resilient_agent,
            create_tool_executor,
            detect_provider,
            enterprise_run,
            evaluate_policy_gate,
            get_pricing,
            resolve_api_key,
            set_pricing,
            tool,
            version,
        )

        assert callable(enterprise_run)
        assert callable(detect_provider)
        assert callable(resolve_api_key)
        assert callable(evaluate_policy_gate)
        assert callable(create_resilient_agent)
        assert callable(tool)
        assert callable(create_tool_executor)
        assert callable(set_pricing)
        assert callable(get_pricing)
        assert callable(clear_pricing)
        assert version() == "2.0.0-test"
        assert detect_provider() == ProviderType.OPENAI
        assert resolve_api_key(ProviderType.OPENAI) == "sk-test-key"
        assert TypedToolDef is not None
        assert McpClient is not None
        assert McpClientConfig is not None
        assert McpToolResult is not None
        assert ModelPricing is not None


# ===========================================================================
# Memory Tests
# ===========================================================================

class TestMemory:
    def test_store_and_recall(self) -> None:
        from gauss.memory import Memory
        mem = Memory()
        mem.store("conversation", "Hello!", session_id="s1")
        entries = mem.recall(session_id="s1")
        assert len(entries) == 1
        assert entries[0]["content"] == "Hi"
        # Verify recall passes JSON options
        _mock_native.memory_recall.assert_called_once_with(
            2, '{"sessionId": "s1"}'
        )
        mem.destroy()

    def test_store_dict(self) -> None:
        from gauss.memory import Memory
        mem = Memory()
        mem.store({"id": "m1", "content": "World", "entry_type": "conversation", "timestamp": "2024-01-01T00:00:00Z"})
        _mock_native.memory_store.assert_called_once()
        mem.destroy()

    def test_context_manager(self) -> None:
        from gauss.memory import Memory
        with Memory() as mem:
            mem.store("conversation", "Test")
        _mock_native.destroy_memory.assert_called()


# ===========================================================================
# VectorStore Tests
# ===========================================================================

class TestVectorStore:
    def test_upsert_and_search(self) -> None:
        from gauss.vector_store import Chunk, VectorStore
        store = VectorStore()
        store.upsert([Chunk(id="c1", document_id="d1", content="hello", index=0, embedding=[0.1, 0.2])])
        results = store.search([0.1, 0.2], top_k=5)
        assert len(results) == 1
        assert results[0].score == 0.95
        store.destroy()

    def test_cosine_similarity(self) -> None:
        from gauss.vector_store import VectorStore
        assert VectorStore.cosine_similarity([1, 0], [1, 0]) == 0.98


# ===========================================================================
# Graph Tests
# ===========================================================================

class TestGraph:
    def test_create_and_add_node(self) -> None:
        from gauss.agent import Agent
        from gauss.graph import Graph
        a = Agent(name="researcher")
        g = Graph()
        g.add_node("research", a)
        _mock_native.create_graph.assert_called_once()
        _mock_native.graph_add_node.assert_called_once_with(
            10, "research", "researcher", 1, None, None
        )
        a.destroy()
        g.destroy()

    def test_add_edge(self) -> None:
        from gauss.graph import Graph
        g = Graph()
        g.add_edge("a", "b")
        _mock_native.graph_add_edge.assert_called_once_with(10, "a", "b")
        g.destroy()

    def test_chaining(self) -> None:
        from gauss.agent import Agent
        from gauss.graph import Graph
        a = Agent(name="r")
        b = Agent(name="w")
        g = Graph().add_node("r", a).add_node("w", b).add_edge("r", "w")
        assert isinstance(g, Graph)
        a.destroy()
        b.destroy()
        g.destroy()

    def test_context_manager(self) -> None:
        from gauss.graph import Graph
        with Graph() as g:
            pass
        _mock_native.destroy_graph.assert_called_once()

    def test_throws_after_destroy(self) -> None:
        from gauss.errors import DisposedError
        from gauss.graph import Graph
        g = Graph()
        g.destroy()
        with pytest.raises(DisposedError, match="destroyed"):
            g.add_edge("a", "b")


# ===========================================================================
# Workflow Tests
# ===========================================================================

class TestWorkflow:
    def test_create_and_add_step(self) -> None:
        from gauss.agent import Agent
        from gauss.workflow import Workflow
        a = Agent(name="planner")
        wf = Workflow()
        wf.add_step("plan", a)
        _mock_native.create_workflow.assert_called_once()
        _mock_native.workflow_add_step.assert_called_once_with(
            20, "plan", "planner", 1, None, None
        )
        a.destroy()
        wf.destroy()

    def test_add_dependency(self) -> None:
        from gauss.workflow import Workflow
        wf = Workflow()
        wf.add_dependency("execute", "plan")
        _mock_native.workflow_add_dependency.assert_called_once_with(
            20, "execute", "plan"
        )
        wf.destroy()

    def test_chaining(self) -> None:
        from gauss.agent import Agent
        from gauss.workflow import Workflow
        a = Agent(name="p")
        b = Agent(name="e")
        wf = (
            Workflow()
            .add_step("p", a)
            .add_step("e", b)
            .add_dependency("e", "p")
        )
        assert isinstance(wf, Workflow)
        a.destroy()
        b.destroy()
        wf.destroy()

    def test_context_manager(self) -> None:
        from gauss.workflow import Workflow
        with Workflow() as wf:
            pass
        _mock_native.destroy_workflow.assert_called_once()

    def test_throws_after_destroy(self) -> None:
        from gauss.workflow import Workflow
        wf = Workflow()
        wf.destroy()
        with pytest.raises(RuntimeError, match="destroyed"):
            wf.add_dependency("a", "b")


# ===========================================================================
# Network Tests
# ===========================================================================

class TestNetwork:
    def test_add_agent_correct_arg_order(self) -> None:
        from gauss.agent import Agent
        from gauss.network import Network
        a1 = Agent(name="analyst")
        net = Network().add_agent(a1).set_supervisor("analyst")
        # Verify correct argument order: (handle, name, provider_handle, card, conns)
        _mock_native.network_add_agent.assert_called_once_with(
            30, "analyst", 1, None, None
        )
        a1.destroy()
        net.destroy()

    def test_delegate(self) -> None:
        from gauss.agent import Agent
        from gauss.network import Network
        a1 = Agent(name="analyst")
        net = Network().add_agent(a1).set_supervisor("analyst")
        result = net.delegate("analyst", "Write code")
        assert result == {"result": "delegated"}
        a1.destroy()
        net.destroy()

    def test_agent_cards(self) -> None:
        from gauss.agent import Agent
        from gauss.network import Network
        a1 = Agent(name="analyst")
        net = Network().add_agent(a1).set_supervisor("analyst")
        cards = net.agent_cards()
        assert cards == [{"name": "a1"}]
        a1.destroy()
        net.destroy()

    def test_quick(self) -> None:
        from gauss.network import Network

        net = Network.quick("supervisor", [
            {"name": "supervisor", "instructions": "Delegate work"},
            {"name": "coder", "instructions": "Write code"},
        ])
        result = net.delegate("coder", "Implement feature")
        assert result == {"result": "delegated"}
        _mock_native.network_set_supervisor.assert_called_with(30, "supervisor")
        net.destroy()

    def test_templates(self) -> None:
        from gauss.errors import ValidationError
        from gauss.network import Network

        template = Network.template("research-delivery")
        assert template["supervisor"] == "lead"
        assert len(template["agents"]) >= 3
        assert Network.template("support-triage")["supervisor"] == "support-lead"
        assert Network.template("fintech-risk-review")["supervisor"] == "risk-lead"
        assert Network.template("rag-ops")["supervisor"] == "rag-ops-lead"

        net = Network.from_template("incident-response")
        result = net.delegate("incident-commander", "Stabilize production incident")
        assert result == {"result": "delegated"}
        net.destroy()

        with pytest.raises(ValidationError, match='Unknown network template'):
            Network.template("unknown")


# ===========================================================================
# Middleware Tests
# ===========================================================================

class TestMiddleware:
    def test_create_and_use_logging(self) -> None:
        from gauss.middleware import MiddlewareChain
        chain = MiddlewareChain()
        chain.use_logging()
        _mock_native.create_middleware_chain.assert_called_once()
        _mock_native.middleware_use_logging.assert_called_once_with(40)
        chain.destroy()

    def test_use_caching(self) -> None:
        from gauss.middleware import MiddlewareChain
        chain = MiddlewareChain()
        chain.use_caching(ttl_ms=30_000)
        _mock_native.middleware_use_caching.assert_called_once_with(40, 30_000)
        chain.destroy()

    def test_chaining(self) -> None:
        from gauss.middleware import MiddlewareChain
        chain = MiddlewareChain().use_logging().use_caching(ttl_ms=60_000)
        assert isinstance(chain, MiddlewareChain)
        chain.destroy()

    def test_context_manager(self) -> None:
        from gauss.middleware import MiddlewareChain
        with MiddlewareChain() as chain:
            chain.use_logging()
        _mock_native.destroy_middleware_chain.assert_called_once()

    def test_throws_after_destroy(self) -> None:
        from gauss.middleware import MiddlewareChain
        chain = MiddlewareChain()
        chain.destroy()
        with pytest.raises(RuntimeError, match="destroyed"):
            chain.use_logging()


# ===========================================================================
# Plugin Tests
# ===========================================================================

class TestPlugin:
    def test_registry(self) -> None:
        from gauss.plugin import PluginRegistry
        reg = PluginRegistry().add_telemetry().add_memory()
        assert reg.list() == ["telemetry", "memory"]
        reg.emit({"type": "agent.start"})
        reg.destroy()


# ===========================================================================
# Guardrail Tests
# ===========================================================================

class TestGuardrail:
    def test_chain(self) -> None:
        from gauss.guardrail import GuardrailChain
        chain = (
            GuardrailChain()
            .add_content_moderation(blocked_categories=["violence"])
            .add_pii_detection(action="redact")
            .add_token_limit(max_input=4000)
            .add_regex_filter(patterns=[r"\bpassword\b"])
            .add_schema({"type": "object"})
        )
        assert chain.list() == ["content_moderation", "pii"]
        chain.destroy()


# ===========================================================================
# Eval Tests
# ===========================================================================

class TestEval:
    def test_runner(self) -> None:
        from gauss.eval import EvalRunner
        runner = EvalRunner(0.8).add_scorer("exact_match")
        assert runner.handle == 70
        runner.destroy()

    def test_load_datasets(self) -> None:
        from gauss.eval import EvalRunner
        assert EvalRunner.load_dataset_jsonl("data") == [{"input": "q"}]
        assert EvalRunner.load_dataset_json("data") == [{"input": "q"}]


# ===========================================================================
# Telemetry Tests
# ===========================================================================

class TestTelemetry:
    def test_record_and_export(self) -> None:
        from gauss.telemetry import Telemetry
        tel = Telemetry()
        tel.record_span({"name": "test", "duration": 50})
        spans = tel.export_spans()
        assert spans == [{"name": "test", "duration": 100}]
        metrics = tel.export_metrics()
        assert metrics == {"totalSpans": 1}
        tel.clear()
        tel.destroy()


# ===========================================================================
# HITL Tests
# ===========================================================================

class TestApproval:
    def test_approval_flow(self) -> None:
        from gauss.approval import ApprovalManager
        mgr = ApprovalManager()
        req_id = mgr.request("tool", {"action": "delete"})
        assert req_id == "req-123"
        mgr.approve(req_id)
        mgr.deny("other", reason="Not safe")
        assert mgr.list_pending() == []
        mgr.destroy()


class TestCheckpoint:
    def test_save_and_load(self) -> None:
        from gauss.checkpoint import CheckpointStore
        store = CheckpointStore()
        store.save({"id": "cp1", "data": {"step": 3}})
        cp = store.load("cp1")
        assert cp == {"id": "cp1", "data": {}}
        store.destroy()


# ===========================================================================
# MCP Tests
# ===========================================================================

class TestMcp:
    @pytest.mark.asyncio
    async def test_server(self) -> None:
        from gauss.mcp import McpServer
        srv = McpServer("sdk-test-mcp", "1.0.0")
        srv.add_tool({"name": "greet", "description": "Say hello", "inputSchema": {"type": "object"}})
        resp = await srv.handle_message({"jsonrpc": "2.0", "id": 99, "method": "tools/list"})
        assert "result" in resp
        # The tools may show up under different key structures depending on server state
        result = resp.get("result", {})
        assert isinstance(result, dict)
        srv.destroy()


# ===========================================================================
# Resilience Tests
# ===========================================================================

class TestResilience:
    def test_fallback(self) -> None:
        from gauss.resilience import create_fallback_provider
        assert create_fallback_provider([1, 2]) == 100

    def test_circuit_breaker(self) -> None:
        from gauss.resilience import create_circuit_breaker
        assert create_circuit_breaker(1, 5, recovery_timeout_ms=30000) == 101

    def test_resilient(self) -> None:
        from gauss.resilience import create_resilient_provider
        assert create_resilient_provider(1, [2, 3], True) == 102

    def test_resilient_agent(self) -> None:
        from gauss.agent import Agent
        from gauss.resilience import create_resilient_agent

        primary = Agent(name="primary")
        fallback = Agent(name="fallback")
        try:
            handle = create_resilient_agent(primary, [fallback], True)
            assert handle == 102
            _mock_native.create_resilient_provider.assert_called_with(1, "[1]", True)
        finally:
            primary.destroy()
            fallback.destroy()


# ===========================================================================
# Tokens Tests
# ===========================================================================

class TestTokens:
    def test_count(self) -> None:
        from gauss.tokens import count_tokens, count_tokens_for_model, count_message_tokens, get_context_window_size
        assert count_tokens("hello") == 42
        assert count_tokens_for_model("hello", "gpt-4") == 45
        assert count_message_tokens([{"role": "user", "content": "hi"}]) == 100
        assert get_context_window_size("gpt-4") == 128000


# ===========================================================================
# Config Tests
# ===========================================================================

class TestConfig:
    def test_parse(self) -> None:
        from gauss.config import parse_agent_config, resolve_env
        assert parse_agent_config('{"name":"test"}') == '{"name":"test"}'
        assert resolve_env("${VAR}") == "resolved-value"


# ===========================================================================
# ToolValidator Tests
# ===========================================================================

class TestToolValidator:
    def test_validate(self) -> None:
        from gauss.tool_validator import ToolValidator
        v = ToolValidator(["type_cast"])
        result = v.validate({"name": "test"}, {"type": "object"})
        assert result == {"valid": True}
        v.destroy()


# ===========================================================================
# Stream Tests
# ===========================================================================

class TestStream:
    def test_parse_partial(self) -> None:
        from gauss.stream import parse_partial_json
        result = parse_partial_json('{"partial": tru')
        assert result == {"partial": True}


# ===========================================================================
# Types Tests
# ===========================================================================

class TestTypes:
    def test_detect_provider(self) -> None:
        from gauss._types import ProviderType, detect_provider
        assert detect_provider() == ProviderType.OPENAI

    def test_detect_provider_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gauss._types import detect_provider
        for var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY", "OLLAMA_HOST"]:
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(EnvironmentError, match="No API key found"):
            detect_provider()

    def test_message_to_dict(self) -> None:
        from gauss._types import Message
        msg = Message("user", "Hello")
        assert msg.to_dict() == {"role": "user", "content": "Hello"}

    def test_agent_result_str(self) -> None:
        from gauss._types import AgentResult
        result = AgentResult(text="Hello", messages=[], tool_calls=[], usage={})
        assert str(result) == "Hello"


class TestBatch:
    def test_batch_item_repr(self) -> None:
        from gauss.batch import BatchItem
        item = BatchItem("hello")
        assert "hello" in repr(item)
        assert "error" in repr(item)

    def test_batch_item_ok(self) -> None:
        from gauss._types import AgentResult
        from gauss.batch import BatchItem
        item = BatchItem("hello")
        item.result = AgentResult(text="world", messages=[], tool_calls=[], usage={})
        assert "ok" in repr(item)

    def test_batch_item_error(self) -> None:
        from gauss.batch import BatchItem
        item = BatchItem("hello")
        item.error = RuntimeError("boom")
        assert "error" in repr(item)
        assert item.result is None

    def test_batch_exports(self) -> None:
        from gauss import BatchItem, batch
        assert callable(batch)
        assert BatchItem is not None


class TestAgentStream:
    @pytest.mark.asyncio
    async def test_stream_iter_yields_events(self) -> None:
        from gauss import Agent
        agent = Agent()
        stream = agent.stream_iter("Hello")
        events = []
        async for event in stream:
            events.append(event)
        assert len(events) == 3
        assert events[0].type == "text_delta"
        assert events[0].text == "Hello"
        assert events[1].type == "text_delta"
        assert events[1].text == " World"
        assert events[2].type == "done"
        assert stream.text == "Hello World"
        agent.destroy()

    @pytest.mark.asyncio
    async def test_stream_events_property(self) -> None:
        from gauss import Agent
        agent = Agent()
        stream = agent.stream_iter("Hello")
        async for _ in stream:
            pass
        assert len(stream.events) == 3
        agent.destroy()

    def test_stream_event_from_json_invalid(self) -> None:
        from gauss.stream import StreamEvent
        event = StreamEvent.from_json("not json{{{")
        assert event.type == "raw"
        assert event.text == "not json{{{"

    def test_stream_event_from_json_valid(self) -> None:
        from gauss.stream import StreamEvent
        event = StreamEvent.from_json('{"type":"text_delta","data":"hi"}')
        assert event.type == "text_delta"
        assert event.text == "hi"

    def test_stream_exports(self) -> None:
        from gauss import AgentStream, StreamEvent
        assert AgentStream is not None
        assert StreamEvent is not None

    @pytest.mark.asyncio
    async def test_stream_raises_after_destroy(self) -> None:
        from gauss import Agent
        from gauss.errors import DisposedError
        agent = Agent()
        agent.destroy()
        with pytest.raises(DisposedError, match="destroyed"):
            agent.stream_iter("Hello")


class TestRetry:
    def test_succeeds_first_try(self) -> None:
        from gauss.retry import with_retry
        calls = 0
        def fn():
            nonlocal calls
            calls += 1
            return "ok"
        assert with_retry(fn) == "ok"
        assert calls == 1

    def test_retries_on_failure(self) -> None:
        from gauss.retry import with_retry, RetryConfig
        calls = 0
        def fn():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("fail")
            return "ok"
        result = with_retry(fn, RetryConfig(max_retries=3, base_delay_s=0.001))
        assert result == "ok"
        assert calls == 3

    def test_throws_after_max_retries(self) -> None:
        from gauss.retry import with_retry, RetryConfig
        def fn():
            raise RuntimeError("always fail")
        with pytest.raises(RuntimeError, match="always fail"):
            with_retry(fn, RetryConfig(max_retries=2, base_delay_s=0.001))

    def test_retry_if_predicate(self) -> None:
        from gauss.retry import with_retry, RetryConfig
        calls = 0
        def fn():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("retryable")
            raise TypeError("not retryable")
        with pytest.raises(TypeError, match="not retryable"):
            with_retry(fn, RetryConfig(
                max_retries=5,
                base_delay_s=0.001,
                retry_if=lambda e, _: isinstance(e, ValueError),
            ))
        assert calls == 2

    def test_on_retry_callback(self) -> None:
        from gauss.retry import with_retry, RetryConfig
        retries = []
        def fn():
            if len(retries) < 1:
                raise RuntimeError("fail")
            return "ok"
        with_retry(fn, RetryConfig(
            max_retries=2,
            base_delay_s=0.001,
            on_retry=lambda e, a, d: retries.append((a, d)),
        ))
        assert len(retries) == 1
        assert retries[0][0] == 1

    def test_retryable_wraps_agent(self) -> None:
        from gauss import Agent
        from gauss.retry import retryable, RetryConfig
        agent = Agent()
        run = retryable(agent, RetryConfig(max_retries=1, base_delay_s=0.001))
        result = run("Hello")
        assert result.text == "The answer is 42"
        agent.destroy()

    def test_exports(self) -> None:
        from gauss import RetryConfig, retryable, with_retry
        assert callable(with_retry)
        assert callable(retryable)
        assert RetryConfig is not None


class TestStructured:
    def test_extracts_json(self) -> None:
        from gauss import Agent
        from gauss.structured import structured
        _mock_native.agent_run.return_value = json.dumps({
            "text": '{"fruits":["apple","banana"]}',
            "messages": [], "toolCalls": [], "usage": {},
        })
        agent = Agent()
        result = structured(agent, "List fruits", schema={
            "type": "object",
            "properties": {"fruits": {"type": "array"}},
        })
        assert result.data == {"fruits": ["apple", "banana"]}
        agent.destroy()

    def test_handles_code_blocks(self) -> None:
        from gauss import Agent
        from gauss.structured import structured
        _mock_native.agent_run.return_value = json.dumps({
            "text": 'Here:\n```json\n{"name":"Alice"}\n```',
            "messages": [], "toolCalls": [], "usage": {},
        })
        agent = Agent()
        result = structured(agent, "Who?", schema={"type": "object"})
        assert result.data == {"name": "Alice"}
        agent.destroy()

    def test_include_raw(self) -> None:
        from gauss import Agent
        from gauss.structured import structured
        _mock_native.agent_run.return_value = json.dumps({
            "text": '{"ok":true}',
            "messages": [], "toolCalls": [], "usage": {},
        })
        agent = Agent()
        result = structured(agent, "test", schema={"type": "object"}, include_raw=True)
        assert result.raw is not None
        agent.destroy()

    def test_retries_on_parse_failure(self) -> None:
        from gauss import Agent
        from gauss.structured import structured
        call_count = 0
        orig_return = _mock_native.agent_run.return_value
        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return json.dumps({"text": "not json!!!", "messages": [], "toolCalls": [], "usage": {}})
            return json.dumps({"text": '{"valid":true}', "messages": [], "toolCalls": [], "usage": {}})
        _mock_native.agent_run.side_effect = _side_effect
        agent = Agent()
        result = structured(agent, "test", schema={"type": "object"})
        assert result.data == {"valid": True}
        agent.destroy()
        _mock_native.agent_run.side_effect = None
        _mock_native.agent_run.return_value = orig_return

    def test_throws_after_max_retries(self) -> None:
        from gauss import Agent
        from gauss.structured import structured
        _mock_native.agent_run.return_value = json.dumps({
            "text": "never valid json!!!",
            "messages": [], "toolCalls": [], "usage": {},
        })
        agent = Agent()
        with pytest.raises(RuntimeError, match="Failed to extract"):
            structured(agent, "test", schema={"type": "object"}, max_parse_retries=1)
        agent.destroy()

    def test_requires_schema_or_config(self) -> None:
        from gauss import Agent
        from gauss.structured import structured
        agent = Agent()
        with pytest.raises(ValueError, match="schema or config"):
            structured(agent, "test")
        agent.destroy()

    def test_exports(self) -> None:
        from gauss import StructuredConfig, StructuredResult, structured
        assert callable(structured)
        assert StructuredConfig is not None
        assert StructuredResult is not None


class TestTemplate:
    def test_create_template(self) -> None:
        from gauss.template import template
        t = template("Hello {{name}}, age {{age}}")
        assert t.variables == ["name", "age"]
        assert t.raw == "Hello {{name}}, age {{age}}"

    def test_render(self) -> None:
        from gauss.template import template
        t = template("Hello {{name}}!")
        assert t(name="World") == "Hello World!"

    def test_missing_variable(self) -> None:
        from gauss.template import template
        t = template("Hello {{name}}!")
        with pytest.raises(KeyError, match="name"):
            t()

    def test_multiple_occurrences(self) -> None:
        from gauss.template import template
        t = template("{{x}} + {{x}} = 2*{{x}}")
        assert t(x="5") == "5 + 5 = 2*5"
        assert t.variables == ["x"]

    def test_composition(self) -> None:
        from gauss.template import template
        inner = template("Hello {{name}}")
        outer = template("{{greeting}}, welcome!")
        result = outer(greeting=inner(name="Alice"))
        assert result == "Hello Alice, welcome!"

    def test_repr(self) -> None:
        from gauss.template import template
        t = template("{{a}} {{b}}")
        assert "a" in repr(t)
        assert "b" in repr(t)

    def test_builtin_summarize(self) -> None:
        from gauss import summarize
        assert summarize.variables == ["format", "style", "text"]

    def test_builtin_translate(self) -> None:
        from gauss import translate
        result = translate(language="French", text="Hello")
        assert "French" in result
        assert "Hello" in result

    def test_builtin_code_review(self) -> None:
        from gauss import code_review
        result = code_review(language="python", code="x = 1")
        assert "python" in result
        assert "x = 1" in result

    def test_builtin_classify(self) -> None:
        from gauss import classify
        result = classify(categories="spam, ham", text="Buy now!")
        assert "spam" in result

    def test_builtin_extract(self) -> None:
        from gauss import extract
        result = extract(fields="name, email", text="I'm John")
        assert "name" in result

    def test_exports(self) -> None:
        from gauss import PromptTemplate, template
        assert callable(template)
        assert PromptTemplate is not None


class TestPipeline:
    @pytest.mark.asyncio
    async def test_pipe_no_steps(self) -> None:
        from gauss.pipeline import pipe
        assert await pipe("hello") == "hello"

    @pytest.mark.asyncio
    async def test_pipe_sync_steps(self) -> None:
        from gauss.pipeline import pipe
        result = await pipe(5, lambda n: n * 2, lambda n: n + 1)
        assert result == 11

    @pytest.mark.asyncio
    async def test_pipe_async_steps(self) -> None:
        import asyncio
        from gauss.pipeline import pipe
        async def double(n: int) -> int:
            return n * 2
        async def add_one(n: int) -> int:
            return n + 1
        result = await pipe(5, double, add_one)
        assert result == 11

    def test_map_sync(self) -> None:
        from gauss.pipeline import map_sync
        results = map_sync([1, 2, 3], lambda x: x * 2)
        assert results == [2, 4, 6]

    def test_map_sync_with_concurrency(self) -> None:
        from gauss.pipeline import map_sync
        results = map_sync([1, 2, 3], lambda x: x * 2, concurrency=2)
        assert results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_map_async(self) -> None:
        from gauss.pipeline import map_async
        async def double(x: int) -> int:
            return x * 2
        results = await map_async([1, 2, 3], double)
        assert results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_map_async_concurrency(self) -> None:
        from gauss.pipeline import map_async
        async def double(x: int) -> int:
            return x * 2
        results = await map_async([1, 2, 3], double, concurrency=2)
        assert results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_filter_async(self) -> None:
        from gauss.pipeline import filter_async
        async def is_even(x: int) -> bool:
            return x % 2 == 0
        results = await filter_async([1, 2, 3, 4, 5], is_even)
        assert results == [2, 4]

    def test_filter_sync(self) -> None:
        from gauss.pipeline import filter_sync
        results = filter_sync([1, 2, 3, 4, 5], lambda x: x % 2 == 0)
        assert results == [2, 4]

    @pytest.mark.asyncio
    async def test_reduce_async(self) -> None:
        from gauss.pipeline import reduce_async
        async def add(acc: int, x: int) -> int:
            return acc + x
        result = await reduce_async([1, 2, 3], add, 0)
        assert result == 6

    def test_reduce_sync(self) -> None:
        from gauss.pipeline import reduce_sync
        result = reduce_sync([1, 2, 3], lambda acc, x: acc + x, 0)
        assert result == 6

    @pytest.mark.asyncio
    async def test_tap_async(self) -> None:
        from gauss.pipeline import tap_async

        seen: list[tuple[int, int]] = []

        async def capture(value: int, index: int) -> None:
            seen.append((value, index))

        items = [4, 5, 6]
        result = await tap_async(items, capture)
        assert result == items
        assert seen == [(4, 0), (5, 1), (6, 2)]

    def test_compose(self) -> None:
        from gauss.pipeline import compose
        fn = compose(lambda s: s.upper(), lambda s: f"[{s}]")
        assert fn("hello") == "[HELLO]"

    def test_exports(self) -> None:
        from gauss import (
            compose, compose_async, filter_async, filter_sync,
            map_async, map_sync, pipe, reduce_async, reduce_sync, tap_async,
        )
        assert callable(pipe)
        assert callable(map_async)
        assert callable(map_sync)
        assert callable(filter_async)
        assert callable(filter_sync)
        assert callable(reduce_async)
        assert callable(reduce_sync)
        assert callable(tap_async)
        assert callable(compose)


# ===========================================================================
# Team Tests
# ===========================================================================

class TestTeam:
    def test_create_team(self) -> None:
        from gauss.team import Team
        t = Team("my-team")
        _mock_native.create_team.assert_called_once_with("my-team")
        assert t.handle == 200
        t.destroy()

    def test_add_agent(self) -> None:
        from gauss.agent import Agent
        from gauss.team import Team
        a = Agent(name="helper")
        t = Team("t")
        t.add(a)
        _mock_native.team_add_agent.assert_called_once_with(200, "helper", 1, None)
        a.destroy()
        t.destroy()

    def test_add_agent_with_instructions(self) -> None:
        from gauss.agent import Agent
        from gauss.team import Team
        a = Agent(name="helper")
        t = Team("t")
        t.add(a, instructions="custom instructions")
        _mock_native.team_add_agent.assert_called_once_with(200, "helper", 1, "custom instructions")
        a.destroy()
        t.destroy()

    def test_set_strategy(self) -> None:
        from gauss.team import Team
        t = Team("t").strategy("parallel")
        _mock_native.team_set_strategy.assert_called_once_with(200, "parallel")
        t.destroy()

    def test_chaining(self) -> None:
        from gauss.agent import Agent
        from gauss.team import Team
        a = Agent(name="r")
        b = Agent(name="w")
        t = Team("t").add(a).add(b).strategy("sequential")
        assert isinstance(t, Team)
        a.destroy()
        b.destroy()
        t.destroy()

    def test_run(self) -> None:
        from gauss.team import Team
        t = Team("t")
        result = t.run("Do the thing")
        assert result["finalText"] == "team output"
        assert len(result["results"]) == 1
        assert result["results"][0]["text"] == "agent1 result"
        t.destroy()

    def test_context_manager(self) -> None:
        from gauss.team import Team
        with Team("t") as t:
            pass
        _mock_native.destroy_team.assert_called_once()

    def test_throws_after_destroy(self) -> None:
        from gauss.errors import DisposedError
        from gauss.team import Team
        t = Team("t")
        t.destroy()
        with pytest.raises(DisposedError, match="destroyed"):
            t.add(MagicMock())


# ─── Grounding & Image Generation Types ──────────────────────────────

class TestGroundingTypes:
    """Tests for grounding metadata types."""

    def test_grounding_chunk_defaults(self) -> None:
        from gauss._types import GroundingChunk
        gc = GroundingChunk()
        assert gc.url is None
        assert gc.title is None

    def test_grounding_chunk_with_values(self) -> None:
        from gauss._types import GroundingChunk
        gc = GroundingChunk(url="https://example.com", title="Example")
        assert gc.url == "https://example.com"
        assert gc.title == "Example"

    def test_grounding_metadata_defaults(self) -> None:
        from gauss._types import GroundingMetadata
        gm = GroundingMetadata()
        assert gm.search_queries == []
        assert gm.grounding_chunks == []
        assert gm.search_entry_point is None

    def test_grounding_metadata_with_values(self) -> None:
        from gauss._types import GroundingChunk, GroundingMetadata
        gm = GroundingMetadata(
            search_queries=["test query"],
            grounding_chunks=[GroundingChunk(url="https://example.com", title="Example")],
            search_entry_point="<div>rendered</div>",
        )
        assert len(gm.search_queries) == 1
        assert gm.search_queries[0] == "test query"
        assert len(gm.grounding_chunks) == 1
        assert gm.grounding_chunks[0].url == "https://example.com"
        assert gm.search_entry_point == "<div>rendered</div>"


class TestImageGenerationTypes:
    """Tests for image generation types."""

    def test_image_generation_config_defaults(self) -> None:
        from gauss._types import ImageGenerationConfig
        cfg = ImageGenerationConfig()
        assert cfg.model is None
        assert cfg.size is None
        assert cfg.quality is None
        assert cfg.style is None
        assert cfg.aspect_ratio is None
        assert cfg.n is None
        assert cfg.response_format is None

    def test_image_generation_config_dalle(self) -> None:
        from gauss._types import ImageGenerationConfig
        cfg = ImageGenerationConfig(
            model="dall-e-3", size="1024x1024", quality="hd", style="vivid", n=1,
        )
        assert cfg.model == "dall-e-3"
        assert cfg.size == "1024x1024"
        assert cfg.quality == "hd"
        assert cfg.style == "vivid"
        assert cfg.n == 1

    def test_image_generation_config_gemini(self) -> None:
        from gauss._types import ImageGenerationConfig
        cfg = ImageGenerationConfig(aspect_ratio="16:9")
        assert cfg.aspect_ratio == "16:9"

    def test_generated_image_data_defaults(self) -> None:
        from gauss._types import GeneratedImageData
        img = GeneratedImageData()
        assert img.url is None
        assert img.base64 is None
        assert img.mime_type is None

    def test_generated_image_data_url(self) -> None:
        from gauss._types import GeneratedImageData
        img = GeneratedImageData(url="https://example.com/image.png", mime_type="image/png")
        assert img.url == "https://example.com/image.png"
        assert img.mime_type == "image/png"

    def test_image_generation_result(self) -> None:
        from gauss._types import GeneratedImageData, ImageGenerationResult
        result = ImageGenerationResult(
            images=[GeneratedImageData(url="https://example.com/img.png")],
            revised_prompt="A beautiful sunset",
        )
        assert len(result.images) == 1
        assert result.images[0].url == "https://example.com/img.png"
        assert result.revised_prompt == "A beautiful sunset"

    def test_image_generation_result_defaults(self) -> None:
        from gauss._types import ImageGenerationResult
        result = ImageGenerationResult()
        assert result.images == []
        assert result.revised_prompt is None


class TestAgentConfigNewFields:
    """Tests for new AgentConfig fields (grounding, native_code_execution, response_modalities)."""

    def test_default_grounding_false(self) -> None:
        from gauss._types import AgentConfig
        cfg = AgentConfig()
        assert cfg.grounding is False

    def test_grounding_enabled(self) -> None:
        from gauss._types import AgentConfig
        cfg = AgentConfig(grounding=True)
        assert cfg.grounding is True

    def test_native_code_execution_default(self) -> None:
        from gauss._types import AgentConfig
        cfg = AgentConfig()
        assert cfg.native_code_execution is False

    def test_native_code_execution_enabled(self) -> None:
        from gauss._types import AgentConfig
        cfg = AgentConfig(native_code_execution=True)
        assert cfg.native_code_execution is True

    def test_response_modalities_default(self) -> None:
        from gauss._types import AgentConfig
        cfg = AgentConfig()
        assert cfg.response_modalities is None

    def test_response_modalities_set(self) -> None:
        from gauss._types import AgentConfig
        cfg = AgentConfig(response_modalities=["TEXT", "IMAGE"])
        assert cfg.response_modalities == ["TEXT", "IMAGE"]

    def test_agent_result_grounding_metadata(self) -> None:
        from gauss._types import AgentResult, GroundingMetadata
        result = AgentResult(
            text="test",
            messages=[],
            tool_calls=[],
            usage={},
            grounding_metadata=[GroundingMetadata(search_queries=["q"])],
        )
        assert len(result.grounding_metadata) == 1
        assert result.grounding_metadata[0].search_queries == ["q"]

    def test_agent_result_grounding_default(self) -> None:
        from gauss._types import AgentResult
        result = AgentResult(text="test", messages=[], tool_calls=[], usage={})
        assert result.grounding_metadata == []
