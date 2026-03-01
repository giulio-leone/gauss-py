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
_mock_native.generate.return_value = "Hello from Gauss!"
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

# stream_generate returns a JSON array of JSON strings (each is a serialized event)
import asyncio

async def _mock_stream_generate(provider_handle, messages_json, temperature=None, max_tokens=None):
    return json.dumps([
        '{"type":"text_delta","text":"Hello"}',
        '{"type":"text_delta","text":" World"}',
        '{"type":"done"}',
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
        agent = Agent()
        agent.destroy()
        with pytest.raises(RuntimeError, match="destroyed"):
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


# ===========================================================================
# Memory Tests
# ===========================================================================

class TestMemory:
    def test_store_and_recall(self) -> None:
        from gauss.memory import Memory
        mem = Memory()
        mem.store("user", "Hello!", session_id="s1")
        entries = mem.recall(session_id="s1")
        assert len(entries) == 1
        assert entries[0]["content"] == "Hi"
        # Verify recall passes JSON options
        _mock_native.memory_recall.assert_called_once_with(
            2, '{"sessionId": "s1"}'
        )
        mem.destroy()

    def test_store_message_object(self) -> None:
        from gauss._types import Message
        from gauss.memory import Memory
        mem = Memory()
        mem.store(Message("assistant", "World"))
        _mock_native.memory_store.assert_called_once()
        mem.destroy()

    def test_context_manager(self) -> None:
        from gauss.memory import Memory
        with Memory() as mem:
            mem.store("user", "Test")
        _mock_native.destroy_memory.assert_called()


# ===========================================================================
# VectorStore Tests
# ===========================================================================

class TestVectorStore:
    def test_upsert_and_search(self) -> None:
        from gauss.vector_store import Chunk, VectorStore
        store = VectorStore()
        store.upsert([Chunk(id="c1", text="hello", embedding=[0.1, 0.2])])
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
        from gauss.graph import Graph
        g = Graph()
        g.destroy()
        with pytest.raises(RuntimeError, match="destroyed"):
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
    def test_server(self) -> None:
        from gauss._types import ToolDef
        from gauss.mcp import McpServer
        srv = McpServer("test", "1.0.0")
        srv.add_tool(ToolDef("greet", "Say hello"))
        resp = srv.handle_message({"jsonrpc": "2.0", "method": "tools/list"})
        assert resp == {"jsonrpc": "2.0", "result": {}}
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
        from gauss.agent import BatchItem
        item = BatchItem("hello")
        assert "hello" in repr(item)
        assert "error" in repr(item)

    def test_batch_item_ok(self) -> None:
        from gauss._types import AgentResult
        from gauss.agent import BatchItem
        item = BatchItem("hello")
        item.result = AgentResult(text="world", messages=[], tool_calls=[], usage={})
        assert "ok" in repr(item)

    def test_batch_item_error(self) -> None:
        from gauss.agent import BatchItem
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
        from gauss.agent import StreamEvent
        event = StreamEvent.from_json("not json{{{")
        assert event.type == "raw"
        assert event.text == "not json{{{"

    def test_stream_event_from_json_valid(self) -> None:
        from gauss.agent import StreamEvent
        event = StreamEvent.from_json('{"type":"text_delta","text":"hi","extra":1}')
        assert event.type == "text_delta"
        assert event.text == "hi"
        assert event.raw == {"extra": 1}

    def test_stream_exports(self) -> None:
        from gauss import AgentStream, StreamEvent
        assert AgentStream is not None
        assert StreamEvent is not None

    @pytest.mark.asyncio
    async def test_stream_raises_after_destroy(self) -> None:
        from gauss import Agent
        agent = Agent()
        agent.destroy()
        with pytest.raises(RuntimeError, match="destroyed"):
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
