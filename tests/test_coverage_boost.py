"""
Targeted coverage tests for dispose patterns, context managers, and edge cases.
Covers: _check_alive, __enter__/__exit__, compose_async, and uncovered method paths.
"""

from __future__ import annotations

import asyncio
import json
import sys
from unittest.mock import MagicMock

import pytest

# ─── Mock native module ────────────────────────────────────────────

_mock_native = MagicMock()

# Checkpoint
_mock_native.create_checkpoint_store.return_value = 400
_mock_native.destroy_checkpoint_store.return_value = None
_mock_native.checkpoint_save.return_value = None
_mock_native.checkpoint_load.return_value = json.dumps({"id": "cp1", "data": {}})

# ToolValidator
_mock_native.create_tool_validator.return_value = 200
_mock_native.destroy_tool_validator.return_value = None
_mock_native.tool_validator_validate.return_value = json.dumps({"valid": True, "data": {}})

# Network
_mock_native.create_network.return_value = 700
_mock_native.destroy_network.return_value = None
_mock_native.network_add_agent.return_value = None
_mock_native.network_set_supervisor.return_value = None
_mock_native.network_delegate.return_value = json.dumps({"result": "ok"})
_mock_native.network_agent_cards.return_value = json.dumps([{"name": "a1"}])

# Memory
_mock_native.create_memory.return_value = 100
_mock_native.destroy_memory.return_value = None
_mock_native.memory_store.return_value = None
_mock_native.memory_recall.return_value = json.dumps([])
_mock_native.memory_clear.return_value = None

# Graph
_mock_native.create_graph.return_value = 900
_mock_native.destroy_graph.return_value = None
_mock_native.graph_add_node.return_value = None
_mock_native.graph_add_edge.return_value = None
_mock_native.graph_run = MagicMock(return_value=json.dumps({"node1": "result"}))

# Workflow
_mock_native.create_workflow.return_value = 600
_mock_native.destroy_workflow.return_value = None
_mock_native.workflow_add_step.return_value = None
_mock_native.workflow_add_dependency.return_value = None
_mock_native.workflow_run = MagicMock(return_value=json.dumps({"step1": "done"}))

# Provider (needed for Network delegate)
_mock_native.create_provider.return_value = 1
_mock_native.destroy_provider.return_value = None

# Telemetry
_mock_native.create_telemetry.return_value = 800
_mock_native.destroy_telemetry.return_value = None
_mock_native.telemetry_record_span.return_value = None
_mock_native.telemetry_export_spans.return_value = json.dumps([])
_mock_native.telemetry_export_metrics.return_value = json.dumps([])
_mock_native.telemetry_clear.return_value = None

# VectorStore
_mock_native.create_vector_store.return_value = 300
_mock_native.destroy_vector_store.return_value = None
_mock_native.vector_store_upsert.return_value = None
_mock_native.vector_store_search.return_value = json.dumps([])


@pytest.fixture(autouse=True)
def _mock_gauss_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gauss._native", _mock_native)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")


# ─── Context Manager Tests ─────────────────────────────────────────


class TestCheckpointStoreContextManager:
    def test_context_manager(self) -> None:
        from gauss.checkpoint import CheckpointStore

        with CheckpointStore() as cs:
            cs.save({"id": "test", "session_id": "s1"})

    def test_load_latest(self) -> None:
        from gauss.checkpoint import CheckpointStore

        cs = CheckpointStore()
        result = cs.load_latest("session-1")
        assert result is not None
        cs.destroy()

    def test_check_alive_after_destroy(self) -> None:
        from gauss.checkpoint import CheckpointStore
        from gauss.errors import DisposedError

        cs = CheckpointStore()
        cs.destroy()
        with pytest.raises(DisposedError):
            cs.save({"id": "test"})


class TestToolValidatorContextManager:
    def test_context_manager(self) -> None:
        from gauss.tool_validator import ToolValidator

        with ToolValidator([]) as tv:
            result = tv.validate({"name": "test"}, {"type": "object"})
            assert result is not None

    def test_check_alive_after_destroy(self) -> None:
        from gauss.errors import DisposedError
        from gauss.tool_validator import ToolValidator

        tv = ToolValidator([])
        tv.destroy()
        with pytest.raises(DisposedError):
            tv.validate({"name": "test"}, {"type": "object"})


class TestNetworkContextManager:
    def test_context_manager(self) -> None:
        from gauss.network import Network

        with Network() as net:
            net.set_supervisor("agent-1")

    def test_check_alive_after_destroy(self) -> None:
        from gauss.errors import DisposedError
        from gauss.network import Network

        net = Network()
        net.destroy()
        with pytest.raises(DisposedError):
            net.set_supervisor("agent-1")


# ─── Memory Edge Cases ─────────────────────────────────────────────


class TestMemoryEdgeCases:
    def test_clear(self) -> None:
        from gauss.memory import Memory

        m = Memory()
        m.clear()
        m.destroy()

    def test_recall_with_limit(self) -> None:
        from gauss.memory import Memory

        m = Memory()
        result = m.recall("query", limit=5)
        assert isinstance(result, list)
        m.destroy()

    def test_store_invalid_args(self) -> None:
        from gauss.memory import Memory

        m = Memory()
        with pytest.raises((TypeError, ValueError)):
            m.store(12345)  # type: ignore[arg-type]
        m.destroy()

    def test_check_alive_after_destroy(self) -> None:
        from gauss.errors import DisposedError
        from gauss.memory import Memory

        m = Memory()
        m.destroy()
        with pytest.raises(DisposedError):
            m.recall("test")


# ─── Graph/Workflow Tool Conversion & Run ──────────────────────────


class TestGraphToolConversionAndRun:
    def test_add_node_with_tool_dicts(self) -> None:
        from gauss.graph import Graph

        mock_agent = MagicMock()
        mock_agent._handle = 1
        mock_agent._name = "test-agent"

        g = Graph()
        g.add_node(
            "n1",
            mock_agent,
            instructions="do stuff",
            tools=[{"name": "tool1", "description": "d", "parameters": {}}],
        )
        g.destroy()

    def test_run(self) -> None:
        from gauss.graph import Graph

        mock_agent = MagicMock()
        mock_agent._handle = 1
        mock_agent._name = "test-agent"

        g = Graph()
        g.add_node("n1", mock_agent, instructions="step 1")
        result = g.run("test input")
        assert result is not None
        g.destroy()


class TestWorkflowToolConversionAndRun:
    def test_add_step_with_tools(self) -> None:
        from gauss.workflow import Workflow

        mock_agent = MagicMock()
        mock_agent._handle = 1
        mock_agent._name = "test-agent"

        w = Workflow()
        w.add_step(
            "s1",
            mock_agent,
            instructions="do it",
            tools=[{"name": "search", "description": "search web", "parameters": {}}],
        )
        w.destroy()

    def test_run(self) -> None:
        from gauss.workflow import Workflow

        mock_agent = MagicMock()
        mock_agent._handle = 1
        mock_agent._name = "test-agent"

        w = Workflow()
        w.add_step("s1", mock_agent, instructions="step 1")
        result = w.run("input")
        assert result is not None
        w.destroy()


# ─── Pipeline compose_async ────────────────────────────────────────


class TestComposeAsync:
    def test_compose_async(self) -> None:
        from gauss.pipeline import compose_async

        async def double(x: int) -> int:
            return x * 2

        async def add_one(x: int) -> int:
            return x + 1

        async def _test() -> int:
            composed = await compose_async(double, add_one)
            return await composed(5)

        result = asyncio.run(_test())
        assert result == 11  # (5 * 2) + 1

    def test_compose_async_single(self) -> None:
        from gauss.pipeline import compose_async

        async def identity(x: str) -> str:
            return x

        async def _test() -> str:
            composed = await compose_async(identity)
            return await composed("hello")

        result = asyncio.run(_test())
        assert result == "hello"

    def test_compose_async_chain_three(self) -> None:
        from gauss.pipeline import compose_async

        async def step1(x: int) -> int:
            return x + 1

        async def step2(x: int) -> int:
            return x * 10

        async def step3(x: int) -> str:
            return f"result:{x}"

        async def _test() -> str:
            composed = await compose_async(step1, step2, step3)
            return await composed(3)

        result = asyncio.run(_test())
        assert result == "result:40"


# ─── Double Destroy Safety ─────────────────────────────────────────


class TestDoubleDestroy:
    def test_checkpoint(self) -> None:
        from gauss.checkpoint import CheckpointStore

        cs = CheckpointStore()
        cs.destroy()
        cs.destroy()  # should not raise

    def test_tool_validator(self) -> None:
        from gauss.tool_validator import ToolValidator

        tv = ToolValidator([])
        tv.destroy()
        tv.destroy()

    def test_network(self) -> None:
        from gauss.network import Network

        n = Network()
        n.destroy()
        n.destroy()

    def test_graph(self) -> None:
        from gauss.graph import Graph

        g = Graph()
        g.destroy()
        g.destroy()

    def test_workflow(self) -> None:
        from gauss.workflow import Workflow

        w = Workflow()
        w.destroy()
        w.destroy()
