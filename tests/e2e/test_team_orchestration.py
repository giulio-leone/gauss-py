"""
E2E tests for Team, Graph, Workflow, Network orchestration.

These tests use a mock HTTP server simulating OpenAI API responses,
exercising the full Rust native layer end-to-end without real API keys.

Run:
    python -m pytest tests/e2e/test_team_orchestration.py -v
"""

from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

import pytest

from gauss import Agent, AgentConfig, Graph, Network, ProviderType, Team, Workflow


# ── Mock OpenAI Server ───────────────────────────────────────────────────────


class _MockOpenAIHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI /v1/chat/completions mock.

    Returns deterministic responses, incrementing a per-server counter
    so each agent call receives a unique reply.
    """

    def do_POST(self) -> None:
        server: _CountingHTTPServer = self.server  # type: ignore[assignment]
        server.call_count += 1
        n = server.call_count

        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)  # drain body

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        body = json.dumps(
            {
                "id": f"chatcmpl-{n}",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"response-{n}",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )
        self.wfile.write(body.encode())

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # silence request logs


class _CountingHTTPServer(HTTPServer):
    call_count: int = 0


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_openai() -> tuple[str, _CountingHTTPServer]:
    """Start a mock OpenAI server; yield (base_url, server); shutdown on exit."""
    server = _CountingHTTPServer(("127.0.0.1", 0), _MockOpenAIHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/v1", server
    server.shutdown()


def _make_agent(base_url: str, name: str) -> Agent:
    return Agent(
        AgentConfig(
            name=name,
            provider=ProviderType.OPENAI,
            model="gpt-4o-mini",
            api_key="sk-fake-e2e",
            base_url=base_url,
        )
    )


# ── Team Tests ───────────────────────────────────────────────────────────────


class TestTeamOrchestration:
    def test_team_sequential(self, mock_openai: tuple[str, _CountingHTTPServer]) -> None:
        base_url, server = mock_openai
        a1 = _make_agent(base_url, "researcher")
        a2 = _make_agent(base_url, "writer")

        team = Team("seq-team")
        team.add(a1).add(a2).strategy("sequential")
        result = team.run("Explain quantum computing")

        assert "finalText" in result
        assert len(result["results"]) == 2
        assert result["results"][0]["text"] == "response-1"
        assert result["results"][1]["text"] == "response-2"
        assert server.call_count == 2

        team.destroy()
        a1.destroy()
        a2.destroy()

    def test_team_parallel(self, mock_openai: tuple[str, _CountingHTTPServer]) -> None:
        base_url, server = mock_openai
        a1 = _make_agent(base_url, "alpha")
        a2 = _make_agent(base_url, "beta")

        team = Team("par-team")
        team.add(a1).add(a2).strategy("parallel")
        result = team.run("Parallel task")

        assert "finalText" in result
        assert len(result["results"]) == 2
        # Parallel results merge with separator
        assert "response-1" in result["finalText"]
        assert "response-2" in result["finalText"]

        team.destroy()
        a1.destroy()
        a2.destroy()

    def test_team_context_manager(self, mock_openai: tuple[str, _CountingHTTPServer]) -> None:
        base_url, _ = mock_openai
        a1 = _make_agent(base_url, "cm1")
        a2 = _make_agent(base_url, "cm2")

        with Team("ctx-team") as team:
            team.add(a1).add(a2).strategy("sequential")
            result = team.run("Context manager test")
            assert result["finalText"] != ""

        assert team._destroyed
        a1.destroy()
        a2.destroy()


# ── Graph Tests ──────────────────────────────────────────────────────────────


class TestGraphOrchestration:
    def test_graph_linear(self, mock_openai: tuple[str, _CountingHTTPServer]) -> None:
        base_url, server = mock_openai
        a1 = _make_agent(base_url, "step1")
        a2 = _make_agent(base_url, "step2")

        graph = Graph()
        graph.add_node("first", a1).add_node("second", a2)
        graph.add_edge("first", "second")
        result = graph.run("Linear pipeline")

        assert "outputs" in result
        assert "first" in result["outputs"]
        assert "second" in result["outputs"]
        assert result["outputs"]["first"]["text"] == "response-1"
        assert result["outputs"]["second"]["text"] == "response-2"
        assert result["final_text"] == "response-2"
        assert server.call_count == 2

        graph.destroy()
        a1.destroy()
        a2.destroy()

    def test_graph_fork_join(self, mock_openai: tuple[str, _CountingHTTPServer]) -> None:
        base_url, server = mock_openai
        fa = _make_agent(base_url, "fork-a")
        fb = _make_agent(base_url, "fork-b")
        merge = _make_agent(base_url, "merger")

        graph = Graph()
        graph.add_fork("parallel", [fa, fb], consensus="concat")
        graph.add_node("merge", merge)
        graph.add_edge("parallel", "merge")
        result = graph.run("Fork-join test")

        outputs = result["outputs"]
        assert "parallel" in outputs
        assert "merge" in outputs
        # Fork concatenates two sub-agent responses
        parallel_text = outputs["parallel"]["text"]
        assert "response-1" in parallel_text
        assert "response-2" in parallel_text
        # Merge node gets the merged input and produces final response
        assert outputs["merge"]["text"] == "response-3"
        assert server.call_count == 3

        graph.destroy()
        fa.destroy()
        fb.destroy()
        merge.destroy()

    def test_graph_three_node_chain(
        self, mock_openai: tuple[str, _CountingHTTPServer]
    ) -> None:
        base_url, _ = mock_openai
        agents = [_make_agent(base_url, f"n{i}") for i in range(3)]

        graph = Graph()
        for i, a in enumerate(agents):
            graph.add_node(f"n{i}", a)
        graph.add_edge("n0", "n1").add_edge("n1", "n2")
        result = graph.run("Three-step chain")

        assert len(result["outputs"]) == 3
        assert result["final_text"] == "response-3"

        graph.destroy()
        for a in agents:
            a.destroy()

    def test_graph_context_manager(
        self, mock_openai: tuple[str, _CountingHTTPServer]
    ) -> None:
        base_url, _ = mock_openai
        a = _make_agent(base_url, "solo")

        with Graph() as g:
            g.add_node("only", a)
            result = g.run("Single node")
            assert result["final_text"] != ""

        assert g._destroyed
        a.destroy()


# ── Workflow Tests ────────────────────────────────────────────────────────────


class TestWorkflowOrchestration:
    def test_workflow_sequential(
        self, mock_openai: tuple[str, _CountingHTTPServer]
    ) -> None:
        base_url, server = mock_openai
        planner = _make_agent(base_url, "planner")
        executor = _make_agent(base_url, "executor")

        wf = Workflow()
        wf.add_step("plan", planner).add_step("execute", executor)
        wf.add_dependency("execute", "plan")
        result = wf.run("Build a REST API")

        steps = result["steps"]
        assert "plan" in steps
        assert "execute" in steps
        assert steps["plan"]["text"] == "response-1"
        assert steps["execute"]["text"] == "response-2"
        assert server.call_count == 2

        wf.destroy()
        planner.destroy()
        executor.destroy()

    def test_workflow_three_steps(
        self, mock_openai: tuple[str, _CountingHTTPServer]
    ) -> None:
        base_url, _ = mock_openai
        agents = [_make_agent(base_url, f"s{i}") for i in range(3)]

        wf = Workflow()
        for i, a in enumerate(agents):
            wf.add_step(f"s{i}", a)
        wf.add_dependency("s1", "s0").add_dependency("s2", "s1")
        result = wf.run("Three-step workflow")

        assert len(result["steps"]) == 3
        assert result["steps"]["s0"]["text"] == "response-1"
        assert result["steps"]["s2"]["text"] == "response-3"

        wf.destroy()
        for a in agents:
            a.destroy()

    def test_workflow_context_manager(
        self, mock_openai: tuple[str, _CountingHTTPServer]
    ) -> None:
        base_url, _ = mock_openai
        a = _make_agent(base_url, "solo-wf")

        with Workflow() as wf:
            wf.add_step("only", a)
            result = wf.run("Single step")
            assert "steps" in result

        assert wf._destroyed
        a.destroy()


# ── Network Tests ─────────────────────────────────────────────────────────────


class TestNetworkOrchestration:
    def test_network_delegate(
        self, mock_openai: tuple[str, _CountingHTTPServer]
    ) -> None:
        base_url, server = mock_openai
        analyst = _make_agent(base_url, "analyst")
        coder = _make_agent(base_url, "coder")

        net = Network()
        net.add_agent(analyst).add_agent(coder).set_supervisor("analyst")
        result = net.delegate("coder", "Write a sorting algorithm")

        assert result["success"] is True
        assert result["agent_name"] == "coder"
        assert result["result_text"] == "response-1"
        assert server.call_count == 1

        net.destroy()
        analyst.destroy()
        coder.destroy()

    def test_network_delegate_to_supervisor(
        self, mock_openai: tuple[str, _CountingHTTPServer]
    ) -> None:
        base_url, _ = mock_openai
        supervisor = _make_agent(base_url, "supervisor")
        worker = _make_agent(base_url, "worker")

        net = Network()
        net.add_agent(supervisor).add_agent(worker).set_supervisor("supervisor")
        result = net.delegate("supervisor", "Review the code")

        assert result["success"] is True
        assert result["agent_name"] == "supervisor"

        net.destroy()
        supervisor.destroy()
        worker.destroy()

    def test_network_context_manager(
        self, mock_openai: tuple[str, _CountingHTTPServer]
    ) -> None:
        base_url, _ = mock_openai
        a = _make_agent(base_url, "net-solo")

        with Network() as net:
            net.add_agent(a).set_supervisor("net-solo")
            result = net.delegate("net-solo", "Solo delegation")
            assert result["success"] is True

        assert net._destroyed
        a.destroy()
