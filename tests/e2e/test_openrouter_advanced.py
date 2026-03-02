"""E2E: OpenRouter Advanced Features — Graph, Network, Workflow, Middleware.

Model: arcee-ai/trinity-large-preview:free
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import pytest

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "arcee-ai/trinity-large-preview:free"


def _agent_opts(**overrides):
    from gauss._types import ProviderType
    return {
        "provider": ProviderType.OPENAI,
        "model": MODEL,
        "api_key": OPENROUTER_KEY,
        "base_url": "https://openrouter.ai/api/v1",
        "max_tokens": 80,
        **overrides,
    }


@dataclass
class E2ETestResult:
    test: str
    feature: str
    success: bool
    latency_ms: float
    output: str = ""
    error: str = ""


results: list[E2ETestResult] = []


def record(r: E2ETestResult) -> None:
    results.append(r)
    icon = "✅" if r.success else "❌"
    print(f"  {icon} {r.feature:<30} {r.test:<25} {r.latency_ms:.0f}ms")
    if r.output:
        print(f"     → {r.output[:80]}")
    if r.error:
        print(f"     ✖ {r.error[:80]}")


@pytest.fixture(autouse=True)
def _skip_without_key():
    if not OPENROUTER_KEY:
        pytest.skip("OPENROUTER_API_KEY not set")


class TestOpenRouterAdvanced:
    """OpenRouter advanced features — arcee-ai/trinity-large-preview:free."""

    # ─── 1. Basic Agent.run ──────────────────────────────────────────
    def test_basic_run(self) -> None:
        from gauss import Agent, AgentConfig

        start = time.time()
        with Agent(AgentConfig(
            **_agent_opts(),
            name="basic",
            system_prompt="Be concise. Answer in 1-2 words.",
        )) as agent:
            result = agent.run("What is 2+2?")
            latency = (time.time() - start) * 1000

            record(E2ETestResult(
                test="basic-run", feature="Agent.run",
                success=True, latency_ms=latency,
                output=result.text[:50],
            ))
            assert len(result.text) > 0

    # ─── 2. Generate ─────────────────────────────────────────────────
    def test_generate(self) -> None:
        from gauss import Agent, AgentConfig

        start = time.time()
        with Agent(AgentConfig(
            **_agent_opts(),
            name="gen",
            system_prompt="Reply with just the number.",
        )) as agent:
            text = agent.generate("What is 1+1?")
            latency = (time.time() - start) * 1000

            record(E2ETestResult(
                test="generate", feature="Agent.generate",
                success=True, latency_ms=latency,
                output=text[:30],
            ))
            assert len(text) > 0

    # ─── 3. gauss() one-liner ────────────────────────────────────────
    def test_gauss_oneliner(self) -> None:
        from gauss import gauss

        start = time.time()
        text = gauss(
            "What is 1+1? Reply with just the number.",
            **_agent_opts(),
            system_prompt="Reply only with the number.",
        )
        latency = (time.time() - start) * 1000

        record(E2ETestResult(
            test="gauss-oneliner", feature="gauss()",
            success=True, latency_ms=latency,
            output=text[:20],
        ))
        assert "2" in text

    # ─── 4. Structured Output ────────────────────────────────────────
    def test_structured_output(self) -> None:
        from gauss import Agent, AgentConfig
        from gauss.structured import structured

        start = time.time()
        with Agent(AgentConfig(
            **{**_agent_opts(), "max_tokens": 120},
            name="structurer",
            system_prompt="Output valid JSON only. No markdown, no explanation.",
        )) as agent:
            r = structured(
                agent,
                "Bob is 25 years old.",
                schema={
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
                    "required": ["name", "age"],
                },
            )
            latency = (time.time() - start) * 1000

            record(E2ETestResult(
                test="structured-output", feature="structured()",
                success=True, latency_ms=latency,
                output=json.dumps(r.data),
            ))
            assert r.data["name"].lower() == "bob"
            assert r.data["age"] == 25

    # ─── 5. Template ─────────────────────────────────────────────────
    def test_template(self) -> None:
        from gauss import Agent, AgentConfig
        from gauss.template import template

        start = time.time()
        t = template("Say '{{word}}' in {{lang}}. Reply with just the word.")
        with Agent(AgentConfig(
            **_agent_opts(),
            name="translator",
            system_prompt="Reply with exactly one word.",
        )) as agent:
            result = agent.run(t(word="hello", lang="Italian"))
            latency = (time.time() - start) * 1000

            record(E2ETestResult(
                test="template", feature="template()",
                success=True, latency_ms=latency,
                output=result.text[:20],
            ))
            assert "ciao" in result.text.lower()

    # ─── 6. Batch ────────────────────────────────────────────────────
    def test_batch(self) -> None:
        from gauss.agent import batch

        start = time.time()
        results_batch = batch(
            ["What is 1+1?", "What is 2+2?"],
            concurrency=2,
            **_agent_opts(),
            name="batcher",
            system_prompt="Reply with just the number.",
        )
        latency = (time.time() - start) * 1000

        record(E2ETestResult(
            test="batch", feature="batch()",
            success=True, latency_ms=latency,
            output=", ".join(r.result.text[:10] for r in results_batch if r.result),
        ))
        assert len(results_batch) == 2

    # ─── 7. Pipeline ─────────────────────────────────────────────────
    def test_pipeline(self) -> None:
        from gauss import Agent, AgentConfig
        from gauss.pipeline import pipe
        import asyncio

        start = time.time()

        def ask_country(city: str) -> str:
            return f"{city} is a city in which country? Reply with just the country name."

        def run_agent(prompt: str) -> str:
            with Agent(AgentConfig(
                **_agent_opts(),
                name="geographer",
                system_prompt="Reply with just the country name.",
            )) as agent:
                return agent.run(prompt).text

        result = asyncio.run(pipe("Rome", ask_country, run_agent))
        latency = (time.time() - start) * 1000

        record(E2ETestResult(
            test="pipeline", feature="pipe()",
            success=True, latency_ms=latency,
            output=result[:20],
        ))
        assert "italy" in result.lower()

    # ─── 8. Graph (DAG multi-agent) ──────────────────────────────────
    def test_graph(self) -> None:
        from gauss import Agent, AgentConfig
        from gauss.graph import Graph

        start = time.time()
        researcher = Agent(AgentConfig(
            **{**_agent_opts(), "max_tokens": 100},
            name="researcher",
            system_prompt="Provide a brief 1-sentence fact.",
        ))
        writer = Agent(AgentConfig(
            **{**_agent_opts(), "max_tokens": 100},
            name="writer",
            system_prompt="Summarize the research in one sentence.",
        ))

        graph = Graph()
        graph.add_node("research", researcher)
        graph.add_node("write", writer)
        graph.add_edge("research", "write")

        result = graph.run("Tell me about black holes")
        latency = (time.time() - start) * 1000

        record(E2ETestResult(
            test="graph-dag", feature="Graph (DAG)",
            success=True, latency_ms=latency,
            output=str(result)[:80],
        ))
        assert result is not None
        researcher.destroy()
        writer.destroy()

    # ─── 9. Network ──────────────────────────────────────────────────
    def test_network(self) -> None:
        from gauss import Agent, AgentConfig
        from gauss.network import Network

        start = time.time()
        expert = Agent(AgentConfig(
            **{**_agent_opts(), "max_tokens": 100},
            name="math-expert",
            system_prompt="You are a math expert. Answer precisely.",
        ))
        supervisor = Agent(AgentConfig(
            **{**_agent_opts(), "max_tokens": 100},
            name="supervisor",
            system_prompt="You are a supervisor. Delegate to specialists.",
        ))

        net = Network()
        net.add_agent(expert)
        net.add_agent(supervisor)
        net.set_supervisor("supervisor")

        cards_result = "N/A — agent_cards not exposed in Python SDK"
        latency = (time.time() - start) * 1000

        record(E2ETestResult(
            test="network-setup", feature="Network setup",
            success=True, latency_ms=latency,
            output=cards_result[:80],
        ))

        # Test delegation
        delegate_start = time.time()
        try:
            delegate_result = net.delegate("math-expert", "What is 7*8?")
            delegate_latency = (time.time() - delegate_start) * 1000
            record(E2ETestResult(
                test="network-delegate", feature="Network.delegate()",
                success=True, latency_ms=delegate_latency,
                output=str(delegate_result)[:50],
            ))
        except Exception as e:
            delegate_latency = (time.time() - delegate_start) * 1000
            record(E2ETestResult(
                test="network-delegate", feature="Network.delegate()",
                success=False, latency_ms=delegate_latency,
                error=str(e)[:80],
            ))

        expert.destroy()
        supervisor.destroy()

    # ─── 10. Workflow ────────────────────────────────────────────────
    def test_workflow(self) -> None:
        from gauss import Agent, AgentConfig
        from gauss.workflow import Workflow

        start = time.time()
        worker = Agent(AgentConfig(
            **{**_agent_opts(), "max_tokens": 80},
            name="worker",
            system_prompt="Be brief.",
        ))

        wf = Workflow()
        wf.add_step("greet", worker, instructions="Say hello briefly")
        wf.add_step("farewell", worker, instructions="Say goodbye briefly")
        wf.add_dependency("farewell", "greet")

        result = wf.run("Start the workflow")
        latency = (time.time() - start) * 1000

        record(E2ETestResult(
            test="workflow", feature="Workflow",
            success=True, latency_ms=latency,
            output=str(result)[:80],
        ))
        assert result is not None
        worker.destroy()

    # ─── 11. Middleware ──────────────────────────────────────────────
    def test_middleware(self) -> None:
        from gauss.middleware import MiddlewareChain

        start = time.time()
        chain = MiddlewareChain()
        chain.use_logging()
        chain.use_caching(5000)
        latency = (time.time() - start) * 1000

        record(E2ETestResult(
            test="middleware", feature="MiddlewareChain",
            success=True, latency_ms=latency,
            output="logging + caching configured",
        ))
        assert chain is not None

    # ─── Report ──────────────────────────────────────────────────────
    def test_zz_report(self) -> None:
        """Print final report."""
        print("\n" + "═" * 80)
        print("  OpenRouter Advanced Features Report — arcee-ai/trinity-large-preview:free")
        print("═" * 80)

        passed = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)

        print("─" * 80)
        print(f"  Total: {len(results)} | ✅ {passed} | ❌ {failed}")
        print("═" * 80)
