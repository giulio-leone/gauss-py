"""
GPT-5.2 Full Feature Integration Tests — gauss-py

Run: OPENAI_API_KEY=sk-... python -m pytest tests/e2e/test_gpt5.py -v

Tests every gauss-py feature against GPT-5.2 with minimal token usage.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import pytest

# ─── Config ──────────────────────────────────────────────────────────

API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = "gpt-5.2"
skip_no_key = pytest.mark.skipif(not API_KEY, reason="OPENAI_API_KEY not set")


# ─── Report ──────────────────────────────────────────────────────────


@dataclass
class E2ETestResult:
    test: str
    feature: str
    success: bool
    latency_ms: float
    tokens: dict[str, int] | None = None
    output: str | None = None
    error: str | None = None


_results: list[E2ETestResult] = []


def record(r: E2ETestResult) -> None:
    _results.append(r)


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int) -> None:
    if not _results:
        return

    passed = sum(1 for r in _results if r.success)
    failed = sum(1 for r in _results if not r.success)
    total_tokens = sum(
        (r.tokens.get("input", 0) + r.tokens.get("output", 0))
        for r in _results
        if r.tokens
    )

    terminalreporter.write_line("")
    terminalreporter.write_line("═" * 80)
    terminalreporter.write_line("  GPT-5.2 FULL FEATURE INTEGRATION REPORT (Python)")
    terminalreporter.write_line("═" * 80)

    for r in _results:
        icon = "✅" if r.success else "❌"
        tok = ""
        if r.tokens:
            tok = f" [{r.tokens.get('input', 0)}+{r.tokens.get('output', 0)} tok]"
        terminalreporter.write_line(
            f"  {icon} {r.feature:<22} {r.test:<30} {r.latency_ms:.0f}ms{tok}"
        )
        if r.output:
            terminalreporter.write_line(f"     → {r.output[:100]}")
        if r.error:
            terminalreporter.write_line(f"     ✖ {r.error[:100]}")

    terminalreporter.write_line("─" * 80)
    terminalreporter.write_line(
        f"  Total: {len(_results)} | ✅ {passed} | ❌ {failed} | Tokens: {total_tokens}"
    )
    terminalreporter.write_line("═" * 80)


# ─── Helper ──────────────────────────────────────────────────────────

def _agent_opts() -> dict[str, Any]:
    from gauss import ProviderType
    return {
        "provider": ProviderType.OPENAI,
        "model": MODEL,
        "api_key": API_KEY,
        "temperature": 0,
        "max_tokens": 50,
    }


# ─── 1. Simple Completion ────────────────────────────────────────────


@skip_no_key
class TestGPT5Features:
    def test_simple_completion(self) -> None:
        from gauss import Agent, AgentConfig

        start = time.time()
        with Agent(AgentConfig(
            **_agent_opts(),
            system_prompt="Answer in 1 word.",
        )) as agent:
            r = agent.run("Capital of Italy?")
            latency = (time.time() - start) * 1000
            usage = r.usage

            record(E2ETestResult(
                test="simple-completion", feature="Agent.run",
                success=True, latency_ms=latency,
                tokens={"input": usage.get("inputTokens", 0), "output": usage.get("outputTokens", 0)},
                output=r.text,
            ))
            assert "rome" in r.text.lower()

    # ─── 2. System Instructions ──────────────────────────────────────

    def test_system_instructions(self) -> None:
        from gauss import Agent, AgentConfig

        start = time.time()
        with Agent(AgentConfig(
            **_agent_opts(),
            system_prompt="Reply only with JSON: {\"answer\": ...}",
        )) as agent:
            r = agent.run("2+2?")
            latency = (time.time() - start) * 1000

            record(E2ETestResult(
                test="system-instructions", feature="Agent.run",
                success=True, latency_ms=latency,
                tokens={"input": r.usage.get("inputTokens", 0), "output": r.usage.get("outputTokens", 0)},
                output=r.text,
            ))
            assert "{" in r.text

    # ─── 3. Multi-Turn ───────────────────────────────────────────────

    def test_multi_turn(self) -> None:
        from gauss import Agent, AgentConfig, Message

        start = time.time()
        with Agent(AgentConfig(
            **_agent_opts(),
            system_prompt="Be concise.",
        )) as agent:
            r1 = agent.run("My pet is a cat named Luna.")
            r2 = agent.run([
                Message("user", "My pet is a cat named Luna."),
                Message("assistant", r1.text),
                Message("user", "Pet name?"),
            ])
            latency = (time.time() - start) * 1000

            record(E2ETestResult(
                test="multi-turn", feature="Agent.run",
                success=True, latency_ms=latency,
                tokens={"input": r1.usage.get("inputTokens", 0) + r2.usage.get("inputTokens", 0),
                        "output": r1.usage.get("outputTokens", 0) + r2.usage.get("outputTokens", 0)},
                output=r2.text,
            ))
            assert "luna" in r2.text.lower()

    # ─── 4. Generate Raw ─────────────────────────────────────────────

    def test_generate_raw(self) -> None:
        from gauss import Agent, AgentConfig

        start = time.time()
        with Agent(AgentConfig(**_agent_opts())) as agent:
            text = agent.generate("Say OK")
            latency = (time.time() - start) * 1000

            record(E2ETestResult(
                test="generate-raw", feature="Agent.generate",
                success=True, latency_ms=latency,
                output=text,
            ))
            assert text is not None

    # ─── 5. Streaming ────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_streaming(self) -> None:
        from gauss import Agent, AgentConfig

        start = time.time()
        with Agent(AgentConfig(
            **_agent_opts(),
            system_prompt="Count 1-3.",
        )) as agent:
            stream = agent.stream_iter("Go")
            events = []
            async for event in stream:
                events.append(event)
            latency = (time.time() - start) * 1000

            record(E2ETestResult(
                test="streaming", feature="Agent.stream_iter",
                success=True, latency_ms=latency,
                output=f"{len(events)} events, text: {stream.text}",
            ))
            assert len(events) > 0

    # ─── 6. gauss() One-Liner ────────────────────────────────────────

    def test_gauss_oneliner(self) -> None:
        from gauss import gauss as gauss_fn

        start = time.time()
        answer = gauss_fn("2+3? Number only.", **_agent_opts())
        latency = (time.time() - start) * 1000

        record(E2ETestResult(
            test="one-liner", feature="gauss()",
            success=True, latency_ms=latency,
            output=answer,
        ))
        assert "5" in answer

    # ─── 7. Structured Output ────────────────────────────────────────

    def test_structured_output(self) -> None:
        from gauss import Agent, AgentConfig
        from gauss.structured import structured, StructuredConfig

        start = time.time()
        with Agent(AgentConfig(
            **{**_agent_opts(), "max_tokens": 100},
            system_prompt="Output valid JSON only.",
        )) as agent:
            r = structured(
                agent,
                "Alice is 30.",
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
            assert r.data["name"].lower() == "alice"
            assert r.data["age"] == 30

    # ─── 8. Template ─────────────────────────────────────────────────

    def test_template(self) -> None:
        from gauss import gauss as gauss_fn
        from gauss.template import template

        start = time.time()
        t = template("Translate '{{text}}' to {{lang}}. Just the translation.")
        prompt = t(text="hello", lang="Italian")
        answer = gauss_fn(prompt, **_agent_opts())
        latency = (time.time() - start) * 1000

        record(E2ETestResult(
            test="template", feature="template()",
            success=True, latency_ms=latency,
            output=answer,
        ))
        assert "ciao" in answer.lower()

    # ─── 9. Batch ────────────────────────────────────────────────────

    def test_batch(self) -> None:
        from gauss.agent import batch

        start = time.time()
        items = batch(
            ["1+1? Number.", "2+2? Number.", "3+3? Number."],
            concurrency=3,
            **_agent_opts(),
        )
        latency = (time.time() - start) * 1000

        outputs = [i.result.text.strip() if i.result else "ERR" for i in items]
        record(E2ETestResult(
            test="batch-parallel", feature="batch()",
            success=True, latency_ms=latency,
            output=", ".join(outputs),
        ))
        assert all(i.result is not None for i in items)

    # ─── 10. Pipeline ────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_pipeline(self) -> None:
        from gauss import gauss as gauss_fn
        from gauss.pipeline import pipe

        start = time.time()
        result = await pipe(
            "Rome",
            lambda city: gauss_fn(f"Country of {city}? One word.", **_agent_opts()).strip(),
            lambda country: gauss_fn(f"Continent of {country}? One word.", **_agent_opts()).strip(),
        )
        latency = (time.time() - start) * 1000

        record(E2ETestResult(
            test="pipeline", feature="pipe()",
            success=True, latency_ms=latency,
            output=str(result),
        ))
        assert "europe" in str(result).lower()

    # ─── 11. Agent Lifecycle ─────────────────────────────────────────

    def test_lifecycle(self) -> None:
        from gauss import Agent, AgentConfig

        agent = Agent(AgentConfig(**_agent_opts()))
        r = agent.run("Hi")
        assert r.text is not None
        agent.destroy()

        with pytest.raises(RuntimeError, match="destroyed"):
            agent.run("Fail")

        record(E2ETestResult(
            test="lifecycle", feature="Agent.destroy",
            success=True, latency_ms=0,
            output="destroy + post-destroy error OK",
        ))

    # ─── 12. Context Manager ─────────────────────────────────────────

    def test_context_manager(self) -> None:
        from gauss import Agent, AgentConfig

        start = time.time()
        with Agent(AgentConfig(**_agent_opts())) as agent:
            r = agent.run("Say OK")
            assert r.text is not None

        latency = (time.time() - start) * 1000

        # After with-block, agent should be destroyed
        with pytest.raises(RuntimeError, match="destroyed"):
            agent.run("Fail")

        record(E2ETestResult(
            test="context-manager", feature="with Agent()",
            success=True, latency_ms=latency,
            output="with-block + auto-destroy OK",
        ))

    # ─── 13. Agent Properties ────────────────────────────────────────

    def test_agent_properties(self) -> None:
        from gauss import Agent, AgentConfig

        agent = Agent(AgentConfig(**_agent_opts(), name="props-test"))
        assert agent.handle is not None
        assert isinstance(agent.handle, int)
        agent.destroy()

        record(E2ETestResult(
            test="properties", feature="Agent props",
            success=True, latency_ms=0,
            output="handle OK",
        ))
