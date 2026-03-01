"""
Real Provider Integration Tests — gauss-py

Run with actual API keys:
    OPENAI_API_KEY=sk-... python -m pytest tests/e2e/test_providers.py -v
    OPENROUTER_API_KEY=sk-... python -m pytest tests/e2e/test_providers.py -v

These tests call real LLM APIs and verify the full gauss pipeline works end-to-end.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

# ─── Environment Detection ──────────────────────────────────────────

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")

skip_no_openai = pytest.mark.skipif(not OPENAI_KEY, reason="OPENAI_API_KEY not set")
skip_no_openrouter = pytest.mark.skipif(
    not OPENROUTER_KEY, reason="OPENROUTER_API_KEY not set"
)


# ─── Report Helpers ──────────────────────────────────────────────────


@dataclass
class TestResult:
    provider: str
    model: str
    test: str
    success: bool
    latency_ms: float
    error: str | None = None
    output: str | None = None


_results: list[TestResult] = []


def record(r: TestResult) -> None:
    _results.append(r)


# ─── OpenAI Tests ────────────────────────────────────────────────────


@skip_no_openai
class TestOpenAIProvider:
    def test_simple_completion(self) -> None:
        from gauss import Agent

        start = time.time()
        agent = Agent(
            name="test-openai",
            provider="openai",
            model="gpt-4o-mini",
            provider_options={"api_key": OPENAI_KEY},
            instructions="You are a helpful assistant. Reply concisely.",
        )

        try:
            result = asyncio.get_event_loop().run_until_complete(
                agent.arun("What is 2 + 2? Reply with just the number.")
            )
            latency = (time.time() - start) * 1000

            assert result is not None
            text = result.get("text", "") or json.dumps(result)
            assert "4" in text

            record(
                TestResult(
                    provider="openai",
                    model="gpt-4o-mini",
                    test="simple-completion",
                    success=True,
                    latency_ms=latency,
                    output=text[:200],
                )
            )
        finally:
            agent.destroy()

    def test_system_instructions(self) -> None:
        from gauss import Agent

        start = time.time()
        agent = Agent(
            name="test-instructions",
            provider="openai",
            model="gpt-4o-mini",
            provider_options={"api_key": OPENAI_KEY},
            instructions="You are a pirate. Always start your response with 'Arrr!'",
        )

        try:
            result = asyncio.get_event_loop().run_until_complete(
                agent.arun("Say hello")
            )
            latency = (time.time() - start) * 1000

            text = result.get("text", "") or json.dumps(result)
            assert "arrr" in text.lower()

            record(
                TestResult(
                    provider="openai",
                    model="gpt-4o-mini",
                    test="system-instructions",
                    success=True,
                    latency_ms=latency,
                    output=text[:200],
                )
            )
        finally:
            agent.destroy()

    def test_tool_calling(self) -> None:
        from gauss import Agent

        start = time.time()
        agent = Agent(
            name="test-tools",
            provider="openai",
            model="gpt-4o-mini",
            provider_options={"api_key": OPENAI_KEY},
            instructions="Use the calculate tool to solve math problems.",
            tools=[
                {
                    "name": "calculate",
                    "description": "Calculate a mathematical expression",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Math expression",
                            },
                        },
                        "required": ["expression"],
                    },
                }
            ],
        )

        async def tool_executor(name: str, args: str) -> str:
            assert name == "calculate"
            parsed = json.loads(args)
            return json.dumps({"result": eval(parsed["expression"])})

        try:
            result = asyncio.get_event_loop().run_until_complete(
                agent.arun_with_tools("What is 15 * 23?", tool_executor)
            )
            latency = (time.time() - start) * 1000

            text = result.get("text", "") or json.dumps(result)
            assert "345" in text

            record(
                TestResult(
                    provider="openai",
                    model="gpt-4o-mini",
                    test="tool-calling",
                    success=True,
                    latency_ms=latency,
                    output=text[:200],
                )
            )
        finally:
            agent.destroy()

    def test_multi_turn(self) -> None:
        from gauss import Agent

        start = time.time()
        agent = Agent(
            name="test-multiturn",
            provider="openai",
            model="gpt-4o-mini",
            provider_options={"api_key": OPENAI_KEY},
            instructions="Be concise.",
        )

        try:
            r1 = asyncio.get_event_loop().run_until_complete(
                agent.arun("My name is Alice.")
            )
            messages = r1.get("messages", [])
            messages.append({"role": "user", "content": "What is my name?"})

            r2 = asyncio.get_event_loop().run_until_complete(agent.arun(messages))
            latency = (time.time() - start) * 1000

            text = r2.get("text", "") or json.dumps(r2)
            assert "alice" in text.lower()

            record(
                TestResult(
                    provider="openai",
                    model="gpt-4o-mini",
                    test="multi-turn",
                    success=True,
                    latency_ms=latency,
                    output=text[:200],
                )
            )
        finally:
            agent.destroy()


# ─── OpenRouter Tests ──────────────────────────────────────────────────


@skip_no_openrouter
class TestOpenRouterProvider:
    def test_simple_completion(self) -> None:
        from gauss import Agent

        start = time.time()
        agent = Agent(
            name="test-openrouter",
            provider="openai",
            model="openai/gpt-4o-mini",
            provider_options={
                "api_key": OPENROUTER_KEY,
                "base_url": "https://openrouter.ai/api/v1",
            },
            instructions="Reply concisely.",
        )

        try:
            result = asyncio.get_event_loop().run_until_complete(
                agent.arun("What is the capital of France? One word.")
            )
            latency = (time.time() - start) * 1000

            text = result.get("text", "") or json.dumps(result)
            assert "paris" in text.lower()

            record(
                TestResult(
                    provider="openrouter",
                    model="openai/gpt-4o-mini",
                    test="simple-completion",
                    success=True,
                    latency_ms=latency,
                    output=text[:200],
                )
            )
        finally:
            agent.destroy()

    def test_anthropic_via_openrouter(self) -> None:
        from gauss import Agent

        start = time.time()
        agent = Agent(
            name="test-claude",
            provider="openai",
            model="anthropic/claude-3.5-haiku",
            provider_options={
                "api_key": OPENROUTER_KEY,
                "base_url": "https://openrouter.ai/api/v1",
            },
            instructions="Reply concisely.",
        )

        try:
            result = asyncio.get_event_loop().run_until_complete(
                agent.arun("What is 10 + 10? Just the number.")
            )
            latency = (time.time() - start) * 1000

            text = result.get("text", "") or json.dumps(result)
            assert "20" in text

            record(
                TestResult(
                    provider="openrouter",
                    model="anthropic/claude-3.5-haiku",
                    test="claude-completion",
                    success=True,
                    latency_ms=latency,
                    output=text[:200],
                )
            )
        finally:
            agent.destroy()

    def test_tool_calling(self) -> None:
        from gauss import Agent

        start = time.time()
        agent = Agent(
            name="test-openrouter-tools",
            provider="openai",
            model="openai/gpt-4o-mini",
            provider_options={
                "api_key": OPENROUTER_KEY,
                "base_url": "https://openrouter.ai/api/v1",
            },
            instructions="Use the weather tool.",
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        )

        async def tool_executor(name: str, args: str) -> str:
            assert name == "get_weather"
            return json.dumps({"temperature": 22, "condition": "sunny"})

        try:
            result = asyncio.get_event_loop().run_until_complete(
                agent.arun_with_tools("What's the weather in Rome?", tool_executor)
            )
            latency = (time.time() - start) * 1000

            record(
                TestResult(
                    provider="openrouter",
                    model="openai/gpt-4o-mini",
                    test="tool-calling",
                    success=True,
                    latency_ms=latency,
                    output=str(result)[:200],
                )
            )
        finally:
            agent.destroy()


# ─── Report ──────────────────────────────────────────────────────────


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int) -> None:
    """Print integration test report at end."""
    if not _results:
        return

    terminalreporter.write_line("")
    terminalreporter.write_line("═" * 80)
    terminalreporter.write_line("  GAUSS-PY PROVIDER INTEGRATION TEST REPORT")
    terminalreporter.write_line("═" * 80)

    passed = sum(1 for r in _results if r.success)
    failed = sum(1 for r in _results if not r.success)

    for r in _results:
        status = "✅" if r.success else "❌"
        latency = f"{r.latency_ms:.0f}ms"
        terminalreporter.write_line(
            f"  {status} [{r.provider}/{r.model}] {r.test} — {latency}"
        )
        if r.output:
            terminalreporter.write_line(f"     Output: {r.output[:100]}")
        if r.error:
            terminalreporter.write_line(f"     Error: {r.error}")

    terminalreporter.write_line("─" * 80)
    terminalreporter.write_line(
        f"  Total: {len(_results)} | Passed: {passed} | Failed: {failed}"
    )
    terminalreporter.write_line("═" * 80)
