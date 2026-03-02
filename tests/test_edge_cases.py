"""Edge case and property-based tests for gauss-py DX utilities.

Uses hypothesis for property-based testing.
"""

from __future__ import annotations

import asyncio
import json
import sys
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Mock native bindings for deterministic edge-case tests.
# ---------------------------------------------------------------------------
_mock_native = MagicMock()
_mock_native.create_provider.return_value = 1
_mock_native.destroy_provider.return_value = None
_mock_native.agent_run.return_value = json.dumps({
    "text": '{"result":"ok"}',
    "messages": [{"role": "assistant", "content": '{"result":"ok"}'}],
    "toolCalls": [],
    "usage": {"total_tokens": 10},
})
_mock_native.generate.return_value = "Hello"
_mock_native.stream_generate = MagicMock(return_value="[]")

from gauss.agent import Agent, AgentConfig
from gauss.batch import batch
from gauss.retry import with_retry, retryable, RetryConfig, _compute_delay
from gauss.structured import structured, StructuredConfig, _extract_json
from gauss.template import template, PromptTemplate
from gauss.pipeline import (
    pipe, map_sync, map_async, filter_sync, filter_async,
    reduce_sync, reduce_async, compose, compose_async,
)

import os
# Set env for auto-detection so Agent() works
os.environ.setdefault("OPENAI_API_KEY", "sk-test-edge-case")


@pytest.fixture(autouse=True)
def _patch_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gauss._native", _mock_native)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-edge-case")


# ═══════════════════════════════════════════════════════════════════
# RETRY EDGE CASES
# ═══════════════════════════════════════════════════════════════════

class TestRetryEdgeCases:
    def test_max_retries_zero_means_no_retries(self) -> None:
        calls = 0
        def fn():
            nonlocal calls
            calls += 1
            raise ValueError("fail")
        with pytest.raises(ValueError, match="fail"):
            with_retry(fn, config=RetryConfig(max_retries=0, base_delay_s=0.001))
        assert calls == 1

    def test_handles_non_exception_types(self) -> None:
        """Retry handles generic Exception subclasses."""
        attempts = 0
        def fn():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("runtime")
            return "ok"
        result = with_retry(fn, config=RetryConfig(max_retries=5, base_delay_s=0.001))
        assert result == "ok"
        assert attempts == 3

    def test_exponential_caps_at_max_delay(self) -> None:
        cfg = RetryConfig(backoff="exponential", base_delay_s=1.0, max_delay_s=5.0, jitter=0)
        delay = _compute_delay(cfg, 10)  # 2^9 * 1.0 = 512 → capped at 5.0
        assert delay <= 5.0

    def test_linear_backoff_is_linear(self) -> None:
        cfg = RetryConfig(backoff="linear", base_delay_s=0.01, max_delay_s=100, jitter=0)
        delays = [_compute_delay(cfg, i) for i in range(1, 4)]
        assert delays == pytest.approx([0.01, 0.02, 0.03])

    def test_fixed_backoff_is_constant(self) -> None:
        cfg = RetryConfig(backoff="fixed", base_delay_s=0.05, max_delay_s=100, jitter=0)
        delays = [_compute_delay(cfg, i) for i in range(1, 4)]
        assert all(d == pytest.approx(0.05) for d in delays)

    def test_retry_if_stops_immediately(self) -> None:
        calls = 0
        def fn():
            nonlocal calls
            calls += 1
            raise ValueError("stop")
        with pytest.raises(ValueError, match="stop"):
            with_retry(fn, config=RetryConfig(
                max_retries=10, base_delay_s=0.001,
                retry_if=lambda e, a: False,
            ))
        assert calls == 1

    def test_on_retry_receives_correct_attempts(self) -> None:
        attempts_seen: list[int] = []
        def fn():
            raise ValueError("fail")
        with pytest.raises(ValueError):
            with_retry(fn, config=RetryConfig(
                max_retries=3, base_delay_s=0.001,
                on_retry=lambda e, attempt, delay: attempts_seen.append(attempt),
            ))
        assert attempts_seen == [1, 2, 3]

    def test_retryable_wraps_agent(self) -> None:
        agent = Agent()
        run = retryable(agent, config=RetryConfig(max_retries=1, base_delay_s=0.001))
        result = run("hello")
        assert result.text is not None
        agent.destroy()

    def test_kwargs_shorthand(self) -> None:
        calls = 0
        def fn():
            nonlocal calls
            calls += 1
            if calls < 2:
                raise ValueError("fail")
            return "ok"
        result = with_retry(fn, max_retries=3, base_delay_s=0.001)
        assert result == "ok"


# ═══════════════════════════════════════════════════════════════════
# STRUCTURED OUTPUT EDGE CASES
# ═══════════════════════════════════════════════════════════════════

class TestStructuredEdgeCases:
    def _set_agent_text(self, text: str) -> None:
        _mock_native.agent_run.return_value = json.dumps({
            "text": text,
            "messages": [{"role": "assistant", "content": text}],
            "toolCalls": [],
            "usage": {"total_tokens": 10},
        })

    def _reset_agent_text(self) -> None:
        _mock_native.agent_run.return_value = json.dumps({
            "text": '{"result":"ok"}',
            "messages": [{"role": "assistant", "content": '{"result":"ok"}'}],
            "toolCalls": [],
            "usage": {"total_tokens": 10},
        })

    def test_deeply_nested_json(self) -> None:
        self._set_agent_text('{"a":{"b":{"c":{"d":42}}}}')
        agent = Agent()
        result = structured(agent, "test", schema={"type": "object"})
        assert result.data["a"]["b"]["c"]["d"] == 42
        agent.destroy()
        self._reset_agent_text()

    def test_json_with_escaped_quotes(self) -> None:
        self._set_agent_text('{"msg":"He said \\"hello\\""}')
        agent = Agent()
        result = structured(agent, "test", schema={"type": "object"})
        assert result.data["msg"] == 'He said "hello"'
        agent.destroy()
        self._reset_agent_text()

    def test_array_at_top_level(self) -> None:
        self._set_agent_text("Here: [1, 2, 3]")
        agent = Agent()
        result = structured(agent, "test", schema={"type": "array"})
        assert result.data == [1, 2, 3]
        agent.destroy()
        self._reset_agent_text()

    def test_json_with_unicode(self) -> None:
        self._set_agent_text('{"emoji":"🎉","kanji":"日本語"}')
        agent = Agent()
        result = structured(agent, "test", schema={"type": "object"})
        assert result.data["emoji"] == "🎉"
        assert result.data["kanji"] == "日本語"
        agent.destroy()
        self._reset_agent_text()

    def test_json_embedded_in_verbose_text(self) -> None:
        self._set_agent_text('Sure! Here:\n\n{"name":"Alice","age":30}\n\nLet me know!')
        agent = Agent()
        result = structured(agent, "test", schema={"type": "object"})
        assert result.data["name"] == "Alice"
        assert result.data["age"] == 30
        agent.destroy()
        self._reset_agent_text()

    def test_max_parse_retries_zero(self) -> None:
        self._set_agent_text("not json at all")
        agent = Agent()
        with pytest.raises(RuntimeError, match="after 1 attempts"):
            structured(agent, "test", schema={"type": "object"}, max_parse_retries=0)
        agent.destroy()
        self._reset_agent_text()

    def test_empty_object(self) -> None:
        self._set_agent_text("{}")
        agent = Agent()
        result = structured(agent, "test", schema={"type": "object"})
        assert result.data == {}
        agent.destroy()
        self._reset_agent_text()

    def test_extract_json_code_block(self) -> None:
        text = "```json\n{\"x\": 1}\n```"
        assert json.loads(_extract_json(text)) == {"x": 1}

    def test_extract_json_no_json_present(self) -> None:
        assert _extract_json("just text") == "just text"


# ═══════════════════════════════════════════════════════════════════
# TEMPLATE EDGE CASES
# ═══════════════════════════════════════════════════════════════════

class TestTemplateEdgeCases:
    def test_empty_template(self) -> None:
        t = template("")
        assert t() == ""
        assert t.variables == []

    def test_no_variables(self) -> None:
        t = template("Just plain text")
        assert t() == "Just plain text"

    def test_variable_at_start(self) -> None:
        t = template("{{greeting}} world")
        assert t(greeting="Hello") == "Hello world"

    def test_variable_at_end(self) -> None:
        t = template("Hello {{name}}")
        assert t(name="world") == "Hello world"

    def test_adjacent_variables(self) -> None:
        t = template("{{a}}{{b}}{{c}}")
        assert t(a="1", b="2", c="3") == "123"

    def test_malformed_syntax_ignored(self) -> None:
        t = template("{{good}} {bad} {{ spaced }} {{}}")
        assert t.variables == ["good"]

    def test_values_containing_template_syntax(self) -> None:
        t = template("Output: {{text}}")
        assert t(text="{{not_a_var}}") == "Output: {{not_a_var}}"

    def test_multiline_template(self) -> None:
        t = template("L1: {{a}}\nL2: {{b}}\nL3: {{c}}")
        assert t(a="A", b="B", c="C") == "L1: A\nL2: B\nL3: C"

    def test_special_regex_chars_in_values(self) -> None:
        t = template("Pattern: {{regex}}")
        assert t(regex="a.*b+c?") == "Pattern: a.*b+c?"

    def test_repr(self) -> None:
        t = template("{{a}} and {{b}}")
        assert "PromptTemplate" in repr(t)
        assert "a" in repr(t)


# ═══════════════════════════════════════════════════════════════════
# PIPELINE EDGE CASES
# ═══════════════════════════════════════════════════════════════════

class TestPipelineEdgeCases:
    @pytest.mark.asyncio
    async def test_pipe_single_identity(self) -> None:
        result = await pipe(42, lambda n: n)
        assert result == 42

    @pytest.mark.asyncio
    async def test_pipe_propagates_errors(self) -> None:
        async def boom(x: str) -> str:
            raise ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            await pipe("hello", boom)

    def test_map_sync_empty(self) -> None:
        result = map_sync([], lambda n: n * 2)
        assert result == []

    def test_map_sync_single(self) -> None:
        result = map_sync([42], lambda n: n * 2)
        assert result == [84]

    @pytest.mark.asyncio
    async def test_map_async_empty(self) -> None:
        async def double(n: int) -> int:
            return n * 2
        result = await map_async([], double)
        assert result == []

    def test_filter_sync_all_match(self) -> None:
        result = filter_sync([1, 2, 3], lambda n: True)
        assert result == [1, 2, 3]

    def test_filter_sync_none_match(self) -> None:
        result = filter_sync([1, 2, 3], lambda n: False)
        assert result == []

    @pytest.mark.asyncio
    async def test_filter_async_all_match(self) -> None:
        async def always_true(n: int) -> bool:
            return True
        result = await filter_async([1, 2, 3], always_true)
        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_filter_async_none_match(self) -> None:
        async def always_false(n: int) -> bool:
            return False
        result = await filter_async([1, 2, 3], always_false)
        assert result == []

    @pytest.mark.asyncio
    async def test_reduce_async_empty(self) -> None:
        async def add(a: int, b: int) -> int:
            return a + b
        result = await reduce_async([], add, 42)
        assert result == 42

    def test_reduce_sync_empty(self) -> None:
        result = reduce_sync([], lambda a, n: a + n, 42)
        assert result == 42

    def test_compose_no_functions(self) -> None:
        fn = compose()
        assert fn(42) == 42

    def test_map_sync_with_concurrency_1(self) -> None:
        order: list[int] = []
        def fn(n: int) -> int:
            order.append(n)
            return n
        map_sync([1, 2, 3], fn, concurrency=1)
        assert sorted(order) == [1, 2, 3]


# ═══════════════════════════════════════════════════════════════════
# BATCH EDGE CASES
# ═══════════════════════════════════════════════════════════════════

class TestBatchEdgeCases:
    def test_batch_empty(self) -> None:
        result = batch([])
        assert result == []

    def test_batch_single(self) -> None:
        result = batch(["hello"])
        assert len(result) == 1
        assert result[0].result is not None

    def test_batch_concurrency_1(self) -> None:
        result = batch(
            ["a", "b"],
            concurrency=1,
        )
        assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════
# PROPERTY-BASED TESTS (hypothesis)
# ═══════════════════════════════════════════════════════════════════

class TestPropertyTemplate:
    @given(st.lists(
        st.from_regex(r"[a-z][a-z0-9]{0,5}", fullmatch=True),
        min_size=1, max_size=5,
    ))
    @settings(max_examples=100)
    def test_rendered_output_has_no_unsubstituted_vars(self, var_names: list[str]) -> None:
        unique = list(dict.fromkeys(var_names))
        assume(len(unique) > 0)
        tpl_str = " ".join(f"{{{{{v}}}}}" for v in unique)
        t = template(tpl_str)
        vals = {v: f"val_{i}" for i, v in enumerate(unique)}
        result = t(**vals)
        for v in unique:
            assert f"{{{{{v}}}}}" not in result

    @given(st.lists(
        st.from_regex(r"[a-z]\w{0,5}", fullmatch=True),
        min_size=0, max_size=10,
    ))
    @settings(max_examples=100)
    def test_variables_match_template(self, vars: list[str]) -> None:
        unique = list(dict.fromkeys(vars))
        tpl_str = " ".join(f"pre {{{{{v}}}}} post" for v in unique)
        t = template(tpl_str)
        assert t.variables == unique


class TestPropertyPipeline:
    @given(st.integers(min_value=-10000, max_value=10000))
    @settings(max_examples=100)
    def test_reduce_sync_sum_matches_builtin(self, n: int) -> None:
        arr = list(range(n % 20))  # keep arrays small
        result = reduce_sync(arr, lambda a, b: a + b, 0)
        assert result == sum(arr)

    @given(st.lists(st.integers(), min_size=0, max_size=50))
    @settings(max_examples=100)
    def test_filter_sync_is_subset(self, arr: list[int]) -> None:
        result = filter_sync(arr, lambda n: n > 0)
        for item in result:
            assert item in arr
            assert item > 0

    @given(st.lists(st.integers(), min_size=0, max_size=50))
    @settings(max_examples=100)
    def test_map_sync_preserves_length(self, arr: list[int]) -> None:
        result = map_sync(arr, lambda n: n * 2)
        assert len(result) == len(arr)

    @given(st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=100)
    def test_compose_matches_manual(self, n: int) -> None:
        f = lambda x: x * 2
        g = lambda x: x + 1
        composed = compose(f, g)
        assert composed(n) == g(f(n))


class TestPropertyRetry:
    @given(st.integers(min_value=0, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_always_calls_at_least_once(self, max_retries: int) -> None:
        calls = 0
        def fn():
            nonlocal calls
            calls += 1
            raise ValueError("fail")
        with pytest.raises(ValueError):
            with_retry(fn, config=RetryConfig(
                max_retries=max_retries, base_delay_s=0.001, jitter=0,
            ))
        assert 1 <= calls <= max_retries + 1

    @given(st.integers(min_value=0, max_value=5), st.text(min_size=0, max_size=20))
    @settings(max_examples=30, deadline=None)
    def test_succeeds_immediately_if_no_error(self, max_retries: int, value: str) -> None:
        result = with_retry(lambda: value, config=RetryConfig(
            max_retries=max_retries, base_delay_s=0.001,
        ))
        assert result == value
