"""
Gauss Agent — the heart of the SDK.

Quick start::

    from gauss import Agent

    agent = Agent()
    result = agent.run("What is the meaning of life?")
    print(result.text)

One-liner::

    from gauss import gauss
    print(gauss("Explain quantum computing in 3 sentences"))
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from gauss._types import AgentConfig, AgentResult, Message, ToolDef

if TYPE_CHECKING:
    from collections.abc import Sequence


class Agent:
    """Production-grade AI agent powered by Rust.

    Example::

        # Auto-detect provider from env
        agent = Agent()
        result = agent.run("Hello!")
        print(result.text)

        # Explicit config
        agent = Agent(AgentConfig(
            provider=ProviderType.ANTHROPIC,
            model="claude-sonnet-4-20250514",
            system_prompt="You are a helpful assistant.",
        ))

        # Context manager for auto-cleanup
        with Agent() as agent:
            print(agent.run("Hello!").text)
    """

    def __init__(self, config: AgentConfig | None = None, **kwargs: Any) -> None:
        from gauss._native import (  # type: ignore[import-not-found]
            create_provider,
            destroy_provider,
        )

        self._config = config or AgentConfig(**kwargs)
        self._tools: list[ToolDef] = list(self._config.tools)
        self._destroyed = False

        provider_type, model, api_key = self._config.resolve()

        self._provider_handle: int = create_provider(
            provider_type.value,
            model,
            api_key,
            self._config.base_url,
            self._config.max_retries,
        )
        self._model = model
        self._destroy_provider = destroy_provider

    # ── Execution ──────────────────────────────────────────────────────

    def run(self, prompt: str | Sequence[Message | dict[str, str]]) -> AgentResult:
        """Run the agent with a prompt or message list.

        Args:
            prompt: A string (auto-wrapped as user message) or list of messages.

        Returns:
            AgentResult with .text, .messages, .tool_calls, .usage

        Example::

            result = agent.run("What is 2+2?")
            print(result.text)

            result = agent.run([
                Message("system", "Be concise"),
                Message("user", "What is 2+2?"),
            ])
        """
        from gauss._native import agent_run  # type: ignore[import-not-found]

        self._check_alive()
        messages = self._normalize_messages(prompt)
        messages_json = json.dumps(messages)
        tools_json = json.dumps([t.to_dict() for t in self._tools]) if self._tools else "[]"

        result_json: str = agent_run(
            self._provider_handle,
            messages_json,
            tools_json,
            self._config.system_prompt,
            self._config.temperature,
            self._config.max_tokens,
            self._config.stop_condition,
        )

        data = json.loads(result_json)
        return AgentResult(
            text=data.get("text", ""),
            messages=data.get("messages", []),
            tool_calls=data.get("toolCalls", []),
            usage=data.get("usage", {}),
        )

    def generate(self, prompt: str | Sequence[Message | dict[str, str]]) -> str:
        """Simple text generation — returns just the text.

        Example::

            text = agent.generate("Write a haiku about Rust")
        """
        from gauss._native import generate  # type: ignore[import-not-found]

        self._check_alive()
        messages = self._normalize_messages(prompt)

        return generate(
            self._provider_handle,
            json.dumps(messages),
            self._config.temperature,
            self._config.max_tokens,
        )

    def stream_iter(self, prompt: str | Sequence[Message | dict[str, str]]) -> AgentStream:
        """Return an async iterable stream of events.

        Example::

            async for event in agent.stream_iter("Tell me a story"):
                if event.type == "text_delta":
                    print(event.text, end="", flush=True)

            # Access final aggregated text after iteration:
            stream = agent.stream_iter("Tell me a story")
            async for event in stream:
                ...
            print(stream.text)
        """
        self._check_alive()
        messages = self._normalize_messages(prompt)
        return AgentStream(
            provider_handle=self._provider_handle,
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )

    # ── Tool Management ────────────────────────────────────────────────

    def add_tool(self, tool: ToolDef) -> Agent:
        """Add a tool definition. Returns self for chaining.

        Example::

            agent.add_tool(ToolDef(
                name="search",
                description="Search the web",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}}
            ))
        """
        self._tools.append(tool)
        return self

    def add_tools(self, tools: Sequence[ToolDef]) -> Agent:
        """Add multiple tools. Returns self for chaining."""
        self._tools.extend(tools)
        return self

    # ── Lifecycle ──────────────────────────────────────────────────────

    def destroy(self) -> None:
        """Release native resources."""
        if not self._destroyed:
            self._destroy_provider(self._provider_handle)
            self._destroyed = True

    def __enter__(self) -> Agent:
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    def __del__(self) -> None:
        self.destroy()

    # ── Internal ──────────────────────────────────────────────────────

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("Agent has been destroyed")

    @property
    def handle(self) -> int:
        """The native provider handle (for advanced use)."""
        return self._provider_handle

    @staticmethod
    def _normalize_messages(
        prompt: str | Sequence[Message | dict[str, str]],
    ) -> list[dict[str, str]]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        result: list[dict[str, str]] = []
        for msg in prompt:
            if isinstance(msg, Message):
                result.append(msg.to_dict())
            else:
                result.append(msg)
        return result


def gauss(prompt: str, **kwargs: Any) -> str:
    """One-liner AI call. Auto-detects provider from env.

    Example::

        from gauss import gauss

        answer = gauss("What is the meaning of life?")
        print(answer)

        # With options
        answer = gauss("Write a poem", temperature=0.9, model="gpt-4o")
    """
    with Agent(AgentConfig(**kwargs)) as agent:
        return agent.run(prompt).text


# ── Stream events ────────────────────────────────────────────────────


@dataclass
class StreamEvent:
    """A single event from a streaming response.

    Attributes:
        type: Event type (e.g., "text_delta", "tool_call", "raw").
        text: Text content (for text_delta events).
        tool_call: Tool call info dict (for tool_call events).
        raw: Raw data dict for any extra fields.
    """

    type: str
    text: str | None = None
    tool_call: dict[str, str] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, json_str: str) -> StreamEvent:
        """Parse a JSON string into a StreamEvent."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return cls(type="raw", text=json_str)
        return cls(
            type=data.get("type", "unknown"),
            text=data.get("text"),
            tool_call=data.get("toolCall"),
            raw={k: v for k, v in data.items() if k not in ("type", "text", "toolCall")},
        )


class AgentStream:
    """Async iterable wrapper over native streaming.

    Yields :class:`StreamEvent` objects and provides the full aggregated
    text after iteration via :attr:`text`.

    Example::

        stream = agent.stream_iter("Tell me a story")
        async for event in stream:
            if event.type == "text_delta":
                print(event.text, end="", flush=True)
        print()
        print("Full text:", stream.text)
    """

    def __init__(
        self,
        *,
        provider_handle: int,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self._provider_handle = provider_handle
        self._messages = messages
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._events: list[StreamEvent] = []
        self._text: str | None = None

    @property
    def text(self) -> str | None:
        """Aggregated text from all text_delta events (available after iteration)."""
        return self._text

    @property
    def events(self) -> list[StreamEvent]:
        """All events yielded during iteration."""
        return list(self._events)

    async def __aiter__(self) -> AsyncIterator[StreamEvent]:
        from gauss._native import stream_generate  # type: ignore[import-not-found]

        messages_json = json.dumps(self._messages)
        result_json: str = await stream_generate(
            self._provider_handle,
            messages_json,
            self._temperature,
            self._max_tokens,
        )

        raw_events: list[str] = json.loads(result_json)
        parts: list[str] = []

        for raw in raw_events:
            event = StreamEvent.from_json(raw)
            self._events.append(event)
            if event.type == "text_delta" and event.text:
                parts.append(event.text)
            yield event

        self._text = "".join(parts)


# ── Batch execution ──────────────────────────────────────────────────


class BatchItem:
    """Result of a single batch prompt."""

    __slots__ = ("input", "result", "error")

    def __init__(self, input: str) -> None:
        self.input = input
        self.result: AgentResult | None = None
        self.error: Exception | None = None

    def __repr__(self) -> str:
        status = "ok" if self.result else f"error: {self.error}"
        return f"BatchItem(input={self.input!r}, {status})"


def batch(
    prompts: Sequence[str],
    *,
    concurrency: int = 5,
    **kwargs: Any,
) -> list[BatchItem]:
    """Run multiple prompts through an agent with concurrency control.

    Uses a thread pool to execute prompts in parallel.

    Example::

        from gauss import batch

        results = batch(["Translate: Hello", "Translate: World"], concurrency=2)
        for r in results:
            print(r.result.text if r.result else r.error)

    Args:
        prompts: List of string prompts.
        concurrency: Max parallel executions (default: 5).
        **kwargs: Passed to AgentConfig.

    Returns:
        List of BatchItem, one per prompt.
    """
    import concurrent.futures

    items = [BatchItem(p) for p in prompts]
    agent = Agent(AgentConfig(**kwargs))

    def _run(idx: int, item: BatchItem) -> None:
        try:
            item.result = agent.run(item.input)
        except Exception as e:
            item.error = e

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_run, i, item) for i, item in enumerate(items)]
            concurrent.futures.wait(futures)
    finally:
        agent.destroy()

    return items
