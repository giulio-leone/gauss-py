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

from gauss._types import (
    AgentConfig,
    AgentResult,
    Citation,
    CodeExecutionOptions,
    CodeExecutionResult,
    GeneratedImageData,
    GroundingChunk,
    GroundingMetadata,
    ImageGenerationConfig,
    ImageGenerationResult,
    Message,
    ProviderCapabilities,
    ProviderType,
    ToolDef,
    detect_provider,
    resolve_api_key,
    _default_model,
)
from gauss.models import OPENAI_DEFAULT

if TYPE_CHECKING:
    from collections.abc import Sequence


def _run_native(func: Any, *args: Any) -> Any:
    """Call a native function that may return a coroutine or a plain value.

    PyO3 functions using ``pyo3_async_runtimes::tokio::future_into_py`` require
    a running asyncio event loop at call-time (they return a coroutine).  This
    helper transparently handles the three execution contexts:

    1. No event loop running → create one with ``asyncio.run``.
    2. Event loop already running (e.g. Jupyter) → offload to a thread.
    3. Mock / sync return → pass through immediately.
    """
    import asyncio
    import inspect

    async def _call() -> Any:
        res = func(*args)
        if inspect.isawaitable(res):
            return await res
        return res

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, _call()).result()

    # Fast path: try sync first (mock environment)
    try:
        result = func(*args)
        if not inspect.isawaitable(result):
            return result
    except RuntimeError:
        pass

    return asyncio.run(_call())


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

        options: dict[str, Any] = {}
        if self._tools:
            options["tools"] = [t.to_dict() for t in self._tools]
        if self._config.system_prompt:
            options["instructions"] = self._config.system_prompt
        if self._config.temperature is not None:
            options["temperature"] = self._config.temperature
        if self._config.max_tokens is not None:
            options["max_tokens"] = self._config.max_tokens
        if self._config.stop_condition:
            options["stop_on_tool"] = self._config.stop_condition
        if self._config.thinking_budget is not None:
            options["thinking_budget"] = self._config.thinking_budget
        if self._config.reasoning_effort is not None:
            options["reasoning_effort"] = self._config.reasoning_effort
        if self._config.cache_control:
            options["cache_control"] = True
        if self._config.code_execution is not None:
            if self._config.code_execution is True:
                options["code_execution"] = True
            elif isinstance(self._config.code_execution, CodeExecutionOptions):
                ce = self._config.code_execution
                options["code_execution"] = {
                    "python": ce.python,
                    "javascript": ce.javascript,
                    "bash": ce.bash,
                    "unified": ce.unified,
                    "timeout": ce.timeout,
                    "sandbox": ce.sandbox,
                }

        if self._config.grounding:
            options["grounding"] = True
        if self._config.native_code_execution:
            options["native_code_execution"] = True
        if self._config.response_modalities is not None:
            options["response_modalities"] = self._config.response_modalities

        result_json = _run_native(
            agent_run,
            self._config.name,
            self._provider_handle,
            messages_json,
            json.dumps(options) if options else None,
        )

        data = json.loads(result_json)
        raw_citations = data.get("citations", [])
        citations = [
            Citation(
                citation_type=c.get("type", ""),
                cited_text=c.get("cited_text"),
                document_title=c.get("document_title"),
                start=c.get("start"),
                end=c.get("end"),
            )
            for c in raw_citations
        ]
        raw_grounding = data.get("grounding_metadata", [])
        grounding_metadata = [
            GroundingMetadata(
                search_queries=gm.get("search_queries", []),
                grounding_chunks=[
                    GroundingChunk(url=gc.get("url"), title=gc.get("title"))
                    for gc in gm.get("grounding_chunks", [])
                ],
                search_entry_point=gm.get("search_entry_point"),
            )
            for gm in (raw_grounding or [])
        ]
        return AgentResult(
            text=data.get("text", ""),
            messages=data.get("messages", []),
            tool_calls=data.get("toolCalls", []),
            usage=data.get("usage", {}),
            thinking=data.get("thinking"),
            citations=citations,
            grounding_metadata=grounding_metadata,
        )

    def generate(self, prompt: str | Sequence[Message | dict[str, str]]) -> str:
        """Simple text generation — returns just the text.

        Example::

            text = agent.generate("Write a haiku about Rust")
        """
        from gauss._native import generate  # type: ignore[import-not-found]

        self._check_alive()
        messages = self._normalize_messages(prompt)

        result_json = _run_native(
            generate,
            self._provider_handle,
            json.dumps(messages),
            self._config.temperature,
            self._config.max_tokens,
        )

        data = json.loads(result_json)
        return data.get("text", "")

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Query what features this provider/model supports."""
        from gauss._native import get_provider_capabilities  # type: ignore[import-not-found]

        self._check_alive()
        caps_json = get_provider_capabilities(self._provider_handle)
        data = json.loads(caps_json)
        return ProviderCapabilities(
            streaming=data.get("streaming", False),
            tool_use=data.get("tool_use", False),
            vision=data.get("vision", False),
            audio=data.get("audio", False),
            extended_thinking=data.get("extended_thinking", False),
            citations=data.get("citations", False),
            cache_control=data.get("cache_control", False),
            structured_output=data.get("structured_output", False),
            reasoning_effort=data.get("reasoning_effort", False),
            image_generation=data.get("image_generation", False),
            grounding=data.get("grounding", False),
            code_execution=data.get("code_execution", False),
            web_search=data.get("web_search", False),
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

    # ── Code Execution ────────────────────────────────────────────────

    @staticmethod
    def execute_code(
        language: str,
        code: str,
        *,
        timeout: int = 30,
        sandbox: str = "default",
    ) -> CodeExecutionResult:
        """Execute code in a sandboxed runtime.

        Args:
            language: "python", "javascript", or "bash"
            code: Source code to execute
            timeout: Max execution time in seconds (default: 30)
            sandbox: "default", "strict", or "permissive"

        Returns:
            CodeExecutionResult with stdout, stderr, exit_code, etc.

        Example::

            result = Agent.execute_code("python", "print(42)")
            assert result.stdout.strip() == "42"
        """
        from gauss._native import execute_code as _exec  # type: ignore[import-not-found]

        result_json = _run_native(_exec, language, code, timeout, None, sandbox)
        data = json.loads(result_json)
        return CodeExecutionResult(
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            exit_code=data.get("exit_code", -1),
            timed_out=data.get("timed_out", False),
            runtime=data.get("runtime", language),
            success=data.get("success", False),
        )

    @staticmethod
    def available_runtimes() -> list[str]:
        """Check which code execution runtimes are available on this system.

        Returns:
            List of runtime names, e.g. ["python", "bash"]

        Example::

            runtimes = Agent.available_runtimes()
            if "python" in runtimes:
                result = Agent.execute_code("python", "print('ok')")
        """
        from gauss._native import available_runtimes as _runtimes  # type: ignore[import-not-found]

        result_json = _run_native(_runtimes)
        return json.loads(result_json)

    @staticmethod
    def generate_image(
        prompt: str,
        *,
        provider: ProviderType | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        size: str | None = None,
        quality: str | None = None,
        style: str | None = None,
        aspect_ratio: str | None = None,
        n: int | None = None,
        response_format: str | None = None,
    ) -> ImageGenerationResult:
        """Generate images using a provider's image generation API.

        Example::

            # OpenAI DALL-E
            result = Agent.generate_image("A sunset over mountains", model="dall-e-3")
            print(result.images[0].url)

            # Gemini
            result = Agent.generate_image("A cat", provider="google", aspect_ratio="16:9")
        """
        from gauss._native import (  # type: ignore[import-not-found]
            create_provider,
            destroy_provider,
            generate_image as _gen_image,
        )

        resolved_provider = provider or detect_provider()
        resolved_model = model or _default_model(resolved_provider)
        resolved_key = api_key or resolve_api_key(resolved_provider)
        prov_opts: dict[str, Any] = {"api_key": resolved_key}
        if base_url:
            prov_opts["base_url"] = base_url
        handle = _run_native(create_provider, resolved_provider.value, resolved_model, json.dumps(prov_opts))
        try:
            result_json = _run_native(
                _gen_image,
                handle,
                prompt,
                model,
                size,
                quality,
                style,
                aspect_ratio,
                n,
                response_format,
            )
            data = json.loads(result_json)
            images = [
                GeneratedImageData(
                    url=img.get("url"),
                    base64=img.get("base64"),
                    mime_type=img.get("mime_type") or img.get("mimeType"),
                )
                for img in data.get("images", [])
            ]
            return ImageGenerationResult(
                images=images,
                revised_prompt=data.get("revised_prompt") or data.get("revisedPrompt"),
            )
        finally:
            _run_native(destroy_provider, handle)

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
        answer = gauss("Write a poem", temperature=0.9, model=OPENAI_DEFAULT)
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
    def from_json(cls, json_str: str | dict[str, Any]) -> StreamEvent:
        """Parse a JSON string or dict into a StreamEvent."""
        if isinstance(json_str, dict):
            data = json_str
        else:
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                return cls(type="raw", text=json_str)

        event_type = data.get("type", "unknown")
        event_data = data.get("data")

        text = None
        tool_call = None
        if event_type == "text_delta" and isinstance(event_data, str):
            text = event_data
        elif event_type == "tool_call_delta" and isinstance(event_data, dict):
            tool_call = event_data
        elif isinstance(event_data, str):
            text = event_data

        return cls(
            type=event_type,
            text=text,
            tool_call=tool_call,
            raw={k: v for k, v in data.items() if k not in ("type",)},
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
        import inspect

        messages_json = json.dumps(self._messages)
        result = stream_generate(
            self._provider_handle,
            messages_json,
            self._temperature,
            self._max_tokens,
        )
        if inspect.isawaitable(result):
            result_json = await result
        else:
            result_json = result

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
