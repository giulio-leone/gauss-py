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
from typing import TYPE_CHECKING, Any

from gauss._types import (
    AgentConfig,
    AgentResult,
    Citation,
    CodeExecutionOptions,
    GroundingChunk,
    GroundingMetadata,
    Message,
    ProviderCapabilities,
    ProviderType,
    ToolDef,
    detect_provider,
    resolve_api_key,
)
from gauss.models import OPENAI_DEFAULT
from gauss.stream import AgentStream

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
        """Initialize the Agent with an optional configuration.

        If no ``config`` is provided, keyword arguments are forwarded to
        :class:`AgentConfig`.  When neither is given, the provider, model,
        and API key are auto-detected from environment variables.

        Args:
            config: Explicit agent configuration.  When ``None``, an
                :class:`AgentConfig` is built from ``**kwargs``.
            **kwargs: Forwarded to :class:`AgentConfig` when *config* is
                ``None``.  Common keys: ``provider``, ``model``,
                ``api_key``, ``system_prompt``, ``temperature``.

        Raises:
            OSError: If no API key can be resolved from the environment.

        Example:
            >>> agent = Agent()
            >>> agent = Agent(model="gpt-4o", temperature=0.5)
        """
        from gauss._native import (
            create_provider,
            destroy_provider,
        )

        self._config = config or AgentConfig(**kwargs)
        self._tools: list[ToolDef] = list(self._config.tools)
        self._destroyed = False
        self._destroy_provider = destroy_provider
        self._provider_handle: int | None = None

        provider_type, model, api_key = self._config.resolve()

        self._provider_handle = create_provider(
            provider_type.value,
            model,
            api_key,
            self._config.base_url,
            self._config.max_retries,
        )
        self._model = model

    # ── Execution ──────────────────────────────────────────────────────

    def run(self, prompt: str | Sequence[Message | dict[str, str]]) -> AgentResult:
        """Run the agent synchronously and return the full result.

        Sends the prompt (or message history) to the configured LLM provider
        and returns a structured result including the generated text, token
        usage, and any tool calls requested by the model.

        Args:
            prompt: A plain string (auto-wrapped as a ``user`` message) or
                a sequence of :class:`Message` / raw ``dict`` objects
                representing a conversation history.

        Returns:
            :class:`AgentResult` containing ``text``, ``messages``,
            ``tool_calls``, ``usage``, and optional ``thinking`` /
            ``citations`` fields.

        Raises:
            RuntimeError: If the agent has already been destroyed.

        Example:
            >>> result = agent.run("What is 2+2?")
            >>> print(result.text)
        """
        from gauss._native import agent_run

        self._check_alive()
        messages = self._normalize_messages(prompt)
        messages_json = json.dumps(messages)
        options = self._build_options()

        result_json = _run_native(
            agent_run,
            self._config.name,
            self._provider_handle,
            messages_json,
            json.dumps(options) if options else None,
        )

        return self._parse_result(json.loads(result_json))

    def generate(self, prompt: str | Sequence[Message | dict[str, str]]) -> str:
        """Generate text and return only the response string.

        A convenience wrapper around :meth:`run` that discards metadata
        and returns just the generated text.

        Args:
            prompt: A plain string or sequence of messages.

        Returns:
            The generated text as a plain ``str``.

        Raises:
            RuntimeError: If the agent has already been destroyed.

        Example:
            >>> text = agent.generate("Write a haiku about Rust")
        """
        from gauss._native import generate

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
        """Query the feature capabilities of the current provider and model.

        Returns:
            :class:`ProviderCapabilities` with boolean flags for
            ``streaming``, ``tool_use``, ``vision``, ``extended_thinking``,
            ``image_generation``, and more.

        Raises:
            RuntimeError: If the agent has already been destroyed.

        Example:
            >>> caps = agent.capabilities
            >>> if caps.streaming:
            ...     print("Streaming supported")
        """
        from gauss._native import get_provider_capabilities

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

    def run_with_tools(
        self,
        prompt: str | Sequence[Message | dict[str, str]],
        tool_executor: Any,
    ) -> AgentResult:
        """Run the agent with a custom tool executor callback.

        When the model requests a tool call, the ``tool_executor`` is
        invoked with a JSON string ``{"tool": "<name>", "args": {...}}``
        and must return a JSON string with the tool result.  Supports
        both sync and async callables.

        Args:
            prompt: A plain string or sequence of messages.
            tool_executor: A callable ``(call_json: str) -> str`` or
                ``async (call_json: str) -> str`` that handles tool calls.

        Returns:
            :class:`AgentResult` with the final response.

        Raises:
            RuntimeError: If the agent has been destroyed.

        Example:
            >>> def handle_tool(call_json: str) -> str:
            ...     data = json.loads(call_json)
            ...     if data["tool"] == "get_weather":
            ...         return json.dumps({"temp": 72, "unit": "F"})
            ...     return json.dumps({"error": "unknown tool"})
            >>> result = agent.run_with_tools("What's the weather?", handle_tool)
        """
        from gauss._native import agent_run_with_tool_executor

        self._check_alive()
        messages = self._normalize_messages(prompt)
        messages_json = json.dumps(messages)
        options = self._build_options()

        result_json = _run_native(
            agent_run_with_tool_executor,
            self._config.name,
            self._provider_handle,
            messages_json,
            json.dumps(options) if options else None,
            tool_executor,
        )

        return self._parse_result(json.loads(result_json))

    def stream(
        self,
        prompt: str | Sequence[Message | dict[str, str]],
        callback: Any,
    ) -> AgentResult:
        """Stream agent execution, pushing events to a callback.

        Each event is delivered as a JSON string to the ``callback``.
        Event types include ``text_delta``, ``step_start``,
        ``step_finish``, ``tool_result``, ``done``, and ``error``.

        Args:
            prompt: A plain string or sequence of messages.
            callback: A callable ``(event_json: str) -> None`` invoked
                for each streaming event.

        Returns:
            :class:`AgentResult` with the final aggregated result.

        Raises:
            RuntimeError: If the agent has been destroyed.

        Example:
            >>> def on_event(event_json: str):
            ...     event = json.loads(event_json)
            ...     if event["type"] == "text_delta":
            ...         print(event["delta"], end="", flush=True)
            >>> result = agent.stream("Tell me a story", on_event)
        """
        from gauss._native import agent_stream

        self._check_alive()
        messages = self._normalize_messages(prompt)
        messages_json = json.dumps(messages)
        options = self._build_options()

        result_json = _run_native(
            agent_stream,
            self._config.name,
            self._provider_handle,
            messages_json,
            callback,
            json.dumps(options) if options else None,
        )

        data = json.loads(result_json)
        return AgentResult(
            text=data.get("text", ""),
            messages=[],
            tool_calls=[],
            usage=data.get("usage", {}),
        )

    def generate_with_tools(
        self,
        prompt: str | Sequence[Message | dict[str, str]],
        tools: Sequence[ToolDef] | None = None,
    ) -> dict[str, Any]:
        """Single-turn generate with tool definitions (no agent loop).

        Unlike :meth:`run_with_tools`, this does **not** execute tools.
        It returns the model's response including any tool call requests,
        allowing you to handle execution yourself.

        Args:
            prompt: A plain string or sequence of messages.
            tools: Tool definitions to provide to the model.  If ``None``,
                uses the agent's registered tools.

        Returns:
            A dict with ``text``, ``tool_calls``, ``usage``, and
            ``finish_reason`` keys.

        Example:
            >>> result = agent.generate_with_tools("Search for Python docs")
            >>> for call in result["tool_calls"]:
            ...     print(f"Tool: {call['name']}, Args: {call['args']}")
        """
        from gauss._native import generate_with_tools

        self._check_alive()
        messages = self._normalize_messages(prompt)
        effective_tools = list(tools) if tools else self._tools
        tools_json = json.dumps([t.to_dict() for t in effective_tools])

        result_json = _run_native(
            generate_with_tools,
            self._provider_handle,
            json.dumps(messages),
            tools_json,
            self._config.temperature,
            self._config.max_tokens,
        )

        return json.loads(result_json)

    def stream_iter(self, prompt: str | Sequence[Message | dict[str, str]]) -> AgentStream:
        """Return an async-iterable stream of events for the given prompt.

        Each yielded :class:`StreamEvent` represents an incremental chunk
        of the model response (text deltas, tool calls, etc.).  After
        iteration completes, the full aggregated text is available via
        ``stream.text``.

        Args:
            prompt: A plain string or sequence of messages.

        Returns:
            An :class:`AgentStream` that can be used with ``async for``.

        Raises:
            RuntimeError: If the agent has already been destroyed.

        Example:
            >>> async for event in agent.stream_iter("Tell me a story"):
            ...     if event.type == "text_delta":
            ...         print(event.text, end="", flush=True)
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
        """Register a single tool definition with the agent.

        Args:
            tool: A :class:`ToolDef` describing the tool's name,
                description, and JSON-Schema parameters.

        Returns:
            The same :class:`Agent` instance, allowing method chaining.

        Example:
            >>> agent.add_tool(ToolDef(name="get_weather", description="..."))
        """
        self._tools.append(tool)
        return self

    def add_tools(self, tools: Sequence[ToolDef]) -> Agent:
        """Register multiple tool definitions at once.

        Args:
            tools: A sequence of :class:`ToolDef` objects.

        Returns:
            The same :class:`Agent` instance, allowing method chaining.

        Example:
            >>> agent.add_tools([weather_tool, search_tool])
        """
        self._tools.extend(tools)
        return self

    def set_options(self, **kwargs: Any) -> Agent:
        """Update runtime configuration options.

        Merges the provided keyword arguments into the agent's existing
        configuration, allowing you to tweak behaviour between runs
        without recreating the agent.

        Args:
            **kwargs: Configuration keys to update.  Accepted keys include
                ``temperature``, ``max_tokens``, ``system_prompt``,
                ``thinking_budget``, ``reasoning_effort``,
                ``cache_control``, ``code_execution``, ``grounding``,
                ``stop_condition``, and any other :class:`AgentConfig` field.

        Returns:
            The same :class:`Agent` instance, allowing method chaining.

        Example:
            >>> agent.set_options(temperature=0.9, max_tokens=2048)
        """
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
        return self

    # ── Lifecycle ──────────────────────────────────────────────────────

    def destroy(self) -> None:
        """Release native (Rust) resources held by this agent.

        Safe to call multiple times; subsequent calls are no-ops.
        Automatically invoked when the agent is used as a context manager
        or garbage-collected.
        """
        if not self._destroyed and self._provider_handle is not None:
            self._destroy_provider(self._provider_handle)
        self._destroyed = True

    def __enter__(self) -> Agent:
        """Enter the context manager, returning this agent instance."""
        return self

    def __exit__(self, *_: Any) -> None:
        """Exit the context manager and release native resources."""
        self.destroy()

    def __del__(self) -> None:
        self.destroy()

    # ── Internal ──────────────────────────────────────────────────────

    def _check_alive(self) -> None:
        if self._destroyed or self._provider_handle is None:
            raise RuntimeError("Agent has been destroyed")

    def _build_options(self) -> dict[str, Any]:
        """Build the options dict from current config (DRY helper)."""
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
        return options

    @staticmethod
    def _parse_result(data: dict[str, Any]) -> AgentResult:
        """Parse a JSON result dict into an AgentResult (DRY helper)."""
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

    @property
    def handle(self) -> int:
        """Return the native provider handle for advanced / low-level use."""
        self._check_alive()
        return int(self._provider_handle)

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
    """One-liner AI call — send a prompt and get text back.

    Creates a temporary :class:`Agent`, runs the prompt, and returns
    the generated text.  Provider, model, and API key are auto-detected
    from environment variables unless overridden via ``**kwargs``.

    Args:
        prompt: The user prompt string.
        **kwargs: Forwarded to :class:`AgentConfig` (e.g. ``model``,
            ``temperature``, ``provider``).

    Returns:
        The generated text as a plain ``str``.

    Example:
        >>> from gauss import gauss
        >>> answer = gauss("What is the meaning of life?")
        >>> print(answer)
    """
    with Agent(AgentConfig(**kwargs)) as agent:
        return agent.run(prompt).text


def enterprise_preset(config: AgentConfig | None = None, **kwargs: Any) -> Agent:
    """Create an enterprise-ready Agent with production-safe defaults."""
    cfg = config or AgentConfig(**kwargs)
    for key, value in kwargs.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)

    if isinstance(cfg.provider, str):
        cfg.provider = ProviderType(cfg.provider.lower())

    if not cfg.name or cfg.name == "gauss-agent":
        cfg.name = "enterprise-agent"
    cfg.max_retries = max(cfg.max_retries, 5)
    if cfg.temperature is None:
        cfg.temperature = 0.2
    if not cfg.cache_control:
        cfg.cache_control = True
    if cfg.reasoning_effort is None:
        cfg.reasoning_effort = "medium"

    return Agent(cfg)
