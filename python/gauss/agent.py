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
from collections.abc import Callable
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
)
from gauss.base import StatefulResource
from gauss.errors import DisposedError
from gauss.routing_policy import RoutingPolicy, resolve_routing_target
from gauss.stream import AgentStream, StreamEvent
from gauss.tool import TypedToolDef, create_tool_executor

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from gauss.guardrail import GuardrailChain
    from gauss.mcp_client import McpClient
    from gauss.memory import Memory
    from gauss.middleware import MiddlewareChain


from gauss._utils import _run_native  # noqa: F401 — re-exported for backward compat


class Agent(StatefulResource):
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

        super().__init__()
        self._config = config or AgentConfig(**kwargs)
        self._tools: list[ToolDef | TypedToolDef] = list(self._config.tools)
        self._destroy_provider = destroy_provider
        self._provider_handle: int | None = None

        # Integration glue (M35)
        self._middleware: MiddlewareChain | None = None
        self._guardrails: GuardrailChain | None = None
        self._memory: Memory | None = None
        self._session_id: str = ""
        self._mcp_clients: list[McpClient] = []
        self._mcp_tools_loaded = False

        # Cost & tracing (M90)
        self._last_cost: dict[str, Any] | None = None
        self._last_trace: dict[str, Any] | None = None

        provider_type, model, api_key = self._config.resolve()
        provider_type, model = resolve_routing_target(
            self._config.routing_policy,
            provider_type,
            model,
        )

        self._provider_handle = create_provider(
            provider_type.value,
            model,
            api_key,
            self._config.base_url,
            self._config.max_retries,
        )
        self._model = model

    @classmethod
    def from_env(cls, **kwargs: Any) -> Agent:
        """Create an agent using environment auto-detection with optional overrides.

        This helper is equivalent to ``Agent(**kwargs)`` and is provided for
        readability in quick-start flows.

        Args:
            **kwargs: Forwarded to :class:`AgentConfig`.

        Returns:
            A new :class:`Agent` instance.

        .. versionadded:: 2.1.0
        """
        return cls(**kwargs)

    @classmethod
    def quick(
        cls,
        model: str,
        instructions: str,
        *,
        tools: list[Any] | None = None,
        max_steps: int = 10,
    ) -> Agent:
        """Create an agent with minimal configuration.

        A convenience factory that sets the model and system prompt in a
        single call.  Additional tools and a step limit can be provided.

        Args:
            model: Model identifier (e.g. ``"gpt-4o"``).
            instructions: System prompt / instructions for the agent.
            tools: Optional list of tool definitions.
            max_steps: Maximum agentic loop iterations (default ``10``).

        Returns:
            A new :class:`Agent` instance.

        Example:
            >>> agent = Agent.quick('gpt-4o', 'You are helpful')

        .. versionadded:: 2.5.0
        """
        config_kwargs: dict[str, Any] = {
            "model": model,
            "system_prompt": instructions,
            "max_retries": max_steps,
        }
        if tools:
            config_kwargs["tools"] = tools
        return cls(**config_kwargs)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Agent:
        """Create an agent from a plain configuration dict.

        Keys are forwarded to :class:`AgentConfig`.

        Args:
            config: A mapping of configuration keys accepted by
                :class:`AgentConfig`.

        Returns:
            A new :class:`Agent` instance.

        Example:
            >>> agent = Agent.from_config({"model": "gpt-4o", "system_prompt": "Hi"})

        .. versionadded:: 2.5.0
        """
        return cls(**config)

    # ── Execution ──────────────────────────────────────────────────────

    def run(self, prompt: str | Sequence[Message | dict[str, str]]) -> AgentResult:
        """Run the agent synchronously and return the full result.

        Sends the prompt (or message history) to the configured LLM provider
        and returns a structured result including the generated text, token
        usage, and any tool calls requested by the model.

        When memory is attached, previous context is recalled before the
        run and the conversation is stored after.

        When typed tools (with execute callbacks) are registered, a tool
        executor is automatically wired.

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
        self._check_alive()
        self._ensure_mcp_tools()

        messages = self._normalize_messages(prompt)

        # Memory recall: inject context
        if self._memory is not None:
            recalled = self._memory.recall(
                session_id=self._session_id if self._session_id else "default"
            )
            if recalled:
                context_text = "\n".join(
                    e.get("content", "") if isinstance(e, dict) else str(e)
                    for e in recalled
                )
                messages = [
                    {"role": "system", "content": f"Previous context:\n{context_text}"},
                    *messages,
                ]

        # Resolve typed tools
        tool_defs, executor = self._resolve_tools_and_executor()

        messages_json = json.dumps(messages)
        options = self._build_options(tool_defs=tool_defs)

        if executor:
            from gauss._native import agent_run_with_tool_executor

            result_json = _run_native(
                agent_run_with_tool_executor,
                self._config.name,
                self._provider_handle,
                messages_json,
                json.dumps(options) if options else None,
                executor,
            )
        else:
            from gauss._native import agent_run

            result_json = _run_native(
                agent_run,
                self._config.name,
                self._provider_handle,
                messages_json,
                json.dumps(options) if options else None,
            )

        result = self._parse_result(json.loads(result_json))

        # Populate cost & trace from usage (M90)
        usage = result.usage or {}
        if usage:
            self._last_cost = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_usd": usage.get("total_usd"),
            }
            self._last_trace = {
                "model": self._model,
                "usage": usage,
            }

        # Memory store: save conversation
        if self._memory is not None:
            user_text = prompt if isinstance(prompt, str) else " ".join(
                m.content if isinstance(m, Message) else m.get("content", "")
                for m in prompt
            )
            self._memory.store_sync(
                content=user_text,
                entry_type="conversation",
                session_id=self._session_id or None,
            )
            self._memory.store_sync(
                content=result.text,
                entry_type="conversation",
                session_id=self._session_id or None,
            )

        return result

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

    @property
    def last_run_cost(self) -> dict[str, Any] | None:
        """Cost estimate from the last run.

        Returns a dict with keys like ``input_tokens``, ``output_tokens``,
        and ``total_usd``, or ``None`` if no run has been executed yet.

        .. versionadded:: 2.5.0
        """
        return self._last_cost

    @property
    def last_run_trace(self) -> dict[str, Any] | None:
        """Trace / span data from the last run.

        Returns a dict with trace metadata (spans, timings, step info),
        or ``None`` if no run has been executed yet.

        .. versionadded:: 2.5.0
        """
        return self._last_trace

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

    def stream_text(
        self,
        prompt: str | Sequence[Message | dict[str, str]],
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        """Stream text deltas with a minimal helper and return final text.

        Args:
            prompt: A plain string or sequence of messages.
            on_delta: Optional callback invoked for each text delta chunk.

        Returns:
            The final aggregated text response.

        .. versionadded:: 2.1.0
        """
        chunks: list[str] = []

        def _on_event(event_json: str) -> None:
            try:
                event = json.loads(event_json)
            except json.JSONDecodeError:
                return
            if event.get("type") != "text_delta":
                return

            delta = event.get("text")
            if not isinstance(delta, str):
                delta = event.get("delta")
            if not isinstance(delta, str):
                delta = event.get("data")
            if not isinstance(delta, str):
                return

            chunks.append(delta)
            if on_delta is not None:
                on_delta(delta)

        result = self.stream(prompt, _on_event)
        return result.text or "".join(chunks)

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

    def add_tool(self, tool: ToolDef | TypedToolDef) -> Agent:
        """Register a single tool definition with the agent.

        Accepts both raw :class:`ToolDef` and typed :class:`TypedToolDef`
        (created with the ``@tool`` decorator). Typed tools with execute
        callbacks are automatically wired into a tool executor.

        Args:
            tool: A :class:`ToolDef` or :class:`TypedToolDef`.

        Returns:
            The same :class:`Agent` instance, allowing method chaining.

        Example:
            >>> agent.add_tool(ToolDef(name="get_weather", description="..."))
        """
        self._tools.append(tool)
        return self

    def add_tools(self, tools: Sequence[ToolDef | TypedToolDef]) -> Agent:
        """Register multiple tool definitions at once.

        Args:
            tools: A sequence of :class:`ToolDef` or :class:`TypedToolDef`.

        Returns:
            The same :class:`Agent` instance, allowing method chaining.

        Example:
            >>> agent.add_tools([weather_tool, search_tool])
        """
        self._tools.extend(tools)
        return self

    def with_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, object] | None = None,
        *,
        execute: Callable[..., str] | None = None,
    ) -> Agent:
        """Define and register an inline tool in one call.

        Convenience method that creates a :class:`TypedToolDef` and adds it
        to the agent, enabling fluent chaining::

            agent.with_tool(
                "add", "Add two numbers",
                {"a": {"type": "number"}, "b": {"type": "number"}},
                execute=lambda a, b: str(a + b),
            ).with_tool(
                "greet", "Greet a person",
                {"name": {"type": "string"}},
                execute=lambda name: f"Hello, {name}!",
            )

        Args:
            name: Tool name.
            description: Human-readable tool description.
            parameters: JSON Schema-style parameter definitions.
            execute: Callback function invoked when the model calls this tool.

        Returns:
            ``self`` for method chaining.

        .. versionadded:: 2.1.0
        """
        from gauss.tool import TypedToolDef as _TypedToolDef

        td = _TypedToolDef(
            name=name,
            description=description,
            parameters=parameters or {},
            execute=execute,
        )
        self._tools.append(td)
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

    def with_model(self, model: str) -> Agent:
        """Clone this agent with a different model while preserving integrations.

        Args:
            model: Target model identifier.

        Returns:
            A new :class:`Agent` instance configured with ``model``.

        .. versionadded:: 2.1.0
        """
        from dataclasses import replace

        self._check_alive()
        provider, _, api_key = self._config.resolve()
        config = replace(
            self._config,
            provider=provider,
            model=model,
            api_key=api_key,
            tools=list(self._tools),
        )
        clone = Agent(config)
        if self._middleware is not None:
            clone.with_middleware(self._middleware)
        if self._guardrails is not None:
            clone.with_guardrails(self._guardrails)
        if self._memory is not None:
            clone.with_memory(self._memory, self._session_id)
        for client in self._mcp_clients:
            clone.use_mcp_server(client)
        return clone

    def with_routing_policy(self, routing_policy: RoutingPolicy | None) -> Agent:
        """Clone this agent with a different routing policy.

        Args:
            routing_policy: Alias/fallback policy used to resolve provider+model targets.

        Returns:
            A new :class:`Agent` instance configured with ``routing_policy``.
        """
        from dataclasses import replace

        self._check_alive()
        config = replace(
            self._config,
            routing_policy=routing_policy,
            tools=list(self._tools),
        )
        clone = Agent(config)
        if self._middleware is not None:
            clone.with_middleware(self._middleware)
        if self._guardrails is not None:
            clone.with_guardrails(self._guardrails)
        if self._memory is not None:
            clone.with_memory(self._memory, self._session_id)
        for client in self._mcp_clients:
            clone.use_mcp_server(client)
        return clone

    def with_routing_context(
        self,
        *,
        available_providers: list[ProviderType] | None = None,
        estimated_cost_usd: float | None = None,
        current_requests_per_minute: int | None = None,
        current_hour_utc: int | None = None,
        governance_tags: list[str] | None = None,
    ) -> Agent:
        """Clone this agent after applying runtime policy-router context.

        Args:
            available_providers: Providers currently healthy/available.
            estimated_cost_usd: Estimated cost for the next request.
            current_requests_per_minute: Current request rate for throttling checks.
            current_hour_utc: UTC hour used for time-window governance checks.
            governance_tags: Governance/compliance tags required by routing policy DSL.

        Returns:
            A new :class:`Agent` instance resolved through routing policy + runtime context.
        """
        from dataclasses import replace

        self._check_alive()
        requested_provider, requested_model, _ = self._config.resolve()
        provider, model = resolve_routing_target(
            self._config.routing_policy,
            requested_provider,
            requested_model,
            available_providers=available_providers,
            estimated_cost_usd=estimated_cost_usd,
            current_requests_per_minute=current_requests_per_minute,
            current_hour_utc=current_hour_utc,
            governance_tags=governance_tags,
        )
        config = replace(
            self._config,
            provider=provider,
            model=model,
            tools=list(self._tools),
        )
        clone = Agent(config)
        if self._middleware is not None:
            clone.with_middleware(self._middleware)
        if self._guardrails is not None:
            clone.with_guardrails(self._guardrails)
        if self._memory is not None:
            clone.with_memory(self._memory, self._session_id)
        for client in self._mcp_clients:
            clone.use_mcp_server(client)
        return clone

    # ── Integration Glue (M35) ────────────────────────────────────────

    def with_middleware(self, chain: MiddlewareChain) -> Agent:
        """Attach a middleware chain (logging, caching, rate limiting).

        Args:
            chain: A configured :class:`MiddlewareChain` instance.

        Returns:
            The same :class:`Agent` instance for method chaining.

        Example:
            >>> from gauss import MiddlewareChain
            >>> agent.with_middleware(
            ...     MiddlewareChain().use_logging().use_caching(60000)
            ... )

        .. versionadded:: 1.2.0
        """
        self._middleware = chain
        return self

    def with_guardrails(self, chain: GuardrailChain) -> Agent:
        """Attach a guardrail chain (content moderation, PII, validation).

        Args:
            chain: A configured :class:`GuardrailChain` instance.

        Returns:
            The same :class:`Agent` instance for method chaining.

        Example:
            >>> from gauss import GuardrailChain
            >>> agent.with_guardrails(
            ...     GuardrailChain().add_pii_detection("redact")
            ... )

        .. versionadded:: 1.2.0
        """
        self._guardrails = chain
        return self

    def with_memory(self, memory: Memory, session_id: str = "") -> Agent:
        """Attach memory for automatic conversation history.

        When memory is attached, the agent automatically recalls recent
        entries before each run and stores the conversation after.

        Args:
            memory: A :class:`Memory` instance.
            session_id: Optional session ID for scoping memory entries.

        Returns:
            The same :class:`Agent` instance for method chaining.

        Example:
            >>> from gauss import Memory
            >>> agent.with_memory(Memory(), session_id="sess-1")

        .. versionadded:: 1.2.0
        """
        self._memory = memory
        self._session_id = session_id
        return self

    # ── Async Execution (M41) ────────────────────────────────────────

    async def arun(self, prompt: str | Sequence[Message | dict[str, str]]) -> AgentResult:
        """Async version of :meth:`run`.

        Runs the agent in a non-blocking way, suitable for ``asyncio``
        event loops. Identical behaviour to :meth:`run` but awaitable.

        Args:
            prompt: A plain string or sequence of messages.

        Returns:
            :class:`AgentResult` with the full response.

        Example:
            >>> result = await agent.arun("What is 2+2?")
            >>> print(result.text)

        .. versionadded:: 2.0.0
        """
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.run, prompt)

    async def agenerate(self, prompt: str | Sequence[Message | dict[str, str]]) -> str:
        """Async version of :meth:`generate`.

        Returns the generated text as a plain string, awaitable.

        Args:
            prompt: A plain string or sequence of messages.

        Returns:
            The generated text as a plain ``str``.

        Example:
            >>> text = await agent.agenerate("Write a haiku")

        .. versionadded:: 2.0.0
        """
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.generate, prompt)

    async def arun_with_tools(
        self,
        prompt: str | Sequence[Message | dict[str, str]],
        tool_executor: Any,
    ) -> AgentResult:
        """Async version of :meth:`run_with_tools`.

        Args:
            prompt: A plain string or sequence of messages.
            tool_executor: A callable handling tool calls.

        Returns:
            :class:`AgentResult` with the final response.

        .. versionadded:: 2.0.0
        """
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.run_with_tools, prompt, tool_executor)

    async def astream(
        self,
        prompt: str | Sequence[Message | dict[str, str]],
    ) -> AsyncIterator[StreamEvent]:
        """Async iterator for streaming agent execution.

        Yields :class:`StreamEvent` objects as the model generates its
        response.  This is the recommended streaming API for ``asyncio``
        code.

        Args:
            prompt: A plain string or sequence of messages.

        Yields:
            :class:`StreamEvent` instances (``text_delta``,
            ``tool_call_delta``, etc.).

        Example:
            >>> async for event in agent.astream("Hello"):
            ...     if event.type == "text_delta":
            ...         print(event.text, end="")

        .. versionadded:: 2.5.0
        """
        stream_obj = self.stream_iter(prompt)
        async for event in stream_obj:
            yield event

    def use_mcp_server(self, client: McpClient) -> Agent:
        """Consume tools from an external MCP server.

        Registers an MCP client whose tools will be loaded and made
        available to the agent on the first run.

        Args:
            client: A connected :class:`McpClient` instance.

        Returns:
            The same :class:`Agent` instance for method chaining.

        Example:
            >>> from gauss import McpClient
            >>> mcp = McpClient(command="npx", args=["-y", "@mcp/server-fs"])
            >>> mcp.connect()
            >>> agent.use_mcp_server(mcp)

        .. versionadded:: 1.2.0
        """
        self._mcp_clients.append(client)
        self._mcp_tools_loaded = False
        return self

    # ── Lifecycle ──────────────────────────────────────────────────────

    @property
    def _resource_name(self) -> str:
        return "Agent"

    def destroy(self) -> None:
        """Release native (Rust) resources held by this agent.

        Safe to call multiple times; subsequent calls are no-ops.
        Automatically invoked when the agent is used as a context manager
        or garbage-collected.
        """
        if not self._destroyed and self._provider_handle is not None:
            self._destroy_provider(self._provider_handle)
        super().destroy()

    # ── Internal ──────────────────────────────────────────────────────

    def _check_alive(self) -> None:
        if self._destroyed or self._provider_handle is None:
            raise DisposedError("Agent", self._config.name or "agent")

    def _build_options(
        self, *, tool_defs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build the options dict from current config (DRY helper)."""
        options: dict[str, Any] = {}
        effective_tools = tool_defs if tool_defs is not None else [
            t.to_dict() for t in self._tools
        ]
        if effective_tools:
            options["tools"] = effective_tools
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

    def _resolve_tools_and_executor(
        self,
    ) -> tuple[list[dict[str, Any]], Any]:
        """Resolve typed tools into plain dicts + an optional executor."""
        typed_tools = [t for t in self._tools if isinstance(t, TypedToolDef) and t.execute is not None]
        tool_defs = [t.to_dict() for t in self._tools]

        executor = create_tool_executor(typed_tools) if typed_tools else None
        return tool_defs, executor

    def _ensure_mcp_tools(self) -> None:
        """Load tools from all connected MCP clients (lazy, once)."""
        if self._mcp_tools_loaded or not self._mcp_clients:
            return

        for client in self._mcp_clients:
            tools, executor = client.get_tools_with_executor()
            for t in tools:
                mcp_tool = TypedToolDef(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters,
                    execute=lambda args, _t=t, _exec=executor: json.loads(
                        _exec(json.dumps({"tool": _t.name, "args": args}))
                    ),
                )
                self._tools.append(mcp_tool)

        self._mcp_tools_loaded = True

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


def enterprise_run(prompt: str, config: AgentConfig | None = None, **kwargs: Any) -> str:
    """One-liner enterprise helper: run prompt with enterprise defaults."""
    with enterprise_preset(config=config, **kwargs) as agent:
        return agent.run(prompt).text
