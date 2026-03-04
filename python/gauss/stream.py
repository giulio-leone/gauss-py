"""Streaming utilities — AgentStream, StreamEvent, and partial JSON parsing."""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


def parse_partial_json(text: str) -> Any:
    """Parse incomplete/streaming JSON into a valid Python object.

    Example::

        obj = parse_partial_json('{"name": "Jo')
        # Returns: {"name": "Jo"}
    """
    from gauss._native import py_parse_partial_json

    result_json: str = py_parse_partial_json(text)
    return json.loads(result_json)


@dataclass
class StreamEvent:
    """A single event emitted during a streaming response.

    Events are produced incrementally as the model generates its reply.
    The ``type`` field determines which payload fields are populated.

    Attributes:
        type: Event kind — ``"text_delta"``, ``"tool_call_delta"``,
            ``"raw"``, or other provider-specific types.
        text: The text fragment (populated for ``text_delta`` events).
        tool_call: Tool-call payload dict (populated for
            ``tool_call_delta`` events).
        raw: Full raw event data for introspection or custom handling.

    Example:
        >>> event = StreamEvent(type="text_delta", text="Hello")
        >>> event.text
        'Hello'
    """

    type: str
    text: str | None = None
    tool_call: dict[str, str] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, json_str: str | dict[str, Any]) -> StreamEvent:
        """Parse a JSON string or dict into a ``StreamEvent``.

        Args:
            json_str: A raw JSON string or an already-parsed dict
                representing a single streaming event.

        Returns:
            A :class:`StreamEvent` with ``type``, ``text``, and/or
            ``tool_call`` fields populated based on the event data.
        """
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
    """Async-iterable wrapper over a native streaming response.

    Yields :class:`StreamEvent` objects one at a time.  After the
    iteration completes, the full aggregated text is available via the
    :attr:`text` property.

    Use :meth:`Agent.stream_iter` to obtain an instance — do not
    construct directly.

    Example:
        >>> stream = agent.stream_iter("Tell me a story")
        >>> async for event in stream:
        ...     if event.type == "text_delta":
        ...         print(event.text, end="", flush=True)
        >>> print(stream.text)
    """

    def __init__(
        self,
        *,
        provider_handle: int,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialise the stream (internal — prefer ``Agent.stream_iter``).

        Args:
            provider_handle: Opaque native provider handle.
            messages: Normalised message list.
            temperature: Sampling temperature override.
            max_tokens: Maximum tokens override.
        """
        self._provider_handle = provider_handle
        self._messages = messages
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._events: list[StreamEvent] = []
        self._text: str | None = None

    @property
    def text(self) -> str | None:
        """Full aggregated text from all ``text_delta`` events.

        Returns:
            The concatenated text after iteration finishes, or ``None``
            if iteration has not yet completed.
        """
        return self._text

    @property
    def events(self) -> list[StreamEvent]:
        """Return a copy of all events yielded so far during iteration."""
        return list(self._events)

    async def __aiter__(self) -> AsyncIterator[StreamEvent]:
        """Asynchronously iterate over streaming events.

        Yields:
            :class:`StreamEvent` objects in the order received from the
            provider.  After the iterator is exhausted, :attr:`text`
            contains the full concatenated response.
        """
        from gauss._native import stream_generate

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
