"""
Typed Tool System — define tools with inline execute callbacks.

Quick start::

    from gauss import tool, Agent

    @tool
    def get_weather(city: str) -> dict:
        \"\"\"Get current weather for a city.\"\"\"
        return {"temp": 72, "unit": "F", "city": city}

    agent = Agent()
    agent.add_tools([get_weather])
    result = agent.run("What's the weather in Paris?")

The @tool decorator creates a TypedToolDef that the agent auto-wires
into a tool executor when it detects typed tools with execute callbacks.

.. versionadded:: 1.2.0
"""

from __future__ import annotations

import inspect
import json
import logging
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from gauss._types import ToolDef


@dataclass
class TypedToolDef(ToolDef):
    """A tool definition with a typed execute callback.

    Extends :class:`ToolDef` with an ``execute`` callable that is
    automatically invoked when the LLM requests this tool.
    """

    execute: Callable[..., Any] | None = field(default=None, repr=False)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Allow calling the typed tool directly."""
        if self.execute is None:
            raise RuntimeError(f"Tool {self.name!r} has no execute callback")
        return self.execute(*args, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for NAPI (no execute callback)."""
        d: dict[str, Any] = {"name": self.name, "description": self.description}
        if self.parameters:
            d["parameters"] = self.parameters
        return d


def _extract_parameters(func: Callable[..., Any]) -> dict[str, Any]:
    """Extract JSON-schema-style parameters from a function signature."""
    sig = inspect.signature(func)
    props: dict[str, Any] = {}
    required: list[str] = []

    type_map: dict[Any, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    # Also map string representations for __future__ annotations
    str_type_map: dict[str, str] = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
    }

    # Try to resolve string annotations to real types
    try:
        hints = typing.get_type_hints(func)
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).debug("Could not resolve type hints for %s", func.__name__)
        hints = {}

    for name, param in sig.parameters.items():
        prop: dict[str, Any] = {}

        # Prefer resolved type hints over raw annotation
        annotation = hints.get(name, param.annotation)

        if annotation != inspect.Parameter.empty:
            if isinstance(annotation, str):
                json_type = str_type_map.get(annotation, "string")
            else:
                json_type = type_map.get(annotation, "string")
            prop["type"] = json_type
        else:
            prop["type"] = "string"

        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = param.default

        props[name] = prop

    schema: dict[str, Any] = {
        "type": "object",
        "properties": props,
    }
    if required:
        schema["required"] = required

    return schema


def _extract_description(func: Callable[..., Any]) -> str:
    """Extract description from function docstring."""
    doc = inspect.getdoc(func)
    if doc:
        return doc.split("\n\n")[0].strip()
    return ""


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> TypedToolDef | Callable[..., TypedToolDef]:
    """Create a typed tool from a function.

    Can be used as a decorator (with or without arguments) or as a
    function call.

    Examples::

        # As a bare decorator
        @tool
        def search(query: str) -> dict:
            \"\"\"Search the web.\"\"\"
            return {"results": [...]}

        # As a decorator with arguments
        @tool(name="web_search", description="Search the web for information")
        def search(query: str) -> dict:
            return {"results": [...]}

        # As a function call
        search_tool = tool(
            lambda query: {"results": [...]},
            name="search",
            description="Search the web",
        )

    Args:
        func: The function to wrap. When ``None``, returns a decorator.
        name: Tool name (default: function name).
        description: Tool description (default: extracted from docstring).
        parameters: JSON schema for parameters (default: extracted from
            function signature).

    Returns:
        A :class:`TypedToolDef` when called with a function, or a
        decorator when called without.

    .. versionadded:: 1.2.0
    """

    def _wrap(fn: Callable[..., Any]) -> TypedToolDef:
        tool_name = name or fn.__name__
        tool_desc = description or _extract_description(fn)
        tool_params = parameters or _extract_parameters(fn)

        return TypedToolDef(
            name=tool_name,
            description=tool_desc,
            parameters=tool_params,
            execute=fn,
        )

    # Bare decorator: @tool
    if func is not None:
        return _wrap(func)

    # Decorator with arguments: @tool(name="...", ...)
    return _wrap


def create_tool_executor(
    tools: list[TypedToolDef],
    fallback: Callable[[str], str] | None = None,
) -> Callable[[str], str]:
    """Build a tool executor from a list of typed tools.

    Args:
        tools: List of typed tool definitions with execute callbacks.
        fallback: Optional fallback executor for unmatched tool names.

    Returns:
        A callable that accepts a JSON tool-call string and returns
        a JSON result string.

    .. versionadded:: 1.2.0
    """
    tool_map = {t.name: t for t in tools if t.execute is not None}

    def executor(call_json: str) -> str:
        try:
            call = json.loads(call_json)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid tool call JSON"})

        tool_name = call.get("tool") or call.get("name", "")
        tool_def = tool_map.get(tool_name)

        if tool_def is None or tool_def.execute is None:
            if fallback is not None:
                return fallback(call_json)
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            args = call.get("args") or call.get("arguments") or {}
            result = tool_def.execute(**args) if isinstance(args, dict) else tool_def.execute(args)
            if isinstance(result, str):
                return result
            return json.dumps(result)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).debug("Tool execution error for %s: %s", tool_name, exc)
            return json.dumps({"error": str(exc)})

    return executor
