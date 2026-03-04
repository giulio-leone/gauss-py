"""Decorator-based agent, tool, and guardrail creation.

Provides a high-level decorator API for defining Gauss tools, agents,
and guardrails with minimal boilerplate.  Type hints on the decorated
function are automatically converted to JSON Schema for tool parameters.

Quick start::

    from gauss.decorators import tool, agent, guardrail

    @tool
    def get_weather(city: str, unit: str = 'celsius') -> str:
        '''Get current weather for a city.'''
        return f"Weather in {city}: 22°{unit[0].upper()}"

    @agent(model='gpt-4o', tools=[get_weather])
    def weather_assistant(prompt: str) -> str:
        '''A helpful weather assistant.'''
        ...

    @guardrail
    def no_pii(text: str) -> bool:
        '''Block PII in output.'''
        return not any(p in text.lower() for p in ['ssn', 'credit card'])

.. versionadded:: 2.5.0
"""

from __future__ import annotations

import functools
import inspect
import typing
from collections.abc import Callable
from typing import Any, get_type_hints

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=64)
def _python_type_to_json(python_type: Any) -> str:
    """Convert a Python type annotation to a JSON Schema type string."""
    origin = typing.get_origin(python_type)
    if origin is not None:
        python_type = origin
    mapping: dict[type, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    return mapping.get(python_type, "string")


def _build_parameter_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON Schema ``object`` from *fn*'s signature and type hints."""
    try:
        hints = get_type_hints(fn)
    except Exception:  # noqa: BLE001
        hints = {}

    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        param_type = hints.get(param_name, str)
        json_type = _python_type_to_json(param_type)
        properties[param_name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    return schema


# ---------------------------------------------------------------------------
# @tool
# ---------------------------------------------------------------------------

def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Decorator to create a Gauss tool from a function.

    Can be used bare (``@tool``) or with keyword arguments
    (``@tool(name="custom")``).  The decorated function remains callable
    and gains a ``.tool_def`` dict and an ``.is_gauss_tool`` flag.

    Example::

        @tool
        def get_weather(city: str, unit: str = 'celsius') -> str:
            '''Get current weather for a city.'''
            return f"Weather in {city}: 22°{unit[0].upper()}"

        get_weather.tool_def  # → dict with name, description, parameters, execute
        get_weather('Rome')   # → "Weather in Rome: 22°C"
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        schema = _build_parameter_schema(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        wrapper.tool_def = {  # type: ignore[attr-defined]
            "name": name or fn.__name__,
            "description": description or (fn.__doc__ or "").strip(),
            "parameters": schema,
            "execute": fn,
        }
        wrapper.is_gauss_tool = True  # type: ignore[attr-defined]
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# @agent
# ---------------------------------------------------------------------------

def agent(
    func: Callable[..., Any] | None = None,
    *,
    model: str = "gpt-4o",
    instructions: str | None = None,
    tools: list[Any] | None = None,
    max_steps: int = 10,
) -> Any:
    """Decorator to create a Gauss agent from a function.

    The decorated function becomes the agent's entry-point.  Calling it
    triggers an agent run with the configured *model*, *instructions*,
    and *tools*.

    Example::

        @agent(model='gpt-4o', tools=[get_weather])
        def weather_assistant(prompt: str) -> str:
            '''A helpful weather assistant.'''
            ...

        weather_assistant('What is the weather in Rome?')
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        resolved_tools = tools or []
        resolved_instructions = instructions or (fn.__doc__ or "").strip()

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        wrapper.agent_def = {  # type: ignore[attr-defined]
            "name": fn.__name__,
            "model": model,
            "instructions": resolved_instructions,
            "tools": resolved_tools,
            "max_steps": max_steps,
            "execute": fn,
        }
        wrapper.is_gauss_agent = True  # type: ignore[attr-defined]
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# @guardrail
# ---------------------------------------------------------------------------

def guardrail(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
) -> Any:
    """Decorator to create a guardrail from a function.

    The function should accept a ``str`` and return ``bool`` (``True``
    means the text passes the guardrail).

    Example::

        @guardrail
        def no_pii(text: str) -> bool:
            '''Block PII in output.'''
            return not any(p in text.lower() for p in ['ssn', 'credit card'])

        no_pii('My SSN is ...')  # → False
        no_pii.guardrail_def    # → dict with name, description, execute
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        wrapper.guardrail_def = {  # type: ignore[attr-defined]
            "name": name or fn.__name__,
            "description": (fn.__doc__ or "").strip(),
            "execute": fn,
        }
        wrapper.is_gauss_guardrail = True  # type: ignore[attr-defined]
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
