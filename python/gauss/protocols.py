"""Protocol types for structural subtyping in the Gauss SDK."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ToolCallback(Protocol):
    """Protocol for synchronous tool callback functions."""

    def __call__(self, **kwargs: Any) -> Any: ...


@runtime_checkable
class ToolExecutor(Protocol):
    """Protocol for asynchronous tool executors."""

    async def execute(self, tool_name: str, args: dict[str, Any]) -> str: ...
