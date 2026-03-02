"""Tool Registry — searchable tool registry with tags, examples, and batch execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolExample:
    """An example of how to use a tool."""

    description: str
    input: Any
    expected_output: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"description": self.description, "input": self.input}
        if self.expected_output is not None:
            d["expectedOutput"] = self.expected_output
        return d


@dataclass(frozen=True)
class ToolRegistryEntry:
    """A tool entry for the registry."""

    name: str
    description: str
    tags: list[str] = field(default_factory=list)
    examples: list[ToolExample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
        }
        if self.tags:
            d["tags"] = self.tags
        if self.examples:
            d["examples"] = [e.to_dict() for e in self.examples]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolRegistryEntry:
        examples = [
            ToolExample(
                description=e["description"],
                input=e.get("input"),
                expected_output=e.get("expectedOutput"),
            )
            for e in d.get("examples", [])
        ]
        return cls(
            name=d["name"],
            description=d["description"],
            tags=d.get("tags", []),
            examples=examples,
        )


@dataclass(frozen=True)
class ToolSearchResult:
    """Result of a tool search."""

    name: str
    description: str
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolSearchResult:
        return cls(
            name=d["name"],
            description=d["description"],
            tags=d.get("tags", []),
        )


class ToolRegistry:
    """Searchable tool registry with tags, examples, and batch support.

    Example::

        registry = ToolRegistry()
        registry.add(ToolRegistryEntry(name="calc", description="Calculator", tags=["math"]))
        results = registry.search("math")
    """

    def __init__(self) -> None:
        from gauss._native import create_tool_registry

        self._handle: int = create_tool_registry()
        self._destroyed = False

    def add(self, entry: ToolRegistryEntry | dict[str, Any]) -> ToolRegistry:
        """Register a tool. Returns self for chaining."""
        from gauss._native import tool_registry_add

        self._check_alive()
        entry_dict = entry.to_dict() if isinstance(entry, ToolRegistryEntry) else entry
        tool_registry_add(self._handle, json.dumps(entry_dict))
        return self

    def search(self, query: str) -> list[ToolSearchResult]:
        """Search tools by query (matches name, description, tags)."""
        from gauss._native import tool_registry_search

        self._check_alive()
        results_json = tool_registry_search(self._handle, query)
        return [ToolSearchResult.from_dict(d) for d in json.loads(results_json)]

    def by_tag(self, tag: str) -> list[ToolSearchResult]:
        """Find tools by tag."""
        from gauss._native import tool_registry_by_tag

        self._check_alive()
        results_json = tool_registry_by_tag(self._handle, tag)
        return [ToolSearchResult.from_dict(d) for d in json.loads(results_json)]

    def list(self) -> list[ToolRegistryEntry]:
        """List all registered tools."""
        from gauss._native import tool_registry_list

        self._check_alive()
        results_json = tool_registry_list(self._handle)
        return [ToolRegistryEntry.from_dict(d) for d in json.loads(results_json)]

    def destroy(self) -> None:
        if not self._destroyed:
            from gauss._native import destroy_tool_registry

            destroy_tool_registry(self._handle)
            self._destroyed = True

    def __enter__(self) -> ToolRegistry:
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    def __del__(self) -> None:
        self.destroy()

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("ToolRegistry has been destroyed")
