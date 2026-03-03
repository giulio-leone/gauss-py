"""Tests for Tool Registry SDK."""

import pytest
from gauss.tool_registry import (
    ToolExample,
    ToolRegistry,
    ToolRegistryEntry,
    ToolSearchResult,
)


class TestToolRegistry:
    def test_create_destroy(self) -> None:
        reg = ToolRegistry()
        assert reg._handle >= 0
        reg.destroy()

    def test_context_manager(self) -> None:
        with ToolRegistry() as reg:
            assert reg._handle >= 0
        assert reg._destroyed

    def test_add_chaining(self) -> None:
        reg = ToolRegistry()
        result = reg.add(
            ToolRegistryEntry(name="a", description="Tool A")
        ).add(
            ToolRegistryEntry(name="b", description="Tool B")
        )
        assert result is reg
        reg.destroy()

    def test_add_dict(self) -> None:
        reg = ToolRegistry()
        result = reg.add({"name": "x", "description": "Tool X"})
        assert result is reg
        reg.destroy()

    def test_list(self) -> None:
        with ToolRegistry() as reg:
            reg.add(ToolRegistryEntry(name="calc", description="Calculator", tags=["math"]))
            reg.add(ToolRegistryEntry(name="weather", description="Get weather", tags=["api"]))
            tools = reg.list()
            assert len(tools) == 2
            assert tools[0].name == "calc"
            assert tools[1].name == "weather"

    def test_search_by_name(self) -> None:
        with ToolRegistry() as reg:
            reg.add(ToolRegistryEntry(name="calculator", description="Math calculator", tags=["math"]))
            reg.add(ToolRegistryEntry(name="weather", description="Get weather", tags=["api"]))
            results = reg.search("calc")
            assert len(results) == 1
            assert results[0].name == "calculator"

    def test_search_by_description(self) -> None:
        with ToolRegistry() as reg:
            reg.add(ToolRegistryEntry(name="add", description="Add two numbers together"))
            reg.add(ToolRegistryEntry(name="concat", description="Concatenate strings"))
            results = reg.search("numbers")
            assert len(results) == 1
            assert results[0].name == "add"

    def test_search_by_tag(self) -> None:
        with ToolRegistry() as reg:
            reg.add(ToolRegistryEntry(name="calc", description="Calc", tags=["math", "utility"]))
            reg.add(ToolRegistryEntry(name="plot", description="Plot", tags=["math", "viz"]))
            results = reg.search("math")
            assert len(results) == 2

    def test_by_tag(self) -> None:
        with ToolRegistry() as reg:
            reg.add(ToolRegistryEntry(name="a", description="A", tags=["alpha", "beta"]))
            reg.add(ToolRegistryEntry(name="b", description="B", tags=["beta"]))
            reg.add(ToolRegistryEntry(name="c", description="C", tags=["gamma"]))
            assert len(reg.by_tag("beta")) == 2
            assert len(reg.by_tag("gamma")) == 1
            assert len(reg.by_tag("delta")) == 0

    def test_add_with_examples(self) -> None:
        with ToolRegistry() as reg:
            reg.add(ToolRegistryEntry(
                name="add",
                description="Add numbers",
                examples=[
                    ToolExample(description="Add 2 + 3", input={"a": 2, "b": 3}, expected_output=5),
                ],
            ))
            tools = reg.list()
            assert len(tools[0].examples) == 1

    def test_search_empty(self) -> None:
        with ToolRegistry() as reg:
            reg.add(ToolRegistryEntry(name="calc", description="Calculator"))
            assert len(reg.search("nonexistent")) == 0

    def test_throws_after_destroy(self) -> None:
        from gauss.errors import DisposedError

        reg = ToolRegistry()
        reg.destroy()
        with pytest.raises(DisposedError, match="destroyed"):
            reg.add(ToolRegistryEntry(name="x", description="x"))
        with pytest.raises(DisposedError, match="destroyed"):
            reg.search("x")
        with pytest.raises(DisposedError, match="destroyed"):
            reg.list()


class TestToolRegistryTypes:
    def test_tool_example_frozen(self) -> None:
        ex = ToolExample(description="Test", input={"x": 1})
        with pytest.raises(AttributeError):
            ex.description = "changed"  # type: ignore[misc]

    def test_tool_registry_entry_to_dict(self) -> None:
        entry = ToolRegistryEntry(
            name="calc",
            description="Calculator",
            tags=["math"],
            examples=[ToolExample(description="Add", input={"a": 1, "b": 2}, expected_output=3)],
        )
        d = entry.to_dict()
        assert d["name"] == "calc"
        assert d["tags"] == ["math"]
        assert len(d["examples"]) == 1

    def test_tool_registry_entry_from_dict(self) -> None:
        entry = ToolRegistryEntry.from_dict({
            "name": "calc",
            "description": "Calculator",
            "tags": ["math"],
            "examples": [{"description": "Add", "input": {"a": 1}}],
        })
        assert entry.name == "calc"
        assert entry.tags == ["math"]

    def test_tool_search_result_from_dict(self) -> None:
        result = ToolSearchResult.from_dict({
            "name": "calc",
            "description": "Calculator",
            "tags": ["math"],
        })
        assert result.name == "calc"
        assert result.tags == ["math"]
