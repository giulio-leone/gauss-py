"""
11 — ToolRegistry with search, tags, and examples.

Demonstrates:
  • Registering tools with tags and usage examples
  • Searching tools by query
  • Filtering tools by tag
  • Listing all registered tools
"""

from gauss import ToolRegistry, ToolRegistryEntry
from gauss.tool_registry import ToolExample


def main() -> None:
    with ToolRegistry() as registry:
        # ── Register tools ───────────────────────────────────────────
        registry.add(ToolRegistryEntry(
            name="web_search",
            description="Search the web for information using a query string.",
            tags=["search", "web", "information"],
            examples=[
                ToolExample(
                    description="Search for Python tutorials",
                    input={"query": "Python asyncio tutorial"},
                    expected_output={"results": ["..."]},
                ),
            ],
        ))

        registry.add(ToolRegistryEntry(
            name="calculator",
            description="Evaluate mathematical expressions.",
            tags=["math", "calculation"],
            examples=[
                ToolExample(
                    description="Basic arithmetic",
                    input={"expression": "2 + 3 * 4"},
                    expected_output={"result": 14},
                ),
                ToolExample(
                    description="Square root",
                    input={"expression": "sqrt(144)"},
                    expected_output={"result": 12},
                ),
            ],
        ))

        registry.add(ToolRegistryEntry(
            name="translate",
            description="Translate text between languages.",
            tags=["language", "translation", "text"],
        ))

        registry.add(ToolRegistryEntry(
            name="sentiment_analysis",
            description="Analyze the sentiment of text (positive, negative, neutral).",
            tags=["text", "analysis", "nlp"],
        ))

        registry.add(ToolRegistryEntry(
            name="code_executor",
            description="Execute code in a sandboxed environment.",
            tags=["code", "execution", "sandbox"],
        ))

        # ── List all tools ───────────────────────────────────────────
        print("=== All Registered Tools ===\n")
        all_tools = registry.list()
        for tool in all_tools:
            tags = ", ".join(tool.tags) if tool.tags else "none"
            print(f"  {tool.name}: {tool.description} [tags: {tags}]")

        # ── Search by query ──────────────────────────────────────────
        print("\n=== Search: 'math' ===\n")
        results = registry.search("math")
        for r in results:
            print(f"  {r.name}: {r.description}")

        print("\n=== Search: 'text' ===\n")
        results = registry.search("text")
        for r in results:
            print(f"  {r.name}: {r.description}")

        # ── Filter by tag ────────────────────────────────────────────
        print("\n=== Tag: 'text' ===\n")
        results = registry.by_tag("text")
        for r in results:
            print(f"  {r.name} [tags: {', '.join(r.tags)}]")

        print("\n=== Tag: 'search' ===\n")
        results = registry.by_tag("search")
        for r in results:
            print(f"  {r.name} [tags: {', '.join(r.tags)}]")


if __name__ == "__main__":
    main()
