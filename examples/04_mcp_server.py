"""
04 — MCP Server with tools, resources, and prompts.

Demonstrates:
  • Creating an McpServer
  • Registering tools, resources, and prompts
  • Handling JSON-RPC messages (tools/list, resources/list, prompts/list)
"""

import asyncio
import json

from gauss import (
    McpPrompt,
    McpPromptArgument,
    McpResource,
    McpServer,
    ToolDef,
)


async def main() -> None:
    # ── Create MCP server ────────────────────────────────────────────
    with McpServer("gauss-tools", "1.0.0") as server:

        # Register tools
        server.add_tool(ToolDef(
            name="search",
            description="Search the knowledge base.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results"},
                },
                "required": ["query"],
            },
        ))
        server.add_tool(ToolDef(
            name="calculate",
            description="Evaluate a math expression.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        ))

        # Register resources
        server.add_resource(McpResource(
            uri="file:///docs/readme.md",
            name="README",
            description="Project documentation",
            mime_type="text/markdown",
        ))
        server.add_resource(McpResource(
            uri="db://users",
            name="Users Database",
            description="User records",
        ))

        # Register prompts
        server.add_prompt(McpPrompt(
            name="summarize",
            description="Summarize a document",
            arguments=[
                McpPromptArgument(name="text", description="Text to summarize", required=True),
                McpPromptArgument(name="style", description="Summary style"),
            ],
        ))
        server.add_prompt(McpPrompt(
            name="translate",
            description="Translate text to a target language",
            arguments=[
                McpPromptArgument(name="text", required=True),
                McpPromptArgument(name="language", required=True),
            ],
        ))

        # ── Handle JSON-RPC requests ─────────────────────────────────
        # List tools
        response = await server.handle_message({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
        })
        print("Tools:", json.dumps(response, indent=2))

        # List resources
        response = await server.handle_message({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "resources/list",
        })
        print("\nResources:", json.dumps(response, indent=2))

        # List prompts
        response = await server.handle_message({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "prompts/list",
        })
        print("\nPrompts:", json.dumps(response, indent=2))

        # Call a tool
        response = await server.handle_message({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "calculate",
                "arguments": {"expression": "2 + 2"},
            },
        })
        print("\nTool call result:", json.dumps(response, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
