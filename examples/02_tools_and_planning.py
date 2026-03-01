"""
02 — Agent with tool definitions.

Demonstrates:
  • Defining tools via ToolDef
  • Adding tools to an agent (add_tool / add_tools)
  • Running the agent and inspecting tool_calls
  • Querying provider capabilities
"""

import json
import os

from gauss import Agent, ProviderType, ToolDef, OPENAI_DEFAULT


def main() -> None:
    # ── Define tools ─────────────────────────────────────────────────
    calculator = ToolDef(
        name="calculator",
        description="Evaluate a mathematical expression.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression, e.g. '2 + 3 * 4'",
                },
            },
            "required": ["expression"],
        },
    )

    weather = ToolDef(
        name="get_weather",
        description="Get current weather for a city.",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["city"],
        },
    )

    # ── Create agent with tools ──────────────────────────────────────
    with Agent(
        name="tool-agent",
        provider=ProviderType.OPENAI,
        model=OPENAI_DEFAULT,
        api_key=os.environ["OPENAI_API_KEY"],
        system_prompt="Use the available tools to answer questions.",
        tools=[calculator],
    ) as agent:
        # Add more tools via chaining
        agent.add_tool(weather)

        # Check capabilities
        caps = agent.capabilities
        print(f"Tool use supported: {caps.tool_use}")
        print(f"Streaming supported: {caps.streaming}")
        print(f"Vision supported: {caps.vision}")

        # Run with a prompt that should trigger tool use
        result = agent.run("What is 42 * 17?")
        print("\nResponse:", result.text)

        if result.tool_calls:
            print("\nTool calls made:")
            for tc in result.tool_calls:
                print(f"  • {tc.get('name', 'unknown')}({json.dumps(tc.get('arguments', {}))})")


if __name__ == "__main__":
    main()
