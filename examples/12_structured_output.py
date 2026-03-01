"""
12 — Structured output with JSON schema validation.

Demonstrates:
  • structured() with inline schema
  • StructuredConfig for advanced options
  • Extracting typed data from LLM responses
"""

import os

from gauss import Agent, ProviderType, StructuredConfig, structured


def main() -> None:
    agent = Agent(
        name="structured-agent",
        provider=ProviderType.OPENAI,
        model="gpt-4o",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    # ── 1. Simple schema ─────────────────────────────────────────────
    print("=== Extract Fruits ===\n")
    result = structured(agent, "List 3 tropical fruits.", schema={
        "type": "object",
        "properties": {
            "fruits": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["fruits"],
    })
    print("Fruits:", result.data["fruits"])

    # ── 2. Complex nested schema ─────────────────────────────────────
    print("\n=== Extract Person Info ===\n")
    result = structured(
        agent,
        "Extract info: John Smith is a 35-year-old software engineer at Google.",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "occupation": {"type": "string"},
                "company": {"type": "string"},
            },
            "required": ["name", "age", "occupation", "company"],
        },
    )
    print(f"Name: {result.data['name']}")
    print(f"Age:  {result.data['age']}")
    print(f"Job:  {result.data['occupation']} @ {result.data['company']}")

    # ── 3. StructuredConfig with raw result ──────────────────────────
    print("\n=== With StructuredConfig ===\n")
    config = StructuredConfig(
        schema={
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                },
            },
            "required": ["summary", "key_points", "sentiment"],
        },
        max_parse_retries=3,
        include_raw=True,
    )

    result = structured(
        agent,
        "Analyze: The new Rust release brings incredible performance improvements "
        "and the community is thrilled.",
        config=config,
    )
    print(f"Sentiment:  {result.data['sentiment']}")
    print(f"Summary:    {result.data['summary']}")
    print(f"Key points: {result.data['key_points']}")
    if result.raw:
        print(f"Raw tokens: {result.raw.usage}")

    # ── Cleanup ──────────────────────────────────────────────────────
    agent.destroy()


if __name__ == "__main__":
    main()
