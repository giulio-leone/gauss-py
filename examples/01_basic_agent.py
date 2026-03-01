"""
01 — Basic Agent, gauss() shorthand, and batch().

Demonstrates:
  • Creating an Agent with explicit config
  • The gauss() one-liner for quick calls
  • batch() for concurrent prompt execution
  • Context managers for auto-cleanup
  • Streaming with stream_iter()
"""

import asyncio
import os

from gauss import Agent, AgentConfig, Message, ProviderType, batch, gauss, OPENAI_DEFAULT


async def main() -> None:
    # ── 1. One-liner ─────────────────────────────────────────────────
    answer = gauss("What is the capital of France?")
    print("One-liner:", answer)

    # ── 2. Agent with explicit config ────────────────────────────────
    agent = Agent(
        name="basic-agent",
        provider=ProviderType.OPENAI,
        model=OPENAI_DEFAULT,
        api_key=os.environ["OPENAI_API_KEY"],
        system_prompt="You are a concise assistant. Answer in one sentence.",
        temperature=0.3,
    )

    result = agent.run("Explain quantum entanglement.")
    print("Agent result:", result.text)
    print("Token usage:", result.usage)

    agent.destroy()

    # ── 3. Context manager (auto-cleanup) ────────────────────────────
    with Agent(system_prompt="Reply in pirate speak.") as pirate:
        result = pirate.run("What is the weather like?")
        print("Pirate:", result.text)

    # ── 4. Multi-message conversation ────────────────────────────────
    with Agent() as agent:
        result = agent.run([
            Message("system", "You are a math tutor."),
            Message("user", "What is 12 × 13?"),
        ])
        print("Math tutor:", result.text)

    # ── 5. Simple text generation ────────────────────────────────────
    with Agent() as agent:
        text = agent.generate("Write a haiku about Rust programming.")
        print("Haiku:", text)

    # ── 6. Streaming ─────────────────────────────────────────────────
    with Agent() as agent:
        stream = agent.stream_iter("Tell me a short joke.")
        async for event in stream:
            if event.type == "text_delta" and event.text:
                print(event.text, end="", flush=True)
        print()
        print("Full text:", stream.text)

    # ── 7. Batch execution ───────────────────────────────────────────
    prompts = [
        "Translate 'hello' to Spanish",
        "Translate 'hello' to French",
        "Translate 'hello' to Japanese",
    ]
    items = batch(prompts, concurrency=3)
    for item in items:
        if item.result:
            print(f"{item.input} → {item.result.text}")
        else:
            print(f"{item.input} → ERROR: {item.error}")


if __name__ == "__main__":
    asyncio.run(main())
