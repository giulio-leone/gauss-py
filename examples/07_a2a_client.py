"""
07 — A2A (Agent-to-Agent) protocol client.

Demonstrates:
  • Discovering a remote agent via AgentCard
  • Sending messages with A2aMessage
  • Quick ask (text-in → text-out)
  • Getting and cancelling tasks

Requires a running A2A-compliant server (e.g. http://localhost:8080).
"""

import asyncio

from gauss import A2aClient, A2aMessage, Part, TaskState


async def main() -> None:
    base_url = "http://localhost:8080"
    client = A2aClient(base_url, auth_token=None)

    # ── 1. Discover agent capabilities ───────────────────────────────
    print("=== Agent Discovery ===\n")
    card = await client.discover()
    print(f"Name:        {card.name}")
    print(f"URL:         {card.url}")
    print(f"Description: {card.description}")
    print(f"Version:     {card.version}")

    if card.capabilities:
        print(f"Streaming:   {card.capabilities.streaming}")
        print(f"Push notif:  {card.capabilities.push_notifications}")

    for skill in card.skills:
        print(f"\nSkill: {skill.name}")
        print(f"  ID:   {skill.id}")
        print(f"  Tags: {', '.join(skill.tags)}")

    # ── 2. Quick ask ─────────────────────────────────────────────────
    print("\n=== Quick Ask ===\n")
    answer = await client.ask("What is the capital of Japan?")
    print(f"Answer: {answer}")

    # ── 3. Full message exchange ─────────────────────────────────────
    print("\n=== Message Exchange ===\n")
    message = A2aMessage.user_text("Summarize the latest AI research trends.")
    result = await client.send_message(message)

    if hasattr(result, "status"):
        # Got a Task back
        task = result
        print(f"Task ID: {task.id}")
        print(f"State:   {task.status.state.value}")
        if task.text:
            print(f"Text:    {task.text}")

        # Check task status
        updated = await client.get_task(task.id)
        print(f"Updated state: {updated.status.state.value}")

        # Show artifacts
        for artifact in updated.artifacts:
            print(f"Artifact: {artifact.name}")
            for part in artifact.parts:
                if part.text:
                    print(f"  {part.text[:100]}...")
    else:
        # Got a direct A2aMessage back
        print(f"Response: {result.text}")

    # ── 4. Multi-part message ────────────────────────────────────────
    print("\n=== Multi-part Message ===\n")
    multi_msg = A2aMessage(
        role="user",
        parts=(
            Part.text_part("Analyze this data:"),
            Part.data_part({"sales": [100, 200, 150]}, mime_type="application/json"),
        ),
    )
    result = await client.send_message(multi_msg)
    if hasattr(result, "text"):
        print(f"Response: {result.text}")

    # ── 5. Cancel a task ─────────────────────────────────────────────
    print("\n=== Task Lifecycle ===\n")
    msg = A2aMessage.user_text("Write a very long essay about history.")
    result = await client.send_message(msg)
    if hasattr(result, "id"):
        cancelled = await client.cancel_task(result.id)
        print(f"Cancelled task {cancelled.id}: {cancelled.status.state.value}")


if __name__ == "__main__":
    asyncio.run(main())
