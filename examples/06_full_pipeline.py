"""
06 — Full pipeline: Agent + Memory + Middleware + Guardrails + Telemetry.

Demonstrates combining all production-grade primitives into one pipeline:
  • MiddlewareChain for logging and caching
  • GuardrailChain for content safety
  • Memory for conversation context
  • Telemetry for observability
  • Agent for LLM calls
"""

import os
import time

from gauss import (
    Agent,
    GuardrailChain,
    Memory,
    MiddlewareChain,
    ProviderType,
    Telemetry,
)


def main() -> None:
    # ── 1. Set up middleware ──────────────────────────────────────────
    middleware = (
        MiddlewareChain()
        .use_logging()
        .use_caching(ttl_ms=120_000)  # 2-minute cache
    )
    print("Middleware ready:", type(middleware).__name__)

    # ── 2. Set up guardrails ─────────────────────────────────────────
    guardrails = (
        GuardrailChain()
        .add_content_moderation(blocked_categories=["violence", "hate"])
        .add_pii_detection(action="redact")
        .add_token_limit(max_input=4000, max_output=2000)
        .add_regex_filter(patterns=[r"\bpassword\b", r"\bsecret\b"])
        .add_schema({
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
        })
    )
    print("Active guardrails:", guardrails.list())

    # ── 3. Set up telemetry ──────────────────────────────────────────
    telemetry = Telemetry()

    # ── 4. Set up memory ─────────────────────────────────────────────
    memory = Memory()

    # ── 5. Create agent ──────────────────────────────────────────────
    agent = Agent(
        name="production-agent",
        provider=ProviderType.OPENAI,
        model="gpt-4o",
        api_key=os.environ["OPENAI_API_KEY"],
        system_prompt="You are a helpful assistant. Be concise.",
        temperature=0.5,
        max_tokens=500,
    )

    # ── 6. Run a conversation with full pipeline ─────────────────────
    session_id = "user-42"
    prompts = [
        "What is Rust's ownership model?",
        "How does it compare to garbage collection?",
        "Give me a one-line summary.",
    ]

    for prompt in prompts:
        start = time.time()

        # Store user message in memory
        memory.store("conversation", prompt, session_id=session_id)

        # Call the agent
        result = agent.run(prompt)
        elapsed_ms = (time.time() - start) * 1000

        # Store assistant reply in memory
        memory.store("conversation", result.text, session_id=session_id)

        # Record telemetry span
        telemetry.record_span({
            "name": "agent.run",
            "duration_ms": round(elapsed_ms),
            "model": "gpt-4o",
            "tokens_in": result.usage.get("prompt_tokens", 0),
            "tokens_out": result.usage.get("completion_tokens", 0),
        })

        print(f"\n> {prompt}")
        print(f"  {result.text[:200]}")
        print(f"  ({elapsed_ms:.0f}ms)")

    # ── 7. Inspect telemetry ─────────────────────────────────────────
    print("\n=== Telemetry ===")
    spans = telemetry.export_spans()
    print(f"Recorded {len(spans)} spans")
    for span in spans:
        print(f"  {span.get('name')}: {span.get('duration_ms')}ms")

    metrics = telemetry.export_metrics()
    print("Metrics:", metrics)

    # ── 8. Inspect memory ────────────────────────────────────────────
    print("\n=== Memory ===")
    entries = memory.recall(session_id=session_id)
    print(f"{len(entries)} entries in session '{session_id}'")

    # ── Cleanup ──────────────────────────────────────────────────────
    telemetry.clear()
    agent.destroy()
    memory.destroy()
    middleware.destroy()
    guardrails.destroy()
    telemetry.destroy()
    print("\nAll resources cleaned up.")


if __name__ == "__main__":
    main()
