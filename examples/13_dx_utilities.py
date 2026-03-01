"""
13 — DX utilities: template(), pipe(), map_async(), compose(), with_retry().

Demonstrates:
  • PromptTemplate with {{variables}}
  • Built-in templates (summarize, translate, code_review, classify, extract)
  • pipe() for async pipelines
  • map_sync() / map_async() for batch processing
  • compose() for function composition
  • with_retry() / retryable() for resilient execution
"""

import asyncio
import os

from gauss import (
    Agent,
    OPENAI_DEFAULT,
    ProviderType,
    RetryConfig,
    compose,
    map_sync,
    pipe,
    retryable,
    template,
    with_retry,
)
from gauss.template import classify, code_review, extract, summarize, translate


async def main() -> None:
    # ── 1. Prompt Templates ──────────────────────────────────────────
    print("=== Templates ===\n")

    # Custom template
    greet = template("Hello {{name}}, welcome to {{place}}!")
    print(greet(name="Alice", place="Gauss"))

    analysis = template(
        "Analyze the {{aspect}} of {{subject}}.\n"
        "Focus on: {{focus}}\n"
        "Format: {{format}}"
    )
    print("Variables:", analysis.variables)
    prompt = analysis(
        aspect="performance",
        subject="Python",
        focus="async I/O",
        format="bullet points",
    )
    print(prompt[:80], "...")

    # Built-in templates
    print("\nBuilt-in templates:")
    print(summarize(format="article", style="concise", text="Lorem ipsum...")[:60])
    print(translate(language="Spanish", text="Hello world")[:60])
    print(code_review(language="Python", code="def f(): pass")[:60])
    print(classify(categories="tech, science, art", text="AI is amazing")[:60])
    print(extract(fields="name, date", text="John was born on Jan 1")[:60])

    # ── 2. pipe() ────────────────────────────────────────────────────
    print("\n=== Pipe ===\n")

    result = await pipe(
        "hello world",
        lambda s: s.upper(),
        lambda s: f"[{s}]",
        lambda s: s.replace("WORLD", "GAUSS"),
    )
    print("Pipe result:", result)

    # ── 3. compose() ─────────────────────────────────────────────────
    print("\n=== Compose ===\n")

    enhance = compose(
        lambda text: text.strip(),
        lambda text: text.upper(),
        lambda text: f">>> {text} <<<",
    )
    print(enhance("  hello gauss  "))

    # ── 4. map_sync() ────────────────────────────────────────────────
    print("\n=== Map Sync ===\n")

    items = ["apple", "banana", "cherry"]
    results = map_sync(items, lambda x: x.upper(), concurrency=2)
    print("Mapped:", results)

    # ── 5. with_retry() ──────────────────────────────────────────────
    print("\n=== Retry ===\n")

    attempt_count = 0

    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError(f"Attempt {attempt_count} failed")
        return "Success!"

    result = with_retry(
        flaky_function,
        config=RetryConfig(
            max_retries=5,
            backoff="exponential",
            base_delay_s=0.1,
            jitter=0.05,
            on_retry=lambda err, attempt, delay: print(
                f"  Retry {attempt}: {err} (waiting {delay:.2f}s)"
            ),
        ),
    )
    print(f"Result: {result} (after {attempt_count} attempts)")

    # ── 6. retryable() with an agent ─────────────────────────────────
    print("\n=== Retryable Agent ===\n")
    print("(Wraps agent.run() with automatic retries)")

    agent = Agent(
        name="retry-demo",
        provider=ProviderType.OPENAI,
        model=OPENAI_DEFAULT,
        api_key=os.environ["OPENAI_API_KEY"],
    )

    resilient_run = retryable(agent, max_retries=3, backoff="exponential")
    result = resilient_run("What is 2 + 2?")
    print(f"Answer: {result.text}")

    agent.destroy()


if __name__ == "__main__":
    asyncio.run(main())
