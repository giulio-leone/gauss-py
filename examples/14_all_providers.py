"""
14 — Multi-provider support: OpenAI, Anthropic, Google, Groq, DeepSeek, Ollama.

Demonstrates:
  • Configuring agents for different LLM providers
  • Auto-detection from environment variables
  • Provider-specific features (extended thinking, grounding, etc.)

Set the appropriate API key env vars before running:
  OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY, DEEPSEEK_API_KEY
"""

import os

from gauss import (
    Agent,
    AgentConfig,
    ANTHROPIC_DEFAULT,
    DEEPSEEK_DEFAULT,
    GOOGLE_IMAGE,
    OPENAI_DEFAULT,
    OPENAI_FAST,
    PROVIDER_DEFAULTS,
    ProviderType,
)


def run_agent(label: str, **kwargs) -> None:
    """Helper to create, run, and destroy an agent."""
    try:
        with Agent(**kwargs) as agent:
            result = agent.run("Explain recursion in one sentence.")
            print(f"[{label}] {result.text}")
    except Exception as e:
        print(f"[{label}] Skipped: {e}")


def main() -> None:
    # ── 1. OpenAI ────────────────────────────────────────────────────
    print("=== OpenAI ===")
    if os.environ.get("OPENAI_API_KEY"):
        run_agent(
            "GPT-4o",
            provider=ProviderType.OPENAI,
            model=OPENAI_DEFAULT,
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0.5,
        )
        run_agent(
            "GPT-4o-mini",
            provider=ProviderType.OPENAI,
            model=OPENAI_FAST,
            api_key=os.environ["OPENAI_API_KEY"],
        )
    else:
        print("  OPENAI_API_KEY not set, skipping.")

    # ── 2. Anthropic ─────────────────────────────────────────────────
    print("\n=== Anthropic ===")
    if os.environ.get("ANTHROPIC_API_KEY"):
        run_agent(
            "Claude Sonnet",
            provider=ProviderType.ANTHROPIC,
            model=ANTHROPIC_DEFAULT,
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
        # Extended thinking
        run_agent(
            "Claude (thinking)",
            provider=ProviderType.ANTHROPIC,
            model=ANTHROPIC_DEFAULT,
            api_key=os.environ["ANTHROPIC_API_KEY"],
            thinking_budget=2048,
        )
    else:
        print("  ANTHROPIC_API_KEY not set, skipping.")

    # ── 3. Google (Gemini) ───────────────────────────────────────────
    print("\n=== Google ===")
    if os.environ.get("GOOGLE_API_KEY"):
        run_agent(
            "Gemini 2.0 Flash",
            provider=ProviderType.GOOGLE,
            model=GOOGLE_IMAGE,
            api_key=os.environ["GOOGLE_API_KEY"],
        )
        # With grounding (Google Search)
        run_agent(
            "Gemini (grounded)",
            provider=ProviderType.GOOGLE,
            model=GOOGLE_IMAGE,
            api_key=os.environ["GOOGLE_API_KEY"],
            grounding=True,
        )
    else:
        print("  GOOGLE_API_KEY not set, skipping.")

    # ── 4. Groq ──────────────────────────────────────────────────────
    print("\n=== Groq ===")
    if os.environ.get("GROQ_API_KEY"):
        run_agent(
            "Llama 3 (Groq)",
            provider=ProviderType.GROQ,
            model=PROVIDER_DEFAULTS["groq"],
            api_key=os.environ["GROQ_API_KEY"],
        )
    else:
        print("  GROQ_API_KEY not set, skipping.")

    # ── 5. DeepSeek ──────────────────────────────────────────────────
    print("\n=== DeepSeek ===")
    if os.environ.get("DEEPSEEK_API_KEY"):
        run_agent(
            "DeepSeek Chat",
            provider=ProviderType.DEEPSEEK,
            model=DEEPSEEK_DEFAULT,
            api_key=os.environ["DEEPSEEK_API_KEY"],
        )
    else:
        print("  DEEPSEEK_API_KEY not set, skipping.")

    # ── 6. Ollama (local) ────────────────────────────────────────────
    print("\n=== Ollama (local) ===")
    run_agent(
        "Ollama",
        provider=ProviderType.OLLAMA,
        model=PROVIDER_DEFAULTS["ollama"],
        base_url="http://localhost:11434",
    )

    # ── 7. OpenRouter (via base_url override) ────────────────────────
    print("\n=== OpenRouter ===")
    if os.environ.get("OPENROUTER_API_KEY"):
        run_agent(
            "OpenRouter",
            provider=ProviderType.OPENAI,  # OpenRouter is OpenAI-compatible
            model=f"anthropic/{ANTHROPIC_DEFAULT}",
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        print("  OPENROUTER_API_KEY not set, skipping.")

    # ── 8. Auto-detect from env ──────────────────────────────────────
    print("\n=== Auto-detect ===")
    with Agent() as agent:
        result = agent.run("Say 'hello' in 3 languages.")
        print(f"Auto-detected: {result.text}")


if __name__ == "__main__":
    main()
