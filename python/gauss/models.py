"""Model Constants — Single Source of Truth.

Update these when new model versions are released.
All examples, tests, and defaults reference this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# ─── OpenAI ──────────────────────────────────────────
OPENAI_DEFAULT = "gpt-5.2"
OPENAI_FAST = "gpt-4.1"
OPENAI_REASONING = "o4-mini"
OPENAI_IMAGE = "gpt-image-1"

# ─── Anthropic ───────────────────────────────────────
ANTHROPIC_DEFAULT = "claude-sonnet-4-20250514"
ANTHROPIC_FAST = "claude-haiku-4-20250414"
ANTHROPIC_PREMIUM = "claude-opus-4-20250414"

# ─── Google ──────────────────────────────────────────
GOOGLE_DEFAULT = "gemini-2.5-flash"
GOOGLE_PREMIUM = "gemini-2.5-pro"
GOOGLE_IMAGE = "gemini-2.0-flash"

# ─── OpenRouter ──────────────────────────────────────
OPENROUTER_DEFAULT = "openai/gpt-5.2"

# ─── DeepSeek ────────────────────────────────────────
DEEPSEEK_DEFAULT = "deepseek-chat"
DEEPSEEK_REASONING = "deepseek-reasoner"

# ─── Enterprise OpenAI-Compatible Providers ─────────
TOGETHER_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
FIREWORKS_DEFAULT = "accounts/fireworks/models/llama-v3p1-70b-instruct"
MISTRAL_DEFAULT = "mistral-large-latest"
PERPLEXITY_DEFAULT = "sonar-pro"
XAI_DEFAULT = "grok-3-beta"

# ─── Provider Defaults Map ───────────────────────────
PROVIDER_DEFAULTS: dict[str, str] = {
    "openai": OPENAI_DEFAULT,
    "anthropic": ANTHROPIC_DEFAULT,
    "google": GOOGLE_DEFAULT,
    "openrouter": OPENROUTER_DEFAULT,
    "deepseek": DEEPSEEK_DEFAULT,
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.2",
    "together": TOGETHER_DEFAULT,
    "fireworks": FIREWORKS_DEFAULT,
    "mistral": MISTRAL_DEFAULT,
    "perplexity": PERPLEXITY_DEFAULT,
    "xai": XAI_DEFAULT,
}


def default_model(provider: str) -> str:
    """Get the default model for a provider."""
    return PROVIDER_DEFAULTS.get(provider, OPENAI_DEFAULT)
