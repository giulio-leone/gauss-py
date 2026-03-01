"""Shared types and helpers for Gauss SDK."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class ProviderType(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    GROQ = "groq"
    OLLAMA = "ollama"
    DEEPSEEK = "deepseek"


@dataclass
class Message:
    """A chat message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ToolDef:
    """Tool definition for function calling."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass
class AgentResult:
    """Result from an agent run."""

    text: str
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    usage: dict[str, Any]

    def __str__(self) -> str:
        return self.text


@dataclass
class SearchResult:
    """Vector search result."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Agent configuration with sensible defaults.

    All fields are optional. If no provider/model/api_key is set,
    Gauss auto-detects from environment variables.

    Example::

        # Minimal — auto-detect everything
        config = AgentConfig()

        # Explicit
        config = AgentConfig(
            provider=ProviderType.OPENAI,
            model="gpt-4o",
            api_key="sk-...",
            temperature=0.7,
        )
    """

    name: str = "gauss-agent"
    provider: ProviderType | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_retries: int = 3
    tools: list[ToolDef] = field(default_factory=list)
    stop_condition: str | None = None

    def resolve(self) -> tuple[ProviderType, str, str]:
        """Resolve provider, model, and API key from config or env."""
        provider = self.provider or detect_provider()
        model = self.model or _default_model(provider)
        api_key = self.api_key or resolve_api_key(provider)
        return provider, model, api_key


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

_ENV_MAP: dict[ProviderType, str] = {
    ProviderType.OPENAI: "OPENAI_API_KEY",
    ProviderType.ANTHROPIC: "ANTHROPIC_API_KEY",
    ProviderType.GOOGLE: "GOOGLE_API_KEY",
    ProviderType.GROQ: "GROQ_API_KEY",
    ProviderType.DEEPSEEK: "DEEPSEEK_API_KEY",
}

_DEFAULT_MODELS: dict[ProviderType, str] = {
    ProviderType.OPENAI: "gpt-4o",
    ProviderType.ANTHROPIC: "claude-sonnet-4-20250514",
    ProviderType.GOOGLE: "gemini-2.0-flash",
    ProviderType.GROQ: "llama-3.3-70b-versatile",
    ProviderType.OLLAMA: "llama3.2",
    ProviderType.DEEPSEEK: "deepseek-chat",
}


def detect_provider() -> ProviderType:
    """Auto-detect provider from environment variables."""
    for provider, env_var in _ENV_MAP.items():
        if os.environ.get(env_var):
            return provider
    if os.environ.get("OLLAMA_HOST"):
        return ProviderType.OLLAMA
    raise OSError(
        "No API key found. Set one of: " + ", ".join(_ENV_MAP.values()) + " or OLLAMA_HOST"
    )


def resolve_api_key(provider: ProviderType) -> str:
    """Resolve API key for a provider from environment."""
    if provider == ProviderType.OLLAMA:
        return "ollama"
    env_var = _ENV_MAP.get(provider)
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")
    key = os.environ.get(env_var, "")
    if not key:
        raise OSError(f"{env_var} not set")
    return key


def _default_model(provider: ProviderType) -> str:
    return _DEFAULT_MODELS.get(provider, "gpt-4o")
