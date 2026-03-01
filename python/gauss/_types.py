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
class Citation:
    """A citation reference from document-aware responses."""

    citation_type: str
    """Citation type: char_location, page_location, content_block_location."""
    cited_text: str | None = None
    """The cited text from the document."""
    document_title: str | None = None
    """Title of the source document."""
    start: int | None = None
    """Start index (character, page, or block depending on type)."""
    end: int | None = None
    """End index (character, page, or block depending on type)."""


@dataclass
class AgentResult:
    """Result from an agent run."""

    text: str
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    usage: dict[str, Any]
    thinking: str | None = None
    citations: list[Citation] = field(default_factory=list)
    grounding_metadata: list[GroundingMetadata] = field(default_factory=list)

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
class ProviderCapabilities:
    """Feature capabilities of a provider/model combination."""

    streaming: bool = False
    tool_use: bool = False
    vision: bool = False
    audio: bool = False
    extended_thinking: bool = False
    citations: bool = False
    cache_control: bool = False
    structured_output: bool = False
    reasoning_effort: bool = False
    image_generation: bool = False
    grounding: bool = False
    code_execution: bool = False
    web_search: bool = False


@dataclass
class CodeExecutionOptions:
    """Configuration for programmatic tool calling (code execution).

    Example::

        # Enable all runtimes
        opts = CodeExecutionOptions()

        # Python only with 60s timeout
        opts = CodeExecutionOptions(python=True, javascript=False, bash=False, timeout=60)
    """

    python: bool = True
    javascript: bool = True
    bash: bool = True
    unified: bool = False
    timeout: int = 30
    sandbox: Literal["default", "strict", "permissive"] = "default"


@dataclass
class CodeExecutionResult:
    """Result from code execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    runtime: str
    success: bool


@dataclass
class GroundingChunk:
    """A single grounding chunk (web search result)."""

    url: str | None = None
    title: str | None = None


@dataclass
class GroundingMetadata:
    """Metadata from Google Search grounding."""

    search_queries: list[str] = field(default_factory=list)
    grounding_chunks: list[GroundingChunk] = field(default_factory=list)
    search_entry_point: str | None = None


@dataclass
class ImageGenerationConfig:
    """Configuration for image generation.

    Example::

        # OpenAI DALL-E
        config = ImageGenerationConfig(model="dall-e-3", size="1024x1024")

        # Gemini
        config = ImageGenerationConfig(aspect_ratio="16:9")
    """

    model: str | None = None
    size: str | None = None
    quality: str | None = None
    style: str | None = None
    aspect_ratio: str | None = None
    n: int | None = None
    response_format: str | None = None


@dataclass
class GeneratedImageData:
    """A single generated image."""

    url: str | None = None
    base64: str | None = None
    mime_type: str | None = None


@dataclass
class ImageGenerationResult:
    """Result of image generation."""

    images: list[GeneratedImageData] = field(default_factory=list)
    revised_prompt: str | None = None


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

        # With code execution
        config = AgentConfig(code_execution=True)
        config = AgentConfig(code_execution=CodeExecutionOptions(python=True))
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
    thinking_budget: int | None = None
    cache_control: bool = False
    code_execution: bool | CodeExecutionOptions | None = None
    grounding: bool = False
    """Enable Google Search grounding (Gemini only)."""
    native_code_execution: bool = False
    """Enable native code execution / Gemini code interpreter."""
    response_modalities: list[str] | None = None
    """Response modalities (e.g. ["TEXT", "IMAGE"] for Gemini image generation)."""

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

from gauss.models import (
    ANTHROPIC_DEFAULT,
    DEEPSEEK_DEFAULT,
    GOOGLE_DEFAULT,
    OPENAI_DEFAULT,
    PROVIDER_DEFAULTS,
)

_DEFAULT_MODELS: dict[ProviderType, str] = {
    ProviderType.OPENAI: OPENAI_DEFAULT,
    ProviderType.ANTHROPIC: ANTHROPIC_DEFAULT,
    ProviderType.GOOGLE: GOOGLE_DEFAULT,
    ProviderType.GROQ: PROVIDER_DEFAULTS["groq"],
    ProviderType.OLLAMA: PROVIDER_DEFAULTS["ollama"],
    ProviderType.DEEPSEEK: DEEPSEEK_DEFAULT,
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
    return _DEFAULT_MODELS.get(provider, OPENAI_DEFAULT)
