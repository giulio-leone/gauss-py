"""
Gauss AI — Production-grade AI agents powered by Rust.

Quick start::

    from gauss import Agent

    agent = Agent()
    result = agent.run("What is the meaning of life?")
    print(result.text)

One-liner::

    from gauss import gauss
    print(gauss("What is the meaning of life?"))
"""

from gauss._types import (
    AgentConfig,
    AgentResult,
    Citation,
    CodeExecutionOptions,
    CodeExecutionResult,
    GeneratedImageData,
    GroundingChunk,
    GroundingMetadata,
    ImageGenerationConfig,
    ImageGenerationResult,
    Message,
    ProviderCapabilities,
    ProviderType,
    SearchResult,
    ToolDef,
)
from gauss.agent import Agent, gauss
from gauss.batch import BatchItem, batch
from gauss.stream import AgentStream, StreamEvent, parse_partial_json
from gauss.code_execution import available_runtimes, execute_code, generate_image
from gauss.models import (
    ANTHROPIC_DEFAULT,
    ANTHROPIC_FAST,
    ANTHROPIC_PREMIUM,
    DEEPSEEK_DEFAULT,
    DEEPSEEK_REASONING,
    GOOGLE_DEFAULT,
    GOOGLE_IMAGE,
    GOOGLE_PREMIUM,
    OPENAI_DEFAULT,
    OPENAI_FAST,
    OPENAI_IMAGE,
    OPENAI_REASONING,
    OPENROUTER_DEFAULT,
    PROVIDER_DEFAULTS,
    default_model,
)
from gauss.approval import ApprovalManager
from gauss.checkpoint import CheckpointStore
from gauss.config import parse_agent_config, resolve_env
from gauss.eval import EvalRunner
from gauss.graph import Graph
from gauss.guardrail import GuardrailChain
from gauss.mcp import (
    McpServer,
    McpResource,
    McpPrompt,
    McpPromptArgument,
    McpContent,
    McpResourceContent,
    McpPromptMessage,
    McpPromptResult,
    McpModelHint,
    McpModelPreferences,
    McpSamplingMessage,
    McpSamplingRequest,
    McpSamplingResponse,
)
from gauss.memory import Memory
from gauss.middleware import MiddlewareChain
from gauss.network import Network
from gauss.team import Team
from gauss.pipeline import (
    compose,
    compose_async,
    filter_async,
    filter_sync,
    map_async,
    map_sync,
    pipe,
    reduce_async,
    reduce_sync,
)
from gauss.plugin import PluginRegistry
from gauss.resilience import (
    create_circuit_breaker,
    create_fallback_provider,
    create_resilient_provider,
)
from gauss.retry import RetryConfig, retryable, with_retry
from gauss.structured import StructuredConfig, StructuredResult, structured
from gauss.telemetry import Telemetry
from gauss.template import (
    PromptTemplate,
    classify,
    code_review,
    extract,
    summarize,
    template,
    translate,
)
from gauss.tokens import (
    count_message_tokens,
    count_tokens,
    count_tokens_for_model,
    get_context_window_size,
)
from gauss.tool_validator import ToolValidator
from gauss.vector_store import VectorStore
from gauss.workflow import Workflow
from gauss.spec import AgentSpec, AgentToolSpec, SkillSpec, SkillStep, SkillParam, discover_agents
from gauss.a2a import (
    A2aClient,
    A2aMessage,
    AgentCard,
    AgentCapabilities,
    AgentSkill,
    Artifact,
    Part,
    Task,
    TaskState,
    TaskStatus,
)
from gauss.tool_registry import (
    ToolRegistry,
    ToolRegistryEntry,
    ToolExample as ToolRegistryExample,
    ToolSearchResult,
)

__all__ = [
    # One-liner
    "gauss",
    "batch",
    "BatchItem",
    # Streaming
    "AgentStream",
    "StreamEvent",
    # Model Constants
    "OPENAI_DEFAULT",
    "OPENAI_FAST",
    "OPENAI_REASONING",
    "OPENAI_IMAGE",
    "ANTHROPIC_DEFAULT",
    "ANTHROPIC_FAST",
    "ANTHROPIC_PREMIUM",
    "GOOGLE_DEFAULT",
    "GOOGLE_PREMIUM",
    "GOOGLE_IMAGE",
    "OPENROUTER_DEFAULT",
    "DEEPSEEK_DEFAULT",
    "DEEPSEEK_REASONING",
    "PROVIDER_DEFAULTS",
    "default_model",
    # Code Execution & Image Generation
    "execute_code",
    "available_runtimes",
    "generate_image",
    # Core
    "Agent",
    "AgentConfig",
    "AgentResult",
    "Citation",
    "CodeExecutionOptions",
    "CodeExecutionResult",
    "GeneratedImageData",
    "GroundingChunk",
    "GroundingMetadata",
    "ImageGenerationConfig",
    "ImageGenerationResult",
    "Message",
    "ProviderCapabilities",
    "ProviderType",
    "ToolDef",
    # Memory & RAG
    "Memory",
    "VectorStore",
    "SearchResult",
    # Orchestration
    "Graph",
    "Workflow",
    "Network",
    "Team",
    # Middleware & Plugins
    "MiddlewareChain",
    "PluginRegistry",
    # Safety
    "GuardrailChain",
    "ToolValidator",
    # Eval
    "EvalRunner",
    # Telemetry
    "Telemetry",
    # HITL
    "ApprovalManager",
    "CheckpointStore",
    # MCP
    "McpServer",
    "McpResource",
    "McpPrompt",
    "McpPromptArgument",
    "McpContent",
    "McpResourceContent",
    "McpPromptMessage",
    "McpPromptResult",
    "McpModelHint",
    "McpModelPreferences",
    "McpSamplingMessage",
    "McpSamplingRequest",
    "McpSamplingResponse",
    # Resilience
    "create_fallback_provider",
    "create_circuit_breaker",
    "create_resilient_provider",
    # Utilities
    "count_tokens",
    "count_tokens_for_model",
    "count_message_tokens",
    "get_context_window_size",
    "parse_partial_json",
    "parse_agent_config",
    "resolve_env",
    # Retry
    "with_retry",
    "retryable",
    "RetryConfig",
    # Structured Output
    "structured",
    "StructuredConfig",
    "StructuredResult",
    # Prompt Templates
    "template",
    "PromptTemplate",
    "summarize",
    "translate",
    "code_review",
    "classify",
    "extract",
    # Pipeline & Async Helpers
    "pipe",
    "map_async",
    "map_sync",
    "filter_async",
    "filter_sync",
    "reduce_async",
    "reduce_sync",
    "compose",
    "compose_async",
    # AGENTS.MD & SKILL.MD Parsers
    "AgentSpec",
    "AgentToolSpec",
    "SkillSpec",
    "SkillStep",
    "SkillParam",
    "discover_agents",
    # A2A Protocol
    "A2aClient",
    "A2aMessage",
    "AgentCard",
    "AgentCapabilities",
    "AgentSkill",
    "Artifact",
    "Part",
    "Task",
    "TaskState",
    "TaskStatus",
    # Tool Registry
    "ToolRegistry",
    "ToolRegistryEntry",
    "ToolRegistryExample",
    "ToolSearchResult",
]
