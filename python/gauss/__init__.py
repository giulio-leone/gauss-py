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

from __future__ import annotations

import importlib
from typing import Any

# ── Eager imports: lightweight core that every user needs ──────────────────
from gauss._types import (
    AgentConfig,
    AgentResult,
    Citation,
    CostEstimate,
    Message,
    ProviderType,
    ToolDef,
)
from gauss.agent import Agent, enterprise_preset, gauss
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
    TOGETHER_DEFAULT,
    FIREWORKS_DEFAULT,
    MISTRAL_DEFAULT,
    PERPLEXITY_DEFAULT,
    XAI_DEFAULT,
    PROVIDER_DEFAULTS,
    default_model,
)

# ── Lazy-loading registry: module_path → attr_name ────────────────────────
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # _types (non-core)
    "CodeExecutionOptions": ("gauss._types", "CodeExecutionOptions"),
    "CodeExecutionResult": ("gauss._types", "CodeExecutionResult"),
    "GeneratedImageData": ("gauss._types", "GeneratedImageData"),
    "GroundingChunk": ("gauss._types", "GroundingChunk"),
    "GroundingMetadata": ("gauss._types", "GroundingMetadata"),
    "ImageGenerationConfig": ("gauss._types", "ImageGenerationConfig"),
    "ImageGenerationResult": ("gauss._types", "ImageGenerationResult"),
    "ProviderCapabilities": ("gauss._types", "ProviderCapabilities"),
    "SearchResult": ("gauss._types", "SearchResult"),
    # batch
    "batch": ("gauss.batch", "batch"),
    "BatchItem": ("gauss.batch", "BatchItem"),
    # stream
    "AgentStream": ("gauss.stream", "AgentStream"),
    "StreamEvent": ("gauss.stream", "StreamEvent"),
    "parse_partial_json": ("gauss.stream", "parse_partial_json"),
    # code execution
    "execute_code": ("gauss.code_execution", "execute_code"),
    "available_runtimes": ("gauss.code_execution", "available_runtimes"),
    "generate_image": ("gauss.code_execution", "generate_image"),
    # approval / hitl
    "ApprovalManager": ("gauss.approval", "ApprovalManager"),
    "CheckpointStore": ("gauss.checkpoint", "CheckpointStore"),
    # control plane
    "ControlPlane": ("gauss.control_plane", "ControlPlane"),
    # routing policy
    "RoutingPolicy": ("gauss.routing_policy", "RoutingPolicy"),
    "RoutingCandidate": ("gauss.routing_policy", "RoutingCandidate"),
    "RoutingPolicyError": ("gauss.routing_policy", "RoutingPolicyError"),
    "resolve_routing_target": ("gauss.routing_policy", "resolve_routing_target"),
    "resolve_fallback_provider": ("gauss.routing_policy", "resolve_fallback_provider"),
    "enforce_routing_cost_limit": ("gauss.routing_policy", "enforce_routing_cost_limit"),
    # config
    "parse_agent_config": ("gauss.config", "parse_agent_config"),
    "resolve_env": ("gauss.config", "resolve_env"),
    # eval
    "EvalRunner": ("gauss.eval", "EvalRunner"),
    # graph
    "Graph": ("gauss.graph", "Graph"),
    "RouterFn": ("gauss.graph", "RouterFn"),
    # guardrail
    "GuardrailChain": ("gauss.guardrail", "GuardrailChain"),
    # mcp
    "McpServer": ("gauss.mcp", "McpServer"),
    "McpResource": ("gauss.mcp", "McpResource"),
    "McpPrompt": ("gauss.mcp", "McpPrompt"),
    "McpPromptArgument": ("gauss.mcp", "McpPromptArgument"),
    "McpContent": ("gauss.mcp", "McpContent"),
    "McpResourceContent": ("gauss.mcp", "McpResourceContent"),
    "McpPromptMessage": ("gauss.mcp", "McpPromptMessage"),
    "McpPromptResult": ("gauss.mcp", "McpPromptResult"),
    "McpModelHint": ("gauss.mcp", "McpModelHint"),
    "McpModelPreferences": ("gauss.mcp", "McpModelPreferences"),
    "McpSamplingMessage": ("gauss.mcp", "McpSamplingMessage"),
    "McpSamplingRequest": ("gauss.mcp", "McpSamplingRequest"),
    "McpSamplingResponse": ("gauss.mcp", "McpSamplingResponse"),
    # memory
    "Memory": ("gauss.memory", "Memory"),
    # middleware
    "MiddlewareChain": ("gauss.middleware", "MiddlewareChain"),
    # network
    "Network": ("gauss.network", "Network"),
    # team
    "Team": ("gauss.team", "Team"),
    # pipeline
    "pipe": ("gauss.pipeline", "pipe"),
    "map_async": ("gauss.pipeline", "map_async"),
    "map_sync": ("gauss.pipeline", "map_sync"),
    "filter_async": ("gauss.pipeline", "filter_async"),
    "filter_sync": ("gauss.pipeline", "filter_sync"),
    "reduce_async": ("gauss.pipeline", "reduce_async"),
    "reduce_sync": ("gauss.pipeline", "reduce_sync"),
    "compose": ("gauss.pipeline", "compose"),
    "compose_async": ("gauss.pipeline", "compose_async"),
    # plugin
    "PluginRegistry": ("gauss.plugin", "PluginRegistry"),
    # resilience
    "create_fallback_provider": ("gauss.resilience", "create_fallback_provider"),
    "create_circuit_breaker": ("gauss.resilience", "create_circuit_breaker"),
    "create_resilient_provider": ("gauss.resilience", "create_resilient_provider"),
    # retry
    "RetryConfig": ("gauss.retry", "RetryConfig"),
    "retryable": ("gauss.retry", "retryable"),
    "with_retry": ("gauss.retry", "with_retry"),
    # structured
    "structured": ("gauss.structured", "structured"),
    "StructuredConfig": ("gauss.structured", "StructuredConfig"),
    "StructuredResult": ("gauss.structured", "StructuredResult"),
    # telemetry
    "Telemetry": ("gauss.telemetry", "Telemetry"),
    # template
    "template": ("gauss.template", "template"),
    "PromptTemplate": ("gauss.template", "PromptTemplate"),
    "summarize": ("gauss.template", "summarize"),
    "translate": ("gauss.template", "translate"),
    "code_review": ("gauss.template", "code_review"),
    "classify": ("gauss.template", "classify"),
    "extract": ("gauss.template", "extract"),
    # tokens
    "count_tokens": ("gauss.tokens", "count_tokens"),
    "count_tokens_for_model": ("gauss.tokens", "count_tokens_for_model"),
    "count_message_tokens": ("gauss.tokens", "count_message_tokens"),
    "get_context_window_size": ("gauss.tokens", "get_context_window_size"),
    "estimate_cost": ("gauss.tokens", "estimate_cost"),
    # tool_validator
    "ToolValidator": ("gauss.tool_validator", "ToolValidator"),
    # vector_store
    "VectorStore": ("gauss.vector_store", "VectorStore"),
    # text_splitter (M40)
    "TextSplitter": ("gauss.text_splitter", "TextSplitter"),
    "TextChunk": ("gauss.text_splitter", "TextChunk"),
    "split_text": ("gauss.text_splitter", "split_text"),
    # document_loader (M40)
    "LoadedDocument": ("gauss.document_loader", "LoadedDocument"),
    "load_text": ("gauss.document_loader", "load_text"),
    "load_markdown": ("gauss.document_loader", "load_markdown"),
    "load_json": ("gauss.document_loader", "load_json"),
    # workflow
    "Workflow": ("gauss.workflow", "Workflow"),
    # spec
    "AgentSpec": ("gauss.spec", "AgentSpec"),
    "AgentToolSpec": ("gauss.spec", "AgentToolSpec"),
    "SkillSpec": ("gauss.spec", "SkillSpec"),
    "SkillStep": ("gauss.spec", "SkillStep"),
    "SkillParam": ("gauss.spec", "SkillParam"),
    "discover_agents": ("gauss.spec", "discover_agents"),
    # a2a
    "A2aClient": ("gauss.a2a", "A2aClient"),
    "A2aMessage": ("gauss.a2a", "A2aMessage"),
    "AgentCard": ("gauss.a2a", "AgentCard"),
    "AgentCapabilities": ("gauss.a2a", "AgentCapabilities"),
    "AgentSkill": ("gauss.a2a", "AgentSkill"),
    "Artifact": ("gauss.a2a", "Artifact"),
    "Part": ("gauss.a2a", "Part"),
    "Task": ("gauss.a2a", "Task"),
    "TaskState": ("gauss.a2a", "TaskState"),
    "TaskStatus": ("gauss.a2a", "TaskStatus"),
    # errors
    "GaussError": ("gauss.errors", "GaussError"),
    "DisposedError": ("gauss.errors", "DisposedError"),
    "ProviderError": ("gauss.errors", "ProviderError"),
    "ToolExecutionError": ("gauss.errors", "ToolExecutionError"),
    "ValidationError": ("gauss.errors", "ValidationError"),
    # tool_registry
    "ToolRegistry": ("gauss.tool_registry", "ToolRegistry"),
    "ToolRegistryEntry": ("gauss.tool_registry", "ToolRegistryEntry"),
    "ToolRegistryExample": ("gauss.tool_registry", "ToolExample"),
    "ToolSearchResult": ("gauss.tool_registry", "ToolSearchResult"),
    # typed tools (M36)
    "tool": ("gauss.tool", "tool"),
    "TypedToolDef": ("gauss.tool", "TypedToolDef"),
    "create_tool_executor": ("gauss.tool", "create_tool_executor"),
    # mcp client (M37)
    "McpClient": ("gauss.mcp_client", "McpClient"),
    "McpClientConfig": ("gauss.mcp_client", "McpClientConfig"),
    "McpToolResult": ("gauss.mcp_client", "McpToolResult"),
    # pricing (M42)
    "ModelPricing": ("gauss.tokens", "ModelPricing"),
    "set_pricing": ("gauss.tokens", "set_pricing"),
    "get_pricing": ("gauss.tokens", "get_pricing"),
    "clear_pricing": ("gauss.tokens", "clear_pricing"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path)
        # Cache all lazy names from the same module to avoid submodule shadowing
        for k, (mp, a) in _LAZY_IMPORTS.items():
            if mp == module_path:
                globals()[k] = getattr(mod, a)
        return globals()[name]
    raise AttributeError(f"module 'gauss' has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(globals().keys()) + list(_LAZY_IMPORTS.keys())

__all__ = [
    # One-liner
    "gauss",
    "enterprise_preset",
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
    "TOGETHER_DEFAULT",
    "FIREWORKS_DEFAULT",
    "MISTRAL_DEFAULT",
    "PERPLEXITY_DEFAULT",
    "XAI_DEFAULT",
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
    "CostEstimate",
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
    "RouterFn",
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
    # Unified Control Plane
    "ControlPlane",
    # Routing Policy
    "RoutingPolicy",
    "RoutingCandidate",
    "RoutingPolicyError",
    "resolve_routing_target",
    "resolve_fallback_provider",
    "enforce_routing_cost_limit",
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
    "estimate_cost",
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
    # RAG / Text Splitting (M40)
    "TextSplitter",
    "TextChunk",
    "split_text",
    "LoadedDocument",
    "load_text",
    "load_markdown",
    "load_json",
    # Errors
    "GaussError",
    "DisposedError",
    "ProviderError",
    "ToolExecutionError",
    "ValidationError",
]
