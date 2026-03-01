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
    Message,
    ProviderType,
    SearchResult,
    ToolDef,
)
from gauss.agent import Agent, AgentStream, BatchItem, StreamEvent, batch, gauss
from gauss.approval import ApprovalManager
from gauss.checkpoint import CheckpointStore
from gauss.config import parse_agent_config, resolve_env
from gauss.eval import EvalRunner
from gauss.graph import Graph
from gauss.guardrail import GuardrailChain
from gauss.mcp import McpServer
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
from gauss.stream import parse_partial_json
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

__all__ = [
    # One-liner
    "gauss",
    "batch",
    "BatchItem",
    # Streaming
    "AgentStream",
    "StreamEvent",
    # Core
    "Agent",
    "AgentConfig",
    "AgentResult",
    "Citation",
    "Message",
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
]
