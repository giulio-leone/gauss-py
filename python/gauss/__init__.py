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
    Message,
    ProviderType,
    SearchResult,
    ToolDef,
)
from gauss.agent import Agent, BatchItem, batch, gauss
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
from gauss.plugin import PluginRegistry
from gauss.resilience import (
    create_circuit_breaker,
    create_fallback_provider,
    create_resilient_provider,
)
from gauss.stream import parse_partial_json
from gauss.telemetry import Telemetry
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
    # Core
    "Agent",
    "AgentConfig",
    "AgentResult",
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
]
