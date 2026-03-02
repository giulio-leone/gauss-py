<div align="center">

# 🔮 gauss-py

### Rust-powered AI Agent SDK for Python

[![CI](https://github.com/giulio-leone/gauss-py/actions/workflows/ci.yml/badge.svg)](https://github.com/giulio-leone/gauss-py/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gauss-py.svg)](https://pypi.org/project/gauss-py/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Rust-powered • Multi-provider • Enterprise-grade • Plug-and-play DX**

</div>

Gauss gives Python developers a production-grade agent API on top of a native Rust engine.
Build single agents, hierarchical teams, DAG pipelines, MCP/A2A systems, and enterprise workflows with minimal boilerplate.

---

## Install

```bash
pip install gauss-py
```

Set one provider key (auto-detection is built in):

```bash
export OPENAI_API_KEY=sk-...
# or ANTHROPIC_API_KEY / GOOGLE_API_KEY / OPENROUTER_API_KEY / ...
```

---

## Quick Start

### One-liner

```python
from gauss import gauss

print(gauss("Explain retrieval-augmented generation in 3 bullets."))
```

### Full control

```python
from gauss import Agent, OPENAI_DEFAULT

agent = Agent(
    name="assistant",
    model=OPENAI_DEFAULT,  # "gpt-5.2"
    system_prompt="You are a concise senior engineer.",
    temperature=0.2,
)

result = agent.run("Design a clean API for a weather service.")
print(result.text)
agent.destroy()
```

> `run()` is synchronous.
> Use `await agent.arun(...)` for async execution.

---

## Multi-Agent in a Few Lines

### 1) Team.quick() (hierarchical team bootstrap)

```python
from gauss import Team

team = Team.quick("architecture-team", "parallel", [
    {"name": "planner", "instructions": "Break work into milestones."},
    {"name": "implementer", "instructions": "Produce production-ready code."},
    {"name": "reviewer", "instructions": "Find defects and risks."},
])

out = team.run("Implement a resilient webhook ingestion service.")
print(out["finalText"])
team.destroy()
```

### 2) Graph.pipeline() (2-line DAG)

```python
from gauss import Agent, Graph

graph = Graph.pipeline([
    {"node_id": "analyze", "agent": Agent(system_prompt="Analyze requirements")},
    {"node_id": "build", "agent": Agent(system_prompt="Implement solution")},
    {"node_id": "verify", "agent": Agent(system_prompt="Validate outputs")},
])

out = graph.run("Build a typed SDK wrapper around a REST API.")
print(out["final_text"])
graph.destroy()
```

### 3) Network.quick() (swarm delegation bootstrap)

```python
from gauss import Network

network = Network.quick("supervisor", [
    {"name": "supervisor", "instructions": "Delegate to best specialist."},
    {"name": "math-expert", "instructions": "Solve math tasks precisely."},
    {"name": "writer", "instructions": "Write concise user-facing output."},
])

delegated = network.delegate("math-expert", "What is 13 * 7?")
print(delegated)
network.destroy()

# Built-in starter templates
incident = Network.from_template("incident-response")
support = Network.from_template("support-triage")
risk = Network.from_template("fintech-risk-review")
rag_ops = Network.from_template("rag-ops")
incident.destroy()
support.destroy()
risk.destroy()
rag_ops.destroy()
```

---

## Agent DX

### Inline tools with `with_tool()`

```python
from gauss import Agent

agent = (
    Agent(system_prompt="Use tools when useful.")
    .with_tool(
        "sum",
        "Sum two numbers",
        {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
        execute=lambda a, b: str(a + b),
    )
)

out = agent.run("What is 12 + 30?")
print(out.text)
agent.destroy()
```

### Streaming helpers

```python
# Event stream (async iterable)
stream = agent.stream_iter("Write a short release note")

# Minimal DX helper: callback + final text
chunks = []
text = agent.stream_text("Write a changelog", on_delta=chunks.append)
print(text)
```

### Config helpers

```python
# Explicit env-intent constructor
agent = Agent.from_env(system_prompt="Be precise.")

# Clone with a different model
fast_agent = agent.with_model("gpt-4.1")

# Optional routing policy: alias + provider/model target
from gauss import (
    GovernancePolicyPack,
    GovernanceRule,
    ProviderType,
    RoutingPolicy,
    RoutingCandidate,
)

routed_agent = Agent(
    model="fast-chat",
    routing_policy=RoutingPolicy(
        aliases={
            "fast-chat": [
                RoutingCandidate(
                    provider=ProviderType.ANTHROPIC,
                    model="claude-3-5-haiku-latest",
                    priority=10,
                )
            ]
        },
        fallback_order=[ProviderType.ANTHROPIC, ProviderType.OPENAI],
        max_total_cost_usd=2.0,
        max_requests_per_minute=60,
        governance=GovernancePolicyPack(
            rules=[
                GovernanceRule(type="allow_provider", provider=ProviderType.OPENAI),
                GovernanceRule(type="require_tag", tag="pci"),
            ]
        ),
    ),
)

# Runtime policy-router decision (availability + budget + rate + governance tags)
runtime_routed = routed_agent.with_routing_context(
    available_providers=[ProviderType.OPENAI],
    estimated_cost_usd=1.2,
    current_requests_per_minute=20,
    current_hour_utc=11,
    governance_tags=["pci"],
)

# Apply built-in enterprise governance packs
from gauss import apply_governance_pack, explain_routing_target
hardened_policy = apply_governance_pack(
    RoutingPolicy(fallback_order=[ProviderType.ANTHROPIC, ProviderType.OPENAI]),
    "balanced_mix",
)

explanation = explain_routing_target(
    hardened_policy,
    ProviderType.OPENAI,
    "gpt-5.2",
    current_hour_utc=11,
    governance_tags=["balanced"],
)
print(explanation)
```

### Unified Control Plane (M51 foundation)

```python
from gauss import ControlPlane, Telemetry, ApprovalManager

cp = ControlPlane(
    telemetry=Telemetry(),
    approvals=ApprovalManager(),
    model="gpt-5.2",
)
cp.set_cost_usage(1200, 600)
url = cp.start_server(port=0)
print(f"Control Plane: {url}")
# SSE stream:
# single event quick-check -> GET {url}/api/stream?channel=timeline&once=1
# multiplex channels -> GET {url}/api/stream?channels=snapshot,timeline&once=1
# reconnect/replay cursor -> GET {url}/api/stream?channel=snapshot&lastEventId=42
# hosted ops capabilities -> GET {url}/api/ops/capabilities
# hosted ops health -> GET {url}/api/ops/health
# hosted ops summary -> GET {url}/api/ops/summary
# hosted ops tenant breakdown -> GET {url}/api/ops/tenants
# hosted ops dashboard -> GET {url}/ops
# hosted tenant dashboard -> GET {url}/ops/tenants
```

### Async API

```python
# run() is sync
result = agent.run("sync call")

# async variants
result_async = await agent.arun("async call")
text_async = await agent.agenerate("write a haiku")
```

---

## Core Features

- **Agents**: `Agent`, `gauss()`
- **Teams**: `Team`, `Team.quick()`
- **Graphs**: `Graph`, `Graph.pipeline()`, `add_conditional_edge()`
- **Workflows / Networks**: `Workflow`, `Network`, `Network.quick()`, `Network.from_template()`
- **Typed tools**: `@tool`, `create_tool_executor()`, `with_tool()`
- **MCP**: `McpServer`, `McpClient`
- **A2A**: `A2aClient`, `text_message()`, `user_message()`, `agent_message()`, `extract_text()`, `task_text()`
- **Memory + RAG**: `Memory`, `VectorStore`, `TextSplitter`, `load_text/load_markdown/load_json`
- **Guardrails + Middleware**: `GuardrailChain`, `MiddlewareChain`
- **Reliability**: retry, circuit breaker, fallback providers
- **Pipeline helpers**: `pipe`, `map_*`, `filter_*`, `reduce_*`, `tap_async()`
- **Observability & quality**: `Telemetry`, `EvalRunner`
- **Control plane**: `ControlPlane` (local snapshot API + dashboard)
- **Routing policy**: `RoutingPolicy`, `RoutingCandidate`, `resolve_routing_target()`, `apply_governance_pack()`
- **Enterprise preset**: `enterprise_preset()`, `enterprise_run()`

---

## Errors (typed hierarchy)

```python
from gauss import (
    GaussError,
    DisposedError,
    ProviderError,
    ToolExecutionError,
    ValidationError,
)

try:
    agent.run("hello")
except DisposedError:
    print("resource already destroyed")
```

---

## Model Constants

```python
from gauss import (
    OPENAI_DEFAULT, OPENAI_FAST, OPENAI_REASONING,
    ANTHROPIC_DEFAULT, ANTHROPIC_FAST, ANTHROPIC_PREMIUM,
    GOOGLE_DEFAULT, GOOGLE_PREMIUM,
    DEEPSEEK_DEFAULT, DEEPSEEK_REASONING,
    PROVIDER_DEFAULTS, default_model,
)
```

---

## Providers

| Provider | Env Variable | Example Default |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `gpt-5.2` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-20250514` |
| Google | `GOOGLE_API_KEY` | `gemini-2.5-flash` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat` |
| Groq | `GROQ_API_KEY` | provider-dependent |
| Ollama | local runtime | `llama3.2` |
| OpenRouter | `OPENROUTER_API_KEY` | `openai/gpt-5.2` |

---

## Architecture

```text
gauss-py (Python SDK)
        │
        ▼
gauss-python (PyO3 bindings)
        │
        ▼
gauss-core (Rust engine)
```

All heavy orchestration and runtime logic is executed in Rust.

---

## Ecosystem

| Package | Language | Repo |
|---|---|---|
| `gauss-core` | Rust | https://github.com/giulio-leone/gauss-core |
| `gauss-ts` | TypeScript | https://github.com/giulio-leone/gauss |
| `gauss-py` | Python | https://github.com/giulio-leone/gauss-py |

## License

MIT © [Giulio Leone](https://github.com/giulio-leone)
