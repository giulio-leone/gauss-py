# 🧠 Gauss AI

**Production-grade AI agents powered by Rust.**

Gauss gives you the full power of a Rust AI engine with a beautiful Python API. Zero config, instant setup, maximum performance.

## Quick Start

```bash
pip install gauss-ai
```

```python
from gauss import gauss

# One-liner — auto-detects provider from env
answer = gauss("What is the meaning of life?")
print(answer)
```

## Agent

```python
from gauss import Agent, AgentConfig, ProviderType

# Auto-detect provider from OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
agent = Agent()
result = agent.run("Explain quantum computing")
print(result.text)

# Explicit config
agent = Agent(AgentConfig(
    provider=ProviderType.ANTHROPIC,
    model="claude-sonnet-4-20250514",
    system_prompt="You are a physicist.",
    temperature=0.7,
))

# Context manager for auto-cleanup
with Agent() as agent:
    print(agent.run("Hello!").text)
```

## Multi-Agent Graph

```python
import asyncio
from gauss import Agent, Graph

researcher = Agent(name="researcher", system_prompt="You research topics deeply")
writer = Agent(name="writer", system_prompt="You write clear articles")

async def main():
    result = await (
        Graph()
        .add_node("research", researcher)
        .add_node("write", writer)
        .add_edge("research", "write")
        .run("Write about AI safety")
    )
    print(result)

asyncio.run(main())
```

## Workflow

```python
import asyncio
from gauss import Agent, Workflow

async def main():
    result = await (
        Workflow()
        .add_step("analyze", Agent(name="analyzer"), instructions="Analyze the data")
        .add_step("report", Agent(name="reporter"))
        .add_dependency("report", "analyze")
        .run("Q3 sales data")
    )
    print(result)

asyncio.run(main())
```

## RAG (Vector Store)

```python
from gauss import VectorStore, Chunk

store = VectorStore()
store.upsert([Chunk(id="c1", text="Rust is fast", embedding=[0.1, 0.2, 0.3])])
results = store.search([0.1, 0.2, 0.3], top_k=5)
```

## Guardrails

```python
from gauss import GuardrailChain

chain = (
    GuardrailChain()
    .add_content_moderation(blocked_categories=["violence"])
    .add_pii_detection(action="redact")
    .add_token_limit(max_input=4000)
)
```

## MCP Server

```python
from gauss import McpServer, ToolDef

server = McpServer("my-tools", "1.0.0")
server.add_tool(ToolDef(name="search", description="Web search"))
response = server.handle_message({"jsonrpc": "2.0", "method": "tools/list"})
```

## Middleware

```python
from gauss import MiddlewareChain

chain = (
    MiddlewareChain()
    .use_logging()
    .use_caching(ttl_ms=60_000)
)
```

## Features

| Feature | Module |
|---------|--------|
| AI Agents | `Agent`, `gauss()` |
| Multi-Agent | `Network`, `Graph`, `Workflow` |
| Memory | `Memory` |
| RAG | `VectorStore` |
| MCP | `McpServer` |
| Guardrails | `GuardrailChain` |
| Middleware | `MiddlewareChain` |
| Eval | `EvalRunner` |
| Telemetry | `Telemetry` |
| HITL | `ApprovalManager`, `CheckpointStore` |
| Plugins | `PluginRegistry` |
| Resilience | `create_fallback_provider`, `create_circuit_breaker` |
| Tokens | `count_tokens`, `get_context_window_size` |

## Supported Providers

- OpenAI (`OPENAI_API_KEY`)
- Anthropic (`ANTHROPIC_API_KEY`)
- Google (`GOOGLE_API_KEY`)
- Groq (`GROQ_API_KEY`)
- DeepSeek (`DEEPSEEK_API_KEY`)
- Ollama (`OLLAMA_HOST`)

## Architecture

Gauss-py is a thin Python SDK wrapping **[gauss-core](https://github.com/giulio-leone/gauss-core)** (Rust) via PyO3 bindings. All heavy lifting runs at native Rust speed.

```
Python SDK (20 modules)
       │
       ▼
  PyO3 Bindings (87+ functions)
       │
       ▼
  gauss-core (Rust engine)
```

## Ecosystem

| Package | Language | Description |
|---------|----------|-------------|
| [`gauss-core`](https://github.com/giulio-leone/gauss-core) | Rust | Core engine — NAPI + PyO3 + WASM |
| [`gauss`](https://github.com/giulio-leone/gauss) | TypeScript | TypeScript SDK (NAPI bindings) |
| [`gauss-py`](https://github.com/giulio-leone/gauss-py) | Python | This SDK (PyO3 bindings) |

## License

MIT
