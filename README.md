# 🧠 Gauss AI

[![PyPI version](https://img.shields.io/pypi/v/gauss-py)](https://pypi.org/project/gauss-py/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Production-grade AI agents powered by Rust.**
> Zero config · Instant setup · Maximum performance via PyO3 bindings.

---

## Install

```bash
pip install gauss-py
```

## Quick Start — One Line

```python
from gauss import gauss

answer = gauss("Explain quantum computing in 3 sentences")
print(answer)
```

That's it. Auto-detects your API key from environment variables.

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

## Streaming

```python
import asyncio
from gauss import Agent

agent = Agent()

async def main():
    async for event in agent.stream_iter("Tell me a story"):
        if event.type == "text_delta":
            print(event.text, end="", flush=True)

asyncio.run(main())
```

## Batch Processing

Run multiple prompts in parallel with concurrency control:

```python
from gauss import batch

results = batch(
    ["Translate: Hello", "Translate: World", "Translate: Goodbye"],
    concurrency=2,
)
for r in results:
    print(r.result.text if r.result else r.error)
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

## Multi-Agent Network

```python
from gauss import Agent, Network

analyst = Agent(name="analyst")
coder = Agent(name="coder")

net = (
    Network()
    .add_agent(analyst)
    .add_agent(coder)
    .set_supervisor("analyst")
)

result = net.delegate("coder", "Implement a sorting algorithm")
```

## Retry with Backoff

```python
from gauss import Agent, with_retry, retryable, RetryConfig

agent = Agent()

# Wrap any callable:
result = with_retry(
    lambda: agent.run("Summarize this"),
    config=RetryConfig(
        max_retries=3,
        backoff="exponential",  # "fixed" | "linear" | "exponential"
        base_delay_s=1.0,
        jitter=0.1,
        on_retry=lambda err, attempt, delay: print(f"Retry {attempt} in {delay}s"),
    ),
)

# Or wrap an agent:
resilient_run = retryable(agent, max_retries=5)
result = resilient_run("Hello")
```

## Structured Output

Extract typed JSON from LLM responses with auto-retry on parse failure:

```python
from gauss import Agent, structured

agent = Agent()

result = structured(agent, "List 3 programming languages", schema={
    "type": "object",
    "properties": {
        "languages": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["languages"],
})

print(result.data["languages"])  # ["TypeScript", "Rust", "Python"]
```

## Prompt Templates

Composable prompt construction with `{{variable}}` placeholders:

```python
from gauss import template, summarize, translate, code_review

# Custom template:
greet = template("Hello {{name}}, you are a {{role}}.")
print(greet(name="Alice", role="developer"))

# Built-in templates:
prompt = summarize(format="article", style="bullet points", text="...")
translated = translate(language="French", text="Hello world")
review = code_review(language="python", code="x = 1")

# Composition:
with_tone = template("{{base}}\n\nUse a {{tone}} tone.")
prompt2 = with_tone(
    base=summarize(format="report", style="concise", text="..."),
    tone="professional",
)
```

## Pipeline & Async Helpers

Compose agent operations into clean data flows:

```python
import asyncio
from gauss import pipe, map_sync, map_async, filter_sync, reduce_sync, compose

agent = Agent()

# Pipe: chain async steps
result = asyncio.run(pipe(
    "Explain AI",
    lambda prompt: agent.run(prompt),
    lambda result: result.text.upper(),
))

# Map: process items with concurrency
descriptions = map_sync(
    ["apple", "banana", "cherry"],
    lambda fruit: agent.run(f"Describe {fruit}"),
    concurrency=2,
)

# Filter: keep items matching predicate
long_ones = filter_sync(
    descriptions,
    lambda r: len(r.text) > 100,
)

# Reduce: sequential aggregation
summary = reduce_sync(
    documents,
    lambda acc, doc: agent.run(f"Combine:\n{acc}\n\nNew:\n{doc}").text,
    "",
)

# Compose: build reusable transforms
enhance = compose(
    lambda text: f"[System] {text}",
    lambda text: text.strip(),
)
```

## RAG (Vector Store)

```python
from gauss import VectorStore

store = VectorStore()
store.upsert([{"id": "c1", "text": "Rust is fast", "embedding": [0.1, 0.2, 0.3]}])
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

## Middleware

```python
from gauss import MiddlewareChain

chain = (
    MiddlewareChain()
    .use_logging()
    .use_caching(ttl_ms=60_000)
)
```

## MCP Server

```python
from gauss import McpServer, ToolDef

server = McpServer("my-tools", "1.0.0")
server.add_tool(ToolDef(name="search", description="Web search"))
response = server.handle_message({"jsonrpc": "2.0", "method": "tools/list"})
```

## Resilience

```python
from gauss import create_fallback_provider, create_circuit_breaker

fallback = create_fallback_provider([
    {"provider": "openai", "model": "gpt-4o"},
    {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
])

breaker = create_circuit_breaker(failure_threshold=5, reset_timeout_ms=30000)
```

## All Features

| Feature | Module | Description |
|---------|--------|-------------|
| **Agent** | `Agent`, `gauss()` | LLM agent with tools, structured output, streaming |
| **Streaming** | `AgentStream` | Async iterable streaming |
| **Batch** | `batch()` | Parallel prompt execution with concurrency control |
| **Graph** | `Graph` | DAG-based multi-agent pipeline |
| **Workflow** | `Workflow` | Step-based execution with dependencies |
| **Network** | `Network` | Multi-agent delegation with supervisor |
| **Memory** | `Memory` | Persistent conversation memory |
| **VectorStore** | `VectorStore` | Embedding storage and semantic search |
| **Middleware** | `MiddlewareChain` | Request/response processing pipeline |
| **Guardrails** | `GuardrailChain` | Content moderation, PII, token limits, regex |
| **Retry** | `with_retry`, `retryable` | Exponential/linear/fixed backoff with jitter |
| **Structured** | `structured()` | Typed JSON extraction with auto-retry |
| **Templates** | `template()` | Composable prompt templates with built-ins |
| **Pipeline** | `pipe`, `map_sync`, `compose` | Data flow composition |
| **Evaluation** | `EvalRunner` | Agent quality scoring with datasets |
| **Telemetry** | `Telemetry` | Spans, metrics, and export |
| **Approval** | `ApprovalManager` | Human-in-the-loop approval flow |
| **Checkpoint** | `CheckpointStore` | Save/restore agent state |
| **MCP** | `McpServer` | Model Context Protocol server |
| **Resilience** | `create_fallback_provider` | Fallback and circuit breaker |
| **Tokens** | `count_tokens` | Token counting and context window info |
| **Plugins** | `PluginRegistry` | Extensible plugin system |

## Supported Providers

| Provider | Environment Variable |
|----------|---------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Google | `GOOGLE_API_KEY` |
| Groq | `GROQ_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Ollama | `OLLAMA_HOST` |

## Architecture

Gauss-py is a thin Python SDK wrapping **[gauss-core](https://github.com/giulio-leone/gauss-core)** (Rust) via PyO3 bindings. All heavy lifting runs at native Rust speed.

```
Python SDK (24 modules)
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
| [`gauss-py`](https://github.com/giulio-leone/gauss) | TypeScript | TypeScript SDK (NAPI bindings) |
| [`gauss-py`](https://github.com/giulio-leone/gauss-py) | Python | This SDK (PyO3 bindings) |

## License

MIT © [Giulio Leone](https://github.com/giulio-leone)
