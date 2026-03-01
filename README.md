<div align="center">

# 🔮 gauss-py

**The AI Agent SDK for Python — powered by Rust**

[![CI](https://github.com/giulio-leone/gauss-py/actions/workflows/ci.yml/badge.svg)](https://github.com/giulio-leone/gauss-py/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gauss-py.svg)](https://pypi.org/project/gauss-py/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Multi-provider • Teams • Tools • MCP • Graphs • Workflows • Memory • RAG**

</div>

---

## Quick Start

```bash
pip install gauss-py
```

```python
from gauss import Agent

agent = Agent(
    name="assistant",
    model="gpt-5.2",
    instructions="You are a helpful assistant.",
)

result = await agent.run("What is quantum computing?")
print(result.text)
```

## Features

### 🤖 Agents

```python
from gauss import Agent, OPENAI_DEFAULT

agent = Agent(
    name="researcher",
    model=OPENAI_DEFAULT,  # "gpt-5.2"
    instructions="You are a research assistant.",
    temperature=0.7,
    max_steps=5,
)

result = await agent.run("Find the population of Tokyo")
print(f"{result.text}")
print(f"Steps: {result.steps}, Tokens: {result.input_tokens + result.output_tokens}")

# Stream
async for event in agent.stream("Tell me a story"):
    if event.type == "text_delta":
        print(event.delta, end="")
```

### 🛠️ Tools

```python
from gauss import Agent, ToolDef

weather_tool = ToolDef(
    name="get_weather",
    description="Get current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    },
    handler=lambda args: {"temperature": 22, "condition": "sunny"},
)

agent = Agent(
    name="weather-bot",
    model="gpt-5.2",
    tools=[weather_tool],
)
```

### 👥 Teams

```python
from gauss import Agent, Team

team = Team(
    name="research-team",
    agents=[
        Agent(name="researcher", model="gpt-5.2", instructions="Research deeply."),
        Agent(name="writer", model="claude-sonnet-4-20250514", instructions="Write clearly."),
        Agent(name="critic", model="gemini-2.5-flash", instructions="Review and critique."),
    ],
    strategy="round-robin",
)

result = await team.run("Analyze the impact of AI on healthcare")
```

### 🧠 Memory

```python
from gauss import Agent, Memory

memory = Memory()
agent = Agent(name="assistant", model="gpt-5.2", memory=memory)

await agent.run("My name is Alice and I love hiking.")
result = await agent.run("What do you know about me?")
# → "You're Alice, and you enjoy hiking!"
```

### 🔗 MCP Server

```python
from gauss import McpServer

server = McpServer("my-tools", "1.0.0")
server.add_tool(
    name="calculate",
    description="Evaluate a math expression",
    parameters={"type": "object", "properties": {"expr": {"type": "string"}}},
    handler=lambda args: {"result": eval(args["expr"])},
)
```

### 📊 Graph Pipelines

```python
from gauss import Graph, Agent

graph = Graph("content-pipeline")
graph.add_node("research", research_agent)
graph.add_node("write", writer_agent)
graph.add_node("review", review_agent)
graph.add_edge("research", "write")
graph.add_edge("write", "review")

result = await graph.run("Write an article about quantum computing")
```

### 🔄 Workflows

```python
from gauss import Workflow, Agent

workflow = Workflow("analysis-pipeline")
workflow.add_step("collect", collector_agent)
workflow.add_step("analyze", analyst_agent)
workflow.add_step("summarize", summarizer_agent)

result = await workflow.run("Analyze market trends in AI")
```

### 🌐 Networks

```python
from gauss import Network, Agent

network = Network()
network.add_agent(Agent(name="math", model="o4-mini", instructions="Solve math."))
network.add_agent(Agent(name="code", model="gpt-5.2", instructions="Write code."))
network.set_supervisor("coordinator")

result = await network.delegate("math", "What is the integral of x²?")
```

### 🎯 Structured Output

```python
agent = Agent(
    name="extractor",
    model="gpt-5.2",
    output_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "skills": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "age", "skills"],
    },
)

result = await agent.run("Extract: John is 30, knows Python and Rust")
print(result.structured_output)
# → {"name": "John", "age": 30, "skills": ["Python", "Rust"]}
```

### 💭 Reasoning

```python
from gauss import Agent, OPENAI_REASONING, ANTHROPIC_PREMIUM

# OpenAI reasoning models (o4-mini)
reasoner = Agent(
    name="solver",
    model=OPENAI_REASONING,
    reasoning_effort="high",
)

# Anthropic extended thinking
thinker = Agent(
    name="analyst",
    model=ANTHROPIC_PREMIUM,
    thinking_budget=10000,
)

result = await thinker.run("Analyze this complex problem...")
print(result.thinking)  # Internal reasoning process
```

## 📐 Model Constants

Import model constants as the single source of truth:

```python
from gauss import (
    OPENAI_DEFAULT,      # "gpt-5.2"
    OPENAI_FAST,         # "gpt-4.1"
    OPENAI_REASONING,    # "o4-mini"
    OPENAI_IMAGE,        # "gpt-image-1"
    ANTHROPIC_DEFAULT,   # "claude-sonnet-4-20250514"
    ANTHROPIC_PREMIUM,   # "claude-opus-4-20250414"
    ANTHROPIC_FAST,      # "claude-haiku-4-20250414"
    GOOGLE_DEFAULT,      # "gemini-2.5-flash"
    GOOGLE_PREMIUM,      # "gemini-2.5-pro"
    DEEPSEEK_DEFAULT,    # "deepseek-chat"
    DEEPSEEK_REASONING,  # "deepseek-reasoner"
    PROVIDER_DEFAULTS,   # Provider → model mapping
    default_model,       # Get default model for a provider
)
```

## 🌐 Providers

| Provider | Env Variable | Models |
|----------|-------------|--------|
| OpenAI | `OPENAI_API_KEY` | gpt-5.2, gpt-4.1, o4-mini |
| Anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-4, claude-opus-4, claude-haiku-4 |
| Google | `GOOGLE_API_KEY` | gemini-2.5-flash, gemini-2.5-pro |
| DeepSeek | `DEEPSEEK_API_KEY` | deepseek-chat, deepseek-reasoner |
| Groq | `GROQ_API_KEY` | Any Groq-supported model |
| Ollama | — (local) | Any Ollama model |
| OpenRouter | `OPENROUTER_API_KEY` | 200+ models |

## 📁 Examples

See the [`examples/`](examples/) directory for 15 complete examples covering every feature.

## 🔗 Related

| Package | Language | Repository |
|---------|----------|------------|
| **gauss-core** | Rust | [giulio-leone/gauss-core](https://github.com/giulio-leone/gauss-core) |
| **gauss-ts** | TypeScript | [giulio-leone/gauss](https://github.com/giulio-leone/gauss) |

## License

MIT
