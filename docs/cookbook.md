# Gauss Python Cookbook

Practical recipes for building AI-powered applications with gauss-py.

---

## Quick Start

### One-liner

```python
from gauss import gauss

answer = gauss("What is the capital of France?")
print(answer)  # "Paris"
```

### Full Agent

```python
from gauss import Agent, AgentConfig, ProviderType

agent = Agent(AgentConfig(
    provider=ProviderType.OPENAI,
    model="gpt-4o",
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
))

result = agent.run("Explain quantum computing")
print(result.text)
```

### Context Manager

```python
with Agent() as agent:
    print(agent.run("Hello!").text)
# Resources auto-released
```

---

## Streaming

### Callback-based

```python
import json

def on_event(event_json: str):
    event = json.loads(event_json)
    if event["type"] == "text_delta":
        print(event["delta"], end="", flush=True)

result = agent.stream("Tell me a story", on_event)
print(f"\n\nFinal: {result.text}")
```

### Async Iterator

```python
async for event in agent.stream_iter("Tell me a joke"):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
```

---

## Tools

### Defining Tools

```python
from gauss import Agent, ToolDef

weather_tool = ToolDef(
    name="get_weather",
    description="Get current weather for a city",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    },
)

agent = Agent()
agent.add_tool(weather_tool)
```

### Custom Tool Executor

```python
import json

def handle_tool(call_json: str) -> str:
    data = json.loads(call_json)
    if data["tool"] == "get_weather":
        return json.dumps({"temp": 22, "unit": "C", "condition": "sunny"})
    return json.dumps({"error": "unknown tool"})

result = agent.run_with_tools("What's the weather in Tokyo?", handle_tool)
print(result.text)
```

### Async Tool Executor

```python
import json

async def handle_tool_async(call_json: str) -> str:
    data = json.loads(call_json)
    if data["tool"] == "search":
        results = await fetch_search_results(data["args"]["query"])
        return json.dumps(results)
    return json.dumps({"error": "unknown tool"})

result = agent.run_with_tools("Search for Python docs", handle_tool_async)
```

---

## Batch Processing

```python
from gauss import batch

items = [
    {"prompt": "Translate 'hello' to Spanish"},
    {"prompt": "Translate 'hello' to French"},
    {"prompt": "Translate 'hello' to Japanese"},
]

results = batch(agent, items, concurrency=3)
for item, result in zip(items, results):
    print(f"{item['prompt']} → {result.text}")
```

---

## Structured Output

```python
from gauss import Agent, AgentConfig

agent = Agent(AgentConfig(
    output_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "skills": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "age", "skills"],
    }
))

result = agent.run("Extract: John is 30, knows Python and Rust")
data = json.loads(result.text)
# {"name": "John", "age": 30, "skills": ["Python", "Rust"]}
```

---

## Teams

```python
from gauss import Team

team = Team("research-team")
team.add_agent(researcher_handle, "researcher")
team.add_agent(writer_handle, "writer")
team.set_strategy("sequential")  # or "parallel", "round_robin"

result = team.run("Write a report on AI trends")
print(result)
```

---

## Graph Pipelines

```python
from gauss import Graph

graph = Graph("pipeline")
graph.add_node("analyze", analyzer_handle)
graph.add_node("summarize", summarizer_handle)
graph.add_edge("analyze", "summarize")

result = graph.run("Raw data to process...")
print(result)
```

---

## Workflows

```python
from gauss import Workflow

workflow = Workflow("content-pipeline")
workflow.add_step("research", researcher_handle)
workflow.add_step("draft", writer_handle)
workflow.add_step("review", reviewer_handle)
workflow.add_dependency("draft", "research")
workflow.add_dependency("review", "draft")

result = workflow.run("Create a blog post about Rust")
print(result)
```

---

## Networks

```python
from gauss import Network

network = Network("multi-agent")
network.add_agent(coder_handle, "coder")
network.add_agent(reviewer_handle, "reviewer")
network.set_supervisor(supervisor_handle)

result = network.delegate("Build a REST API", "coder")
print(result)
```

---

## Memory & RAG

### Conversation Memory

```python
from gauss import Memory

memory = Memory()
memory.store("user", "My name is Alice")
memory.store("assistant", "Hello Alice!")

context = memory.recall("What's my name?")
print(context)  # Returns relevant messages
```

### Vector Store

```python
from gauss import VectorStore

store = VectorStore()
store.upsert([
    {"id": "doc1", "text": "Rust is a systems programming language"},
    {"id": "doc2", "text": "Python is great for data science"},
])

results = store.search("systems programming", top_k=1)
print(results[0]["text"])  # "Rust is a systems programming language"
```

---

## MCP Integration

```python
from gauss import McpServer

server = McpServer("my-tools")
server.add_tool(
    name="calculate",
    description="Evaluate a math expression",
    parameters={"type": "object", "properties": {"expr": {"type": "string"}}},
)

response = server.handle(request_json)
```

---

## Middleware & Guardrails

### Content Filtering

```python
from gauss import GuardrailChain

chain = GuardrailChain()
chain.add_content_moderation()
chain.add_pii_detection()
chain.add_token_limit(4096)
```

### Resilience

```python
from gauss import create_fallback_provider, create_circuit_breaker

fallback = create_fallback_provider([primary_handle, backup_handle])
breaker = create_circuit_breaker(primary_handle, failure_threshold=3, reset_timeout=60)
```

---

## Code Execution

```python
from gauss import execute_code, available_runtimes

# Sandboxed Python execution
result = execute_code("python", 'print("Hello from Python!")')
print(result["stdout"])  # "Hello from Python!"

# Check available runtimes
runtimes = available_runtimes()
print(runtimes)  # ["python", "javascript", "bash"]
```

---

## Eval & Benchmarking

```python
from gauss import EvalRunner

runner = EvalRunner()
runner.add_scorer("accuracy")
dataset = runner.load_dataset_jsonl("test_data.jsonl")

results = runner.run(agent, dataset)
print(f"Accuracy: {results['accuracy']:.2%}")
```

---

## Token Counting

```python
from gauss import count_tokens, get_context_window_size

tokens = count_tokens("Hello, how are you?")
print(tokens)  # ~6

window = get_context_window_size("gpt-4o")
print(window)  # 128000
```

---

## Configuration from Environment

```python
import os

# Auto-detect provider from env vars
os.environ["OPENAI_API_KEY"] = "sk-..."
agent = Agent()  # Automatically uses OpenAI

# Or be explicit
agent = Agent(provider=ProviderType.ANTHROPIC, model="claude-sonnet-4-20250514")
```

---

## Method Chaining

```python
agent = (
    Agent(model="gpt-4o")
    .add_tool(weather_tool)
    .add_tool(search_tool)
    .set_options(temperature=0.8, max_tokens=2048)
)

result = agent.run("What's the weather and latest news?")
```
