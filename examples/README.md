# Gauss Python SDK — Examples

Runnable examples showcasing every major feature of the **gauss-py** SDK.

## Prerequisites

```bash
pip install gauss
```

Set at least one provider API key:

```bash
export OPENAI_API_KEY="sk-..."
# and/or
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AI..."
export GROQ_API_KEY="gsk_..."
export DEEPSEEK_API_KEY="sk-..."
```

## Examples

| #  | File | Topic |
|----|------|-------|
| 01 | [`01_basic_agent.py`](01_basic_agent.py) | Agent, `gauss()` one-liner, `batch()`, streaming |
| 02 | [`02_tools_and_planning.py`](02_tools_and_planning.py) | Tool definitions (`ToolDef`), capabilities query |
| 03 | [`03_team_coordination.py`](03_team_coordination.py) | `Team` with sequential & parallel strategies |
| 04 | [`04_mcp_server.py`](04_mcp_server.py) | `McpServer` — tools, resources, prompts, JSON-RPC |
| 05 | [`05_memory_context.py`](05_memory_context.py) | `Memory` + `VectorStore` for RAG |
| 06 | [`06_full_pipeline.py`](06_full_pipeline.py) | Agent + Memory + Middleware + Guardrails + Telemetry |
| 07 | [`07_a2a_client.py`](07_a2a_client.py) | A2A protocol — discover, ask, tasks |
| 08 | [`08_graph_pipeline.py`](08_graph_pipeline.py) | `Graph` — DAG pipeline with fork nodes |
| 09 | [`09_workflow.py`](09_workflow.py) | `Workflow` — steps with dependencies |
| 10 | [`10_network_delegation.py`](10_network_delegation.py) | `Network` — supervisor-based delegation |
| 11 | [`11_tool_registry.py`](11_tool_registry.py) | `ToolRegistry` — search, tags, examples |
| 12 | [`12_structured_output.py`](12_structured_output.py) | `structured()` — JSON schema extraction |
| 13 | [`13_dx_utilities.py`](13_dx_utilities.py) | `template()`, `pipe()`, `map_async()`, `compose()`, `with_retry()` |
| 14 | [`14_all_providers.py`](14_all_providers.py) | OpenAI, Anthropic, Google, Groq, DeepSeek, Ollama, OpenRouter |

## Running

```bash
# Single example
python examples/01_basic_agent.py

# All examples (stops on first error)
for f in examples/[0-9]*.py; do echo "=== $f ===" && python "$f"; done
```

## Notes

- All examples use `os.environ` for API keys — never hard-code secrets.
- Examples use context managers (`with Agent() as agent:`) for automatic resource cleanup.
- The A2A example (`07`) requires a running A2A-compliant server.
- The Ollama example (`14`) requires a local Ollama instance.
