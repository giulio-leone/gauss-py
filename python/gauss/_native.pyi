"""Type stubs for the Gauss native Rust extension module (PyO3)."""

from __future__ import annotations

# --- Core ---

def version() -> str: ...

# --- Provider ---

def create_provider(
    provider_type: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    max_retries: int | None = None,
) -> int: ...

def destroy_provider(handle: int) -> None: ...

def get_provider_capabilities(provider_handle: int) -> str: ...

# --- Agent ---

async def agent_run(
    name: str,
    provider_handle: int,
    messages_json: str,
    options_json: str | None = None,
) -> str: ...

async def agent_run_with_tool_executor(
    name: str,
    provider_handle: int,
    messages_json: str,
    options_json: str | None = None,
    tool_executor: object | None = None,
) -> str: ...

async def agent_stream(
    name: str,
    provider_handle: int,
    messages_json: str,
    stream_callback: object,
    options_json: str | None = None,
) -> str: ...

# --- Code Execution / Generation ---

async def generate(
    provider_handle: int,
    messages_json: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    thinking_budget: int | None = None,
    reasoning_effort: str | None = None,
    cache_control: bool | None = None,
) -> str: ...

async def stream_generate(
    provider_handle: int,
    messages_json: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str: ...

async def generate_with_tools(
    provider_handle: int,
    messages_json: str,
    tools_json: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    thinking_budget: int | None = None,
    reasoning_effort: str | None = None,
) -> str: ...

async def execute_code(
    language: str,
    code: str,
    timeout_secs: int | None = None,
    working_dir: str | None = None,
    sandbox: str | None = None,
) -> str: ...

async def available_runtimes() -> str: ...

async def generate_image(
    provider_handle: int,
    prompt: str,
    model: str | None = None,
    size: str | None = None,
    quality: str | None = None,
    style: str | None = None,
    aspect_ratio: str | None = None,
    n: int | None = None,
    response_format: str | None = None,
) -> str: ...

# --- Memory ---

def create_memory() -> int: ...

async def memory_store(handle: int, entry_json: str) -> None: ...

async def memory_recall(
    handle: int,
    options_json: str | None = None,
) -> str: ...

async def memory_clear(
    handle: int,
    session_id: str | None = None,
) -> None: ...

def destroy_memory(handle: int) -> None: ...

async def memory_stats(handle: int) -> str: ...

# --- Tokens ---

def count_tokens(text: str) -> int: ...

def count_tokens_for_model(text: str, model: str) -> int: ...

def count_message_tokens(messages_json: str) -> int: ...

def get_context_window_size(model: str) -> int: ...

# --- Vector Store ---

def create_vector_store() -> int: ...

async def vector_store_upsert(handle: int, chunks_json: str) -> None: ...

async def vector_store_search(
    handle: int,
    embedding_json: str,
    top_k: int,
) -> str: ...

def destroy_vector_store(handle: int) -> None: ...

def cosine_similarity(a_json: str, b_json: str) -> float: ...

# --- MCP ---

def create_mcp_server(name: str, version_str: str) -> int: ...

def mcp_server_add_tool(handle: int, tool_json: str) -> None: ...

def mcp_server_add_resource(handle: int, resource_json: str) -> None: ...

def mcp_server_add_prompt(handle: int, prompt_json: str) -> None: ...

async def mcp_server_handle(handle: int, message_json: str) -> str: ...

def destroy_mcp_server(handle: int) -> None: ...

# --- Network ---

def create_network() -> int: ...

def network_add_agent(
    handle: int,
    name: str,
    provider_handle: int,
    card_json: str | None = None,
    connections: list[str] | None = None,
) -> None: ...

def network_set_supervisor(handle: int, name: str) -> None: ...

async def network_delegate(
    handle: int,
    agent_name: str,
    messages_json: str,
) -> str: ...

def destroy_network(handle: int) -> None: ...

def network_agent_cards(handle: int) -> str: ...

# --- A2A Protocol ---

async def a2a_discover(
    base_url: str,
    auth_token: str | None = None,
) -> str: ...

async def a2a_send_message(
    base_url: str,
    auth_token: str | None = None,
    message_json: str = "",
    config_json: str | None = None,
) -> str: ...

async def a2a_ask(
    base_url: str,
    auth_token: str | None = None,
    text: str = "",
) -> str: ...

async def a2a_get_task(
    base_url: str,
    auth_token: str | None = None,
    task_id: str = "",
    history_length: int | None = None,
) -> str: ...

async def a2a_cancel_task(
    base_url: str,
    auth_token: str | None = None,
    task_id: str = "",
) -> str: ...

# --- Middleware ---

def create_middleware_chain() -> int: ...

def middleware_use_logging(handle: int) -> None: ...

def middleware_use_caching(handle: int, ttl_ms: int) -> None: ...

def destroy_middleware_chain(handle: int) -> None: ...

# --- Guardrails ---

def create_guardrail_chain() -> int: ...

def guardrail_chain_add_content_moderation(
    handle: int,
    block_patterns: list[str],
    warn_patterns: list[str],
) -> None: ...

def guardrail_chain_add_pii_detection(handle: int, action: str) -> None: ...

def guardrail_chain_add_token_limit(
    handle: int,
    max_input: int | None = None,
    max_output: int | None = None,
) -> None: ...

def guardrail_chain_add_regex_filter(
    handle: int,
    block_rules: list[str],
    warn_rules: list[str],
) -> None: ...

def guardrail_chain_add_schema(handle: int, schema_json: str) -> None: ...

def guardrail_chain_list(handle: int) -> list[str]: ...

def destroy_guardrail_chain(handle: int) -> None: ...

# --- Resilience ---

def create_fallback_provider(provider_handles: list[int]) -> int: ...

def create_circuit_breaker(
    provider_handle: int,
    failure_threshold: int | None = None,
    recovery_timeout_ms: int | None = None,
) -> int: ...

def create_resilient_provider(
    primary_handle: int,
    fallback_handles: list[int],
    enable_circuit_breaker: bool | None = None,
) -> int: ...

# --- Stream ---

def py_parse_partial_json(text: str) -> str | None: ...

# --- HITL / Approval ---

def create_approval_manager() -> int: ...

def approval_request(
    handle: int,
    tool_name: str,
    args_json: str,
    session_id: str,
) -> str: ...

def approval_approve(
    handle: int,
    request_id: str,
    modified_args: str | None = None,
) -> None: ...

def approval_deny(
    handle: int,
    request_id: str,
    reason: str | None = None,
) -> None: ...

def approval_list_pending(handle: int) -> str: ...

def destroy_approval_manager(handle: int) -> None: ...

# --- Checkpoint ---

def create_checkpoint_store() -> int: ...

async def checkpoint_save(handle: int, checkpoint_json: str) -> None: ...

async def checkpoint_load(handle: int, checkpoint_id: str) -> str: ...

def destroy_checkpoint_store(handle: int) -> None: ...

async def checkpoint_load_latest(handle: int, session_id: str) -> str: ...

# --- Eval ---

def create_eval_runner(threshold: float | None = None) -> int: ...

def eval_add_scorer(handle: int, scorer_type: str) -> None: ...

def load_dataset_jsonl(jsonl: str) -> str: ...

def load_dataset_json(json_str: str) -> str: ...

def destroy_eval_runner(handle: int) -> None: ...

# --- Telemetry ---

def create_telemetry() -> int: ...

def telemetry_record_span(handle: int, span_json: str) -> None: ...

def telemetry_export_spans(handle: int) -> str: ...

def telemetry_export_metrics(handle: int) -> str: ...

def telemetry_clear(handle: int) -> None: ...

def destroy_telemetry(handle: int) -> None: ...

# --- Graph ---

def create_graph() -> int: ...

def graph_add_node(
    handle: int,
    node_id: str,
    agent_name: str,
    provider_handle: int,
    instructions: str | None = None,
    tools_json: str | None = None,
) -> None: ...

def graph_add_edge(handle: int, from_node: str, to_node: str) -> None: ...

def graph_add_fork_node(
    handle: int,
    node_id: str,
    agents_json: str,
    consensus: str,
) -> None: ...

async def graph_run(handle: int, prompt: str) -> str: ...

def destroy_graph(handle: int) -> None: ...

# --- Workflow ---

def create_workflow() -> int: ...

def workflow_add_step(
    handle: int,
    step_id: str,
    agent_name: str,
    provider_handle: int,
    instructions: str | None = None,
    tools_json: str | None = None,
) -> None: ...

def workflow_add_dependency(handle: int, step_id: str, depends_on: str) -> None: ...

async def workflow_run(handle: int, prompt: str) -> str: ...

def destroy_workflow(handle: int) -> None: ...

# --- Team ---

def create_team(name: str) -> int: ...

def team_add_agent(
    handle: int,
    agent_name: str,
    provider_handle: int,
    instructions: str | None = None,
) -> None: ...

def team_set_strategy(handle: int, strategy: str) -> None: ...

async def team_run(handle: int, messages_json: str) -> str: ...

def destroy_team(handle: int) -> None: ...

# --- Plugin ---

def create_plugin_registry() -> int: ...

def plugin_registry_add_telemetry(handle: int) -> None: ...

def plugin_registry_add_memory(handle: int) -> None: ...

def plugin_registry_list(handle: int) -> list[str]: ...

def plugin_registry_emit(handle: int, event_json: str) -> None: ...

def destroy_plugin_registry(handle: int) -> None: ...

# --- Tool Validator ---

def create_tool_validator(strategies: list[str] | None = None) -> int: ...

def tool_validator_validate(handle: int, input: str, schema: str) -> str: ...

def destroy_tool_validator(handle: int) -> None: ...

# --- Tool Registry ---

def create_tool_registry() -> int: ...

def tool_registry_add(handle: int, tool_json: str) -> None: ...

def tool_registry_search(handle: int, query: str) -> str: ...

def tool_registry_by_tag(handle: int, tag: str) -> str: ...

def tool_registry_list(handle: int) -> str: ...

def destroy_tool_registry(handle: int) -> None: ...

# --- Config ---

def agent_config_from_json(json_str: str) -> str: ...

def agent_config_resolve_env(value: str) -> str: ...

def parse_agents_md(content: str) -> str: ...

def discover_agents(dir: str) -> str: ...

def parse_skill_md(content: str) -> str: ...
