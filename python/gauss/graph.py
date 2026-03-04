"""DAG-based agent graph for multi-step pipelines."""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from typing import Any

from gauss._types import ToolDef
from gauss.base import StatefulResource

# Type alias for router functions used in conditional edges.
RouterFn = Callable[[dict[str, Any]], str]


class Graph(StatefulResource):
    """Agent computation graph — DAG-based multi-agent pipeline.

    Example::

        from gauss import Agent, Graph

        researcher = Agent(name="researcher", instructions="Research the topic")
        writer = Agent(name="writer", instructions="Write based on research")

        graph = (
            Graph()
            .add_node("research", researcher)
            .add_node("write", writer)
            .add_edge("research", "write")
        )
        result = graph.run("Explain quantum computing")
    """

    def __init__(self) -> None:
        super().__init__()
        from gauss._native import create_graph

        self._handle: int = create_graph()

        # SDK-level bookkeeping for conditional routing.
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: dict[str, str] = {}
        self._conditional_edges: dict[str, RouterFn] = {}

    @functools.cached_property
    def _resource_name(self) -> str:
        return "Graph"

    def add_node(
        self,
        node_id: str,
        agent: Any,
        *,
        instructions: str | None = None,
        tools: list[ToolDef | dict[str, Any]] | None = None,
    ) -> Graph:
        """Add a node (agent) to the graph. Returns self for chaining.

        Args:
            node_id: Unique identifier for this node.
            agent: An Agent instance.
            instructions: Optional override instructions.
            tools: Optional list of tools for this node.
        """
        from gauss._native import graph_add_node

        self._check_alive()
        tools_json: str | None = None
        if tools:
            tool_list = []
            for t in tools:
                if isinstance(t, dict):
                    tool_list.append(t)
                else:
                    tool_list.append(t.to_dict())
            tools_json = json.dumps(tool_list)
        graph_add_node(
            self._handle,
            node_id,
            agent._config.name,
            agent.handle,
            instructions,
            tools_json,
        )
        self._nodes[node_id] = {"agent": agent, "instructions": instructions}
        return self

    def add_edge(self, from_node: str, to_node: str) -> Graph:
        """Add a directed edge between nodes. Returns self for chaining.

        Args:
            from_node: Source node ID.
            to_node: Target node ID.
        """
        from gauss._native import graph_add_edge

        self._check_alive()
        graph_add_edge(self._handle, from_node, to_node)
        self._edges[from_node] = to_node
        return self

    def add_conditional_edge(self, from_node: str, router: RouterFn) -> Graph:
        """Add a conditional edge — router decides the next node at runtime.

        The router function receives the source node's output dict and must
        return the ID of the next node to execute.

        Args:
            from_node: Source node ID.
            router: Callable that maps a node result dict to the next node ID.
        """
        self._check_alive()
        self._conditional_edges[from_node] = router
        return self

    def add_fork(
        self,
        node_id: str,
        agents: list[Any],
        *,
        consensus: str = "concat",
    ) -> Graph:
        """Add a fork node — runs agents in parallel, merging via consensus.

        Args:
            node_id: Unique identifier for this fork node.
            agents: List of Agent instances or dicts with 'agent' and optional 'instructions'.
            consensus: Merge strategy — 'first' or 'concat' (default: 'concat').

        Example::

            graph.add_fork("parallel", [agent_a, agent_b], consensus="first")
        """
        from gauss._native import graph_add_fork_node

        self._check_alive()
        agent_defs = []
        for a in agents:
            if isinstance(a, dict):
                agent_defs.append({
                    "agent_name": a["agent"]._config.name,
                    "provider_handle": a["agent"].handle,
                    "instructions": a.get("instructions"),
                })
            else:
                agent_defs.append({
                    "agent_name": a._config.name,
                    "provider_handle": a.handle,
                })
        graph_add_fork_node(self._handle, node_id, json.dumps(agent_defs), consensus)
        return self

    def run(self, prompt: str) -> dict[str, Any]:
        """Execute the graph pipeline.

        Args:
            prompt: The initial prompt to feed the entry nodes.

        Returns:
            A dict with ``outputs`` (per-node results) and ``final_text``.
        """
        self._check_alive()

        # Fast path: no conditional edges → delegate to Rust core.
        if not self._conditional_edges:
            from gauss._native import graph_run
            from gauss.agent import _run_native

            result_json = _run_native(graph_run, self._handle, prompt)
            return json.loads(result_json)  # type: ignore[no-any-return]

        return self._run_with_conditionals(prompt)

    def destroy(self) -> None:
        """Release native resources."""
        if not self._destroyed:
            from gauss._native import destroy_graph

            destroy_graph(self._handle)
        super().destroy()

    # ── Private ─────────────────────────────────────────────────────

    def _run_with_conditionals(self, prompt: str) -> dict[str, Any]:
        """SDK-level step-through execution when conditional edges are present."""
        # Determine entry node: a node with no incoming edges.
        targets = set(self._edges.values())
        entry_nodes = [n for n in self._nodes if n not in targets]
        if not entry_nodes:
            raise RuntimeError(
                "Graph has no entry node (every node has an incoming edge)"
            )

        outputs: dict[str, dict[str, Any]] = {}
        current_node: str | None = entry_nodes[0]
        current_prompt = prompt

        while current_node is not None:
            node_cfg = self._nodes.get(current_node)
            if node_cfg is None:
                raise RuntimeError(f'Node "{current_node}" not found in graph')

            agent = node_cfg["agent"]
            instructions = node_cfg["instructions"]
            agent_input = (
                f"{instructions}\n\n{current_prompt}" if instructions else current_prompt
            )

            result = agent.run(agent_input)
            node_output: dict[str, Any] = {"text": result.text}
            outputs[current_node] = node_output

            # Decide next node.
            router = self._conditional_edges.get(current_node)
            if router is not None:
                current_node = router(node_output)
            else:
                current_node = self._edges.get(current_node)

            # Feed previous output as prompt for the next node.
            current_prompt = result.text

        # Build result envelope matching graph_run shape.
        node_ids = list(outputs.keys())
        last_node_id = node_ids[-1] if node_ids else None
        return {
            "outputs": outputs,
            "final_text": outputs[last_node_id]["text"] if last_node_id else "",
        }

    @classmethod
    def pipeline(cls, nodes: list[dict[str, Any]]) -> Graph:
        """Quick graph builder — create a linear pipeline.

        Example::

            result = Graph.pipeline([
                {"node_id": "research", "agent": researcher},
                {"node_id": "write", "agent": writer},
            ]).run("Explain quantum computing")
        """
        graph = cls()
        for node in nodes:
            graph.add_node(**node)
        for i in range(len(nodes) - 1):
            graph.add_edge(nodes[i]["node_id"], nodes[i + 1]["node_id"])
        return graph

    def _check_alive(self) -> None:
        if self._destroyed:
            from gauss.errors import DisposedError

            raise DisposedError("Graph", "graph")
