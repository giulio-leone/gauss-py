"""DAG-based agent graph for multi-step pipelines."""

from __future__ import annotations

import json
from typing import Any

from gauss._types import ToolDef


class Graph:
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
        from gauss._native import create_graph

        self._handle: int = create_graph()
        self._destroyed = False

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
        from gauss._native import graph_run
        from gauss.agent import _run_native

        self._check_alive()
        result_json = _run_native(graph_run, self._handle, prompt)
        return json.loads(result_json)  # type: ignore[no-any-return]

    def destroy(self) -> None:
        """Release native resources."""
        if not self._destroyed:
            from gauss._native import destroy_graph

            destroy_graph(self._handle)
            self._destroyed = True

    def __enter__(self) -> Graph:
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    def __del__(self) -> None:
        self.destroy()

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("Graph has been destroyed")
