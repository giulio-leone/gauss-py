"""Step-based workflow with dependency resolution."""

from __future__ import annotations

import json
from typing import Any

from gauss._types import ToolDef


class Workflow:
    """Execute agents as sequential/parallel workflow steps.

    Example::

        from gauss import Agent, Workflow

        planner = Agent(name="planner", instructions="Create a plan")
        executor = Agent(name="executor", instructions="Execute the plan")

        wf = (
            Workflow()
            .add_step("plan", planner)
            .add_step("execute", executor)
            .add_dependency("execute", "plan")
        )
        result = wf.run("Build a REST API")
    """

    def __init__(self) -> None:
        from gauss._native import create_workflow

        self._handle: int = create_workflow()
        self._destroyed = False

    def add_step(
        self,
        step_id: str,
        agent: Any,
        *,
        instructions: str | None = None,
        tools: list[ToolDef | dict[str, Any]] | None = None,
    ) -> Workflow:
        """Add a step to the workflow. Returns self for chaining.

        Args:
            step_id: Unique identifier for this step.
            agent: An Agent instance.
            instructions: Optional override instructions.
            tools: Optional list of tools for this step.
        """
        from gauss._native import workflow_add_step

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
        workflow_add_step(
            self._handle,
            step_id,
            agent._config.name,
            agent.handle,
            instructions,
            tools_json,
        )
        return self

    def add_dependency(self, step_id: str, depends_on: str) -> Workflow:
        """Declare that *step_id* depends on *depends_on*. Returns self for chaining.

        Args:
            step_id: The step that has a dependency.
            depends_on: The step it depends on.
        """
        from gauss._native import workflow_add_dependency

        self._check_alive()
        workflow_add_dependency(self._handle, step_id, depends_on)
        return self

    def run(self, prompt: str) -> dict[str, Any]:
        """Execute the workflow.

        Args:
            prompt: The initial prompt.

        Returns:
            A dict with ``steps`` containing per-step results.
        """
        from gauss._native import workflow_run
        from gauss.agent import _run_native

        self._check_alive()
        result_json = _run_native(workflow_run, self._handle, prompt)
        return json.loads(result_json)  # type: ignore[no-any-return]

    def destroy(self) -> None:
        """Release native resources."""
        if not self._destroyed:
            from gauss._native import destroy_workflow

            destroy_workflow(self._handle)
            self._destroyed = True

    def __enter__(self) -> Workflow:
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    def __del__(self) -> None:
        self.destroy()

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("Workflow has been destroyed")
