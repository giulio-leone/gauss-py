"""Team — multi-agent coordination backed by Rust core.

Example::

    from gauss import Agent, Team

    researcher = Agent(name="researcher", instructions="Research topics")
    writer = Agent(name="writer", instructions="Write summaries")

    team = Team("content-team")
    team.add(researcher)
    team.add(writer)
    team.strategy("sequential")

    result = team.run("Explain quantum computing")
    print(result["finalText"])
"""

from __future__ import annotations

import functools
import json
from typing import Any, Literal

from gauss.base import StatefulResource

TeamStrategy = Literal["sequential", "parallel"]


class Team(StatefulResource):
    """Multi-agent team with sequential or parallel coordination.

    Args:
        name: Team name.

    Example::

        team = (
            Team("my-team")
            .add(agent_a)
            .add(agent_b)
            .strategy("parallel")
        )
        result = team.run("Do the thing")
    """

    def __init__(self, name: str) -> None:
        super().__init__()
        from gauss._native import create_team

        self._handle: int = create_team(name)
        self._name = name

    @functools.cached_property
    def _resource_name(self) -> str:
        return "Team"

    @property
    def handle(self) -> int:
        return self._handle

    def add(self, agent: Any, *, instructions: str | None = None) -> Team:
        """Add an agent to the team. Returns self for chaining.

        Args:
            agent: An Agent instance.
            instructions: Optional override instructions for this agent in the team.
        """
        from gauss._native import team_add_agent

        self._check_alive()
        team_add_agent(self._handle, agent._config.name, agent.handle, instructions)
        return self

    def strategy(self, s: TeamStrategy) -> Team:
        """Set coordination strategy. Returns self for chaining.

        Args:
            s: Either ``"sequential"`` (agents run in order, each sees previous output)
               or ``"parallel"`` (agents run concurrently, results merged).
        """
        from gauss._native import team_set_strategy

        self._check_alive()
        team_set_strategy(self._handle, s)
        return self

    def run(self, prompt: str) -> dict[str, Any]:
        """Run the team with an initial prompt.

        Args:
            prompt: The user prompt to process.

        Returns:
            A dict with ``finalText`` and ``results`` (per-agent outputs).
        """
        from gauss._native import team_run
        from gauss.agent import _run_native

        self._check_alive()
        messages = json.dumps([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
        result_json = _run_native(team_run, self._handle, messages)
        return json.loads(result_json)  # type: ignore[no-any-return]

    def destroy(self) -> None:
        """Release native resources."""
        if not self._destroyed:
            from gauss._native import destroy_team

            destroy_team(self._handle)
        super().destroy()

    @classmethod
    def quick(
        cls,
        name: str,
        strategy: str,
        agents: list[dict[str, str]],
    ) -> Team:
        """Quick team builder from role descriptions.

        Example::

            result = Team.quick("content-team", "sequential", [
                {"name": "researcher", "instructions": "Research topics"},
                {"name": "writer", "instructions": "Write summaries"}
            ]).run("Explain quantum computing")
        """
        from gauss.agent import Agent

        team = cls(name)
        for spec in agents:
            agent = Agent(
                name=spec.get("name", "agent"),
                provider=spec.get("provider"),
                model=spec.get("model"),
                system_prompt=spec.get("instructions"),
            )
            team.add(agent)
        team.strategy(strategy)
        return team

    def _check_alive(self) -> None:
        if self._destroyed:
            from gauss.errors import DisposedError

            raise DisposedError("Team", getattr(self, "_name", "team"))
