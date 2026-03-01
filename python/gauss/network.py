"""Multi-agent network with delegation and supervisor patterns."""

from __future__ import annotations

import json
from typing import Any

from gauss._types import Message


class Network:
    """Multi-agent network with supervisor-based delegation.

    Example::

        from gauss import Agent, Network

        analyst = Agent(name="analyst")
        coder = Agent(name="coder")

        net = (
            Network()
            .add_agent(analyst)
            .add_agent(coder)
            .set_supervisor("analyst")
        )
        result = net.delegate("coder", "Write a sorting algorithm")
    """

    def __init__(self) -> None:
        from gauss._native import create_network  # type: ignore[import-not-found]

        self._handle: int = create_network()
        self._destroyed = False

    def add_agent(
        self,
        agent: Any,
        card_json: str | None = None,
        connections: list[str] | None = None,
    ) -> Network:
        """Add an agent to the network. Returns self for chaining.

        Args:
            agent: An Agent instance.
            card_json: Optional A2A agent card as JSON string.
            connections: Optional list of connected agent names.
        """
        from gauss._native import network_add_agent  # type: ignore[import-not-found]

        self._check_alive()
        network_add_agent(
            self._handle,
            agent._config.name,
            agent.handle,
            card_json,
            connections,
        )
        return self

    def set_supervisor(self, name: str) -> Network:
        """Set the supervisor agent by name. Returns self for chaining."""
        from gauss._native import network_set_supervisor  # type: ignore[import-not-found]

        self._check_alive()
        network_set_supervisor(self._handle, name)
        return self

    def delegate(
        self,
        agent_name: str,
        prompt: str | list[Message | dict[str, str]],
    ) -> dict[str, Any]:
        """Delegate a task to a specific agent.

        Args:
            agent_name: Name of the agent to delegate to.
            prompt: A string or list of messages.

        Returns:
            The delegation result as a dict.
        """
        from gauss._native import network_delegate  # type: ignore[import-not-found]

        self._check_alive()
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [
                m.to_dict() if isinstance(m, Message) else m for m in prompt
            ]
        result_json: str = network_delegate(
            self._handle, agent_name, json.dumps(messages)
        )
        return json.loads(result_json)  # type: ignore[no-any-return]

    def destroy(self) -> None:
        """Release native resources."""
        if not self._destroyed:
            from gauss._native import destroy_network  # type: ignore[import-not-found]

            destroy_network(self._handle)
            self._destroyed = True

    def __enter__(self) -> Network:
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    def __del__(self) -> None:
        self.destroy()

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("Network has been destroyed")
