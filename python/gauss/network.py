"""Multi-agent network with delegation and supervisor patterns."""

from __future__ import annotations

import functools
import json
from typing import Any

from gauss._types import Message
from gauss.base import StatefulResource
from gauss.errors import DisposedError, ValidationError

__all__ = ["Network"]

class Network(StatefulResource):
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
        super().__init__()
        from gauss._native import create_network

        self._handle: int = create_network()

    @functools.cached_property
    def _resource_name(self) -> str:
        return "Network"

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
        from gauss._native import network_add_agent

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
        from gauss._native import network_set_supervisor

        self._check_alive()
        network_set_supervisor(self._handle, name)
        return self

    @classmethod
    def quick(cls, supervisor: str, agents: list[dict[str, Any]]) -> Network:
        """Quick network builder from role descriptions."""
        from gauss.agent import Agent

        network = cls()
        for spec in agents:
            agent = Agent(
                name=spec.get("name", "agent"),
                provider=spec.get("provider"),
                model=spec.get("model"),
                system_prompt=spec.get("instructions"),
            )
            network.add_agent(
                agent,
                card_json=spec.get("card_json"),
                connections=spec.get("connections"),
            )
        network.set_supervisor(supervisor)
        return network

    @classmethod
    def template(cls, name: str) -> dict[str, Any]:
        if name == "research-delivery":
            return {
                "supervisor": "lead",
                "agents": [
                    {"name": "lead", "instructions": "Coordinate and delegate work."},
                    {"name": "researcher", "instructions": "Research context and constraints."},
                    {"name": "implementer", "instructions": "Implement practical solutions."},
                ],
            }
        if name == "incident-response":
            return {
                "supervisor": "incident-commander",
                "agents": [
                    {
                        "name": "incident-commander",
                        "instructions": "Drive response and coordination.",
                    },
                    {"name": "triage", "instructions": "Assess impact and prioritize mitigation."},
                    {"name": "remediator", "instructions": "Propose and execute remediation steps."},
                ],
            }
        if name == "support-triage":
            return {
                "supervisor": "support-lead",
                "agents": [
                    {"name": "support-lead", "instructions": "Coordinate support workload and escalations."},
                    {"name": "triage-bot", "instructions": "Classify incoming support requests."},
                    {"name": "resolver", "instructions": "Draft and validate customer resolutions."},
                ],
            }
        if name == "fintech-risk-review":
            return {
                "supervisor": "risk-lead",
                "agents": [
                    {
                        "name": "risk-lead",
                        "instructions": "Approve risk recommendations and report rationale.",
                    },
                    {
                        "name": "policy-analyst",
                        "instructions": "Map events to compliance policy obligations.",
                    },
                    {
                        "name": "fraud-scorer",
                        "instructions": "Score suspicious patterns and flag anomalies.",
                    },
                ],
            }
        if name == "rag-ops":
            return {
                "supervisor": "rag-ops-lead",
                "agents": [
                    {
                        "name": "rag-ops-lead",
                        "instructions": "Coordinate retrieval quality and index freshness.",
                    },
                    {
                        "name": "retrieval-auditor",
                        "instructions": "Audit recall/precision and retrieval grounding.",
                    },
                    {
                        "name": "index-maintainer",
                        "instructions": "Propose chunking and indexing improvements.",
                    },
                ],
            }
        raise ValidationError(f'Unknown network template "{name}"', "name")

    @classmethod
    def from_template(cls, name: str) -> Network:
        template = cls.template(name)
        return cls.quick(template["supervisor"], template["agents"])

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
        from gauss._native import network_delegate
        from gauss.agent import _run_native

        self._check_alive()
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [m.to_dict() if isinstance(m, Message) else m for m in prompt]
        result_json = _run_native(network_delegate, self._handle, agent_name, json.dumps(messages))
        return json.loads(result_json)  # type: ignore[no-any-return]

    def destroy(self) -> None:
        """Release native resources."""
        if not self._destroyed:
            from gauss._native import destroy_network

            destroy_network(self._handle)
        super().destroy()

    def agent_cards(self) -> list[dict[str, Any]]:
        """Return network agent cards exposed by the native runtime."""
        from gauss._native import network_agent_cards

        self._check_alive()
        cards_json = network_agent_cards(self._handle)
        return json.loads(cards_json)  # type: ignore[no-any-return]

    def _check_alive(self) -> None:
        if self._destroyed:
            raise DisposedError("Network", "network")
