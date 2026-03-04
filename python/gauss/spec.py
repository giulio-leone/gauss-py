"""
AGENTS.MD and SKILL.MD parsers — parse agent/skill specifications from markdown.

Example::

    from gauss import AgentSpec, SkillSpec, discover_agents

    # Parse a single AGENTS.MD
    spec = AgentSpec.from_markdown(content)
    print(spec.name, spec.tools)

    # Discover all agents in a directory tree
    agents = discover_agents("./agents")

    # Parse a SKILL.MD
    skill = SkillSpec.from_markdown(skill_content)
    print(skill.steps)
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from typing import Any

from gauss._native import discover_agents as _native_discover
from gauss._native import parse_agents_md as _native_parse_agents
from gauss._native import parse_skill_md as _native_parse_skill

__all__ = [
    "AgentToolSpec",
    "SkillStep",
    "SkillParam",
    "AgentSpec",
    "SkillSpec",
    "discover_agents",
]

@dataclass(frozen=True)
class AgentToolSpec:
    """Tool reference within an AGENTS.MD spec."""

    name: str
    description: str = ""
    parameters: dict[str, Any] | None = None


@dataclass(frozen=True)
class SkillStep:
    """Step within a SKILL.MD spec."""

    description: str
    action: str | None = None


@dataclass(frozen=True)
class SkillParam:
    """Input or output parameter in a SKILL.MD spec."""

    name: str
    param_type: str = "string"
    description: str = ""
    required: bool = True


@dataclass(frozen=True)
class AgentSpec:
    """Immutable parsed AGENTS.MD specification."""

    name: str = ""
    description: str = ""
    model: str | None = None
    provider: str | None = None
    instructions: str | None = None
    tools: tuple[AgentToolSpec, ...] = ()
    skills: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    environment: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_markdown(cls, content: str) -> AgentSpec:
        """Parse an AGENTS.MD markdown string into an AgentSpec."""
        raw = _native_parse_agents(content)
        data = json.loads(raw) if isinstance(raw, str) else raw
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> AgentSpec:
        tools = tuple(
            AgentToolSpec(
                name=t.get("name", ""),
                description=t.get("description", ""),
                parameters=t.get("parameters"),
            )
            for t in (data.get("tools") or [])
        )
        env_list = data.get("environment") or []
        environment = dict(env_list) if env_list else {}

        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            model=data.get("model"),
            provider=data.get("provider"),
            instructions=data.get("instructions"),
            tools=tools,
            skills=tuple(data.get("skills") or []),
            capabilities=tuple(data.get("capabilities") or []),
            environment=environment,
            metadata=data.get("metadata") or {},
        )

    def has_tool(self, name: str) -> bool:
        """Check whether a specific tool is declared in this spec."""
        return any(t.name == name for t in self.tools)

    def has_capability(self, name: str) -> bool:
        """Check whether a specific capability is declared."""
        return name in self.capabilities


@dataclass(frozen=True)
class SkillSpec:
    """Immutable parsed SKILL.MD specification."""

    name: str = ""
    description: str = ""
    steps: tuple[SkillStep, ...] = ()
    inputs: tuple[SkillParam, ...] = ()
    outputs: tuple[SkillParam, ...] = ()

    @classmethod
    def from_markdown(cls, content: str) -> SkillSpec:
        """Parse a SKILL.MD markdown string into a SkillSpec."""
        raw = _native_parse_skill(content)
        data = json.loads(raw) if isinstance(raw, str) else raw
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> SkillSpec:
        steps = tuple(
            SkillStep(
                description=s.get("description", ""),
                action=s.get("action"),
            )
            for s in (data.get("steps") or [])
        )
        inputs = tuple(
            SkillParam(
                name=p.get("name", ""),
                param_type=p.get("param_type", "string"),
                description=p.get("description", ""),
                required=p.get("required", True),
            )
            for p in (data.get("inputs") or [])
        )
        outputs = tuple(
            SkillParam(
                name=p.get("name", ""),
                param_type=p.get("param_type", "string"),
                description=p.get("description", ""),
                required=p.get("required", True),
            )
            for p in (data.get("outputs") or [])
        )
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            steps=steps,
            inputs=inputs,
            outputs=outputs,
        )

    @functools.cached_property
    def step_count(self) -> int:
        """Get the total number of steps."""
        return len(self.steps)

    @functools.cached_property
    def required_inputs(self) -> tuple[SkillParam, ...]:
        """Get all required inputs."""
        return tuple(p for p in self.inputs if p.required)


def discover_agents(directory: str) -> list[AgentSpec]:
    """Discover all AGENTS.MD files in a directory tree and parse them."""
    raw = _native_discover(directory)
    data_list = json.loads(raw) if isinstance(raw, str) else raw
    return [AgentSpec._from_dict(d) for d in data_list]
