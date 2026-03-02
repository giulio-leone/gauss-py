"""
A2A (Agent-to-Agent) Protocol SDK for Python.

Provides a high-level, developer-friendly client for interacting with
A2A-compliant agents over HTTP using JSON-RPC 2.0.

Example::

    from gauss import A2aClient

    client = A2aClient("http://localhost:8080")
    card = await client.discover()
    answer = await client.ask("What is the weather?")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ._native import (
    a2a_ask as _a2a_ask,
)
from ._native import (
    a2a_cancel_task as _a2a_cancel_task,
)
from ._native import (
    a2a_discover as _a2a_discover,
)
from ._native import (
    a2a_get_task as _a2a_get_task,
)
from ._native import (
    a2a_send_message as _a2a_send_message,
)

# ── Types ────────────────────────────────────────────────────────────────────


class TaskState(str, Enum):
    """A2A task lifecycle states."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELED = "canceled"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Part:
    """Content part within an A2A message."""

    type: str  # "text", "file", "data"
    text: str | None = None
    mime_type: str | None = None
    data: Any = None
    file: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type}
        if self.text is not None:
            d["text"] = self.text
        if self.mime_type is not None:
            d["mimeType"] = self.mime_type
        if self.data is not None:
            d["data"] = self.data
        if self.file is not None:
            d["file"] = self.file
        if self.metadata is not None:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def text_part(cls, text: str) -> Part:
        return cls(type="text", text=text)

    @classmethod
    def data_part(cls, data: Any, mime_type: str | None = None) -> Part:
        return cls(type="data", data=data, mime_type=mime_type)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Part:
        return cls(
            type=d["type"],
            text=d.get("text"),
            mime_type=d.get("mimeType"),
            data=d.get("data"),
            file=d.get("file"),
            metadata=d.get("metadata"),
        )


@dataclass(frozen=True)
class A2aMessage:
    """A single A2A protocol message."""

    role: str  # "user" or "agent"
    parts: tuple[Part, ...]
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "role": self.role,
            "parts": [p.to_dict() for p in self.parts],
        }
        if self.metadata is not None:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def user_text(cls, text: str) -> A2aMessage:
        return cls(role="user", parts=(Part.text_part(text),))

    @classmethod
    def agent_text(cls, text: str) -> A2aMessage:
        return cls(role="agent", parts=(Part.text_part(text),))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> A2aMessage:
        return cls(
            role=d["role"],
            parts=tuple(Part.from_dict(p) for p in d.get("parts", [])),
            metadata=d.get("metadata"),
        )

    @property
    def text(self) -> str:
        """Extract and join all text from text parts."""
        return "".join(p.text for p in self.parts if p.type == "text" and p.text)


@dataclass(frozen=True)
class TaskStatus:
    """Status of an A2A task."""

    state: TaskState
    message: A2aMessage | None = None
    timestamp: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskStatus:
        state = TaskState(d["state"]) if d.get("state") else TaskState.UNKNOWN
        message = A2aMessage.from_dict(d["message"]) if d.get("message") else None
        return cls(state=state, message=message, timestamp=d.get("timestamp"))


@dataclass(frozen=True)
class Artifact:
    """An artifact produced by an A2A agent."""

    parts: tuple[Part, ...]
    name: str | None = None
    description: str | None = None
    index: int | None = None
    append: bool = False
    last_chunk: bool = False
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Artifact:
        return cls(
            parts=tuple(Part.from_dict(p) for p in d.get("parts", [])),
            name=d.get("name"),
            description=d.get("description"),
            index=d.get("index"),
            append=d.get("append", False),
            last_chunk=d.get("lastChunk", False),
            metadata=d.get("metadata"),
        )


@dataclass(frozen=True)
class AgentSkill:
    """A skill declared in an AgentCard."""

    id: str
    name: str
    description: str | None = None
    tags: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    input_modes: tuple[str, ...] = ()
    output_modes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentSkill:
        return cls(
            id=d["id"],
            name=d["name"],
            description=d.get("description"),
            tags=tuple(d.get("tags", [])),
            examples=tuple(d.get("examples", [])),
            input_modes=tuple(d.get("inputModes", [])),
            output_modes=tuple(d.get("outputModes", [])),
        )


@dataclass(frozen=True)
class AgentCapabilities:
    """Capabilities declared in an AgentCard."""

    streaming: bool = False
    push_notifications: bool = False
    state_transition_history: bool = False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentCapabilities:
        return cls(
            streaming=d.get("streaming", False),
            push_notifications=d.get("pushNotifications", False),
            state_transition_history=d.get("stateTransitionHistory", False),
        )


@dataclass(frozen=True)
class AgentCard:
    """Agent Card — served at /.well-known/agent.json"""

    name: str
    url: str
    description: str | None = None
    version: str | None = None
    documentation_url: str | None = None
    capabilities: AgentCapabilities | None = None
    skills: tuple[AgentSkill, ...] = ()
    default_input_modes: tuple[str, ...] = ()
    default_output_modes: tuple[str, ...] = ()
    provider: dict[str, Any] | None = None
    authentication: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentCard:
        caps = AgentCapabilities.from_dict(d["capabilities"]) if d.get("capabilities") else None
        skills = tuple(AgentSkill.from_dict(s) for s in d.get("skills", []))
        return cls(
            name=d["name"],
            url=d["url"],
            description=d.get("description"),
            version=d.get("version"),
            documentation_url=d.get("documentationUrl"),
            capabilities=caps,
            skills=skills,
            default_input_modes=tuple(d.get("defaultInputModes", [])),
            default_output_modes=tuple(d.get("defaultOutputModes", [])),
            provider=d.get("provider"),
            authentication=d.get("authentication"),
        )


@dataclass(frozen=True)
class Task:
    """An A2A task."""

    id: str
    status: TaskStatus
    session_id: str | None = None
    artifacts: tuple[Artifact, ...] = ()
    history: tuple[A2aMessage, ...] = ()
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Task:
        return cls(
            id=d["id"],
            status=TaskStatus.from_dict(d["status"]),
            session_id=d.get("sessionId"),
            artifacts=tuple(Artifact.from_dict(a) for a in d.get("artifacts", [])),
            history=tuple(A2aMessage.from_dict(m) for m in d.get("history", [])),
            metadata=d.get("metadata"),
        )

    @property
    def text(self) -> str | None:
        """Extract text from the task's latest status message."""
        if self.status.message:
            return self.status.message.text
        return None


# ── Client ───────────────────────────────────────────────────────────────────


class A2aClient:
    """
    Client for communicating with A2A-compliant agents.

    Example::

        client = A2aClient("http://localhost:8080")

        # Discover agent capabilities
        card = await client.discover()
        print(card.name, card.skills)

        # Quick ask (text in → text out)
        answer = await client.ask("Summarize this document.")

        # Full message exchange
        result = await client.send_message(A2aMessage.user_text("Hello!"))

        # Get task status
        task = await client.get_task("task-123")

        # Cancel a task
        await client.cancel_task("task-123")
    """

    def __init__(self, base_url: str, *, auth_token: str | None = None) -> None:
        self._base_url = base_url
        self._auth_token = auth_token

    @property
    def base_url(self) -> str:
        return self._base_url

    async def discover(self) -> AgentCard:
        """Fetch the remote agent's AgentCard from /.well-known/agent.json"""
        raw = await _a2a_discover(self._base_url, self._auth_token)
        return AgentCard.from_dict(json.loads(raw))

    async def send_message(
        self,
        message: A2aMessage,
        *,
        config: dict[str, Any] | None = None,
    ) -> Task | A2aMessage:
        """Send a message to the agent. Returns a Task or an A2aMessage."""
        message_json = json.dumps(message.to_dict())
        config_json = json.dumps(config) if config else None
        raw = await _a2a_send_message(
            self._base_url,
            self._auth_token,
            message_json,
            config_json,
        )
        data = json.loads(raw)
        if "status" in data:
            return Task.from_dict(data)
        return A2aMessage.from_dict(data)

    async def ask(self, text: str) -> str:
        """Quick helper: send text and get text response."""
        return await _a2a_ask(self._base_url, self._auth_token, text)

    async def get_task(
        self,
        task_id: str,
        *,
        history_length: int | None = None,
    ) -> Task:
        """Get a task by ID."""
        raw = await _a2a_get_task(
            self._base_url,
            self._auth_token,
            task_id,
            history_length,
        )
        return Task.from_dict(json.loads(raw))

    async def cancel_task(self, task_id: str) -> Task:
        """Cancel a running task."""
        raw = await _a2a_cancel_task(
            self._base_url,
            self._auth_token,
            task_id,
        )
        return Task.from_dict(json.loads(raw))


def text_message(text: str) -> A2aMessage:
    """Build a user text message."""
    return A2aMessage.user_text(text)


def user_message(text: str) -> A2aMessage:
    """Alias for text_message."""
    return text_message(text)


def agent_message(text: str) -> A2aMessage:
    """Build an agent text message."""
    return A2aMessage.agent_text(text)


def extract_text(message: A2aMessage | dict[str, Any]) -> str:
    """Extract concatenated text content from a message."""
    msg = message if isinstance(message, A2aMessage) else A2aMessage.from_dict(message)
    return msg.text


def task_text(task: Task | dict[str, Any]) -> str:
    """Extract text from task status message or first artifact."""
    parsed = task if isinstance(task, Task) else Task.from_dict(task)
    if parsed.text:
        return parsed.text
    if parsed.artifacts:
        return "".join(
            part.text
            for part in parsed.artifacts[0].parts
            if part.type == "text" and part.text
        )
    return ""
