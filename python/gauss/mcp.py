"""MCP (Model Context Protocol) server implementation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from gauss._types import ToolDef

# ── MCP Types ────────────────────────────────────────────────────


@dataclass(frozen=True)
class McpResource:
    """An MCP resource definition."""

    uri: str
    name: str
    description: str | None = None
    mime_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"uri": self.uri, "name": self.name}
        if self.description is not None:
            d["description"] = self.description
        if self.mime_type is not None:
            d["mimeType"] = self.mime_type
        return d


@dataclass(frozen=True)
class McpPromptArgument:
    """An argument for an MCP prompt."""

    name: str
    description: str | None = None
    required: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "required": self.required}
        if self.description is not None:
            d["description"] = self.description
        return d


@dataclass(frozen=True)
class McpPrompt:
    """An MCP prompt definition."""

    name: str
    description: str | None = None
    arguments: list[McpPromptArgument] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "arguments": [a.to_dict() for a in self.arguments],
        }
        if self.description is not None:
            d["description"] = self.description
        return d


@dataclass(frozen=True)
class McpResourceContent:
    """Content from an MCP resource."""

    uri: str
    mime_type: str | None = None
    text: str | None = None
    blob: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> McpResourceContent:
        return cls(
            uri=d["uri"],
            mime_type=d.get("mimeType"),
            text=d.get("text"),
            blob=d.get("blob"),
        )


@dataclass(frozen=True)
class McpContent:
    """Tagged MCP content — text, image, or resource."""

    type: str
    text: str | None = None
    data: str | None = None
    mime_type: str | None = None
    resource: McpResourceContent | None = None

    @classmethod
    def text_content(cls, text: str) -> McpContent:
        return cls(type="text", text=text)

    @classmethod
    def image_content(cls, data: str, mime_type: str) -> McpContent:
        return cls(type="image", data=data, mime_type=mime_type)

    @classmethod
    def resource_content(cls, resource: McpResourceContent) -> McpContent:
        return cls(type="resource", resource=resource)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> McpContent:
        t = d["type"]
        if t == "text":
            return cls.text_content(d["text"])
        if t == "image":
            return cls.image_content(d["data"], d["mimeType"])
        if t == "resource":
            return cls.resource_content(McpResourceContent.from_dict(d["resource"]))
        return cls(type=t)


@dataclass(frozen=True)
class McpPromptMessage:
    """A message in an MCP prompt result."""

    role: str
    content: McpContent

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> McpPromptMessage:
        return cls(role=d["role"], content=McpContent.from_dict(d["content"]))


@dataclass(frozen=True)
class McpPromptResult:
    """Result of getting a prompt."""

    messages: list[McpPromptMessage]
    description: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> McpPromptResult:
        return cls(
            description=d.get("description"),
            messages=[McpPromptMessage.from_dict(m) for m in d.get("messages", [])],
        )


@dataclass(frozen=True)
class McpModelHint:
    """Hint for model selection in sampling."""

    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.name is not None:
            d["name"] = self.name
        return d


@dataclass(frozen=True)
class McpModelPreferences:
    """Model preferences for sampling."""

    hints: list[McpModelHint] | None = None
    cost_priority: float | None = None
    speed_priority: float | None = None
    intelligence_priority: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.hints is not None:
            d["hints"] = [h.to_dict() for h in self.hints]
        if self.cost_priority is not None:
            d["costPriority"] = self.cost_priority
        if self.speed_priority is not None:
            d["speedPriority"] = self.speed_priority
        if self.intelligence_priority is not None:
            d["intelligencePriority"] = self.intelligence_priority
        return d


@dataclass(frozen=True)
class McpSamplingMessage:
    """A message in a sampling request."""

    role: str
    content: McpContent

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content.__dict__}


@dataclass(frozen=True)
class McpSamplingRequest:
    """An MCP sampling/createMessage request."""

    messages: list[McpSamplingMessage]
    max_tokens: int
    model_preferences: McpModelPreferences | None = None
    system_prompt: str | None = None
    include_context: str | None = None
    temperature: float | None = None
    stop_sequences: list[str] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "messages": [m.to_dict() for m in self.messages],
            "maxTokens": self.max_tokens,
        }
        if self.model_preferences is not None:
            d["modelPreferences"] = self.model_preferences.to_dict()
        if self.system_prompt is not None:
            d["systemPrompt"] = self.system_prompt
        if self.include_context is not None:
            d["includeContext"] = self.include_context
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.stop_sequences is not None:
            d["stopSequences"] = self.stop_sequences
        if self.metadata is not None:
            d["metadata"] = self.metadata
        return d


@dataclass(frozen=True)
class McpSamplingResponse:
    """Response from an MCP sampling request."""

    role: str
    content: McpContent
    model: str
    stop_reason: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> McpSamplingResponse:
        return cls(
            role=d["role"],
            content=McpContent.from_dict(d["content"]),
            model=d["model"],
            stop_reason=d.get("stopReason"),
        )


# ── McpServer Class ──────────────────────────────────────────────


class McpServer:
    """MCP-compliant server with tools, resources, and prompts.

    Example::

        server = McpServer("my-tools", "1.0.0")
        server.add_tool(ToolDef(name="greet", description="Say hello"))
        server.add_resource(McpResource(uri="file:///x", name="X"))
        server.add_prompt(McpPrompt(name="summarize", arguments=[]))
        response = server.handle_message({"jsonrpc": "2.0", "method": "tools/list"})
    """

    def __init__(self, name: str, version: str = "1.0.0") -> None:
        from gauss._native import create_mcp_server

        self._handle: int = create_mcp_server(name, version)
        self._destroyed = False

    def add_tool(self, tool: ToolDef | dict[str, Any]) -> McpServer:
        """Register a tool. Returns self for chaining."""
        from gauss._native import mcp_server_add_tool

        self._check_alive()
        tool_dict = tool.to_dict() if isinstance(tool, ToolDef) else tool
        mcp_server_add_tool(self._handle, json.dumps(tool_dict))
        return self

    def add_resource(self, resource: McpResource | dict[str, Any]) -> McpServer:
        """Register a resource. Returns self for chaining."""
        from gauss._native import mcp_server_add_resource

        self._check_alive()
        res_dict = resource.to_dict() if isinstance(resource, McpResource) else resource
        mcp_server_add_resource(self._handle, json.dumps(res_dict))
        return self

    def add_prompt(self, prompt: McpPrompt | dict[str, Any]) -> McpServer:
        """Register a prompt. Returns self for chaining."""
        from gauss._native import mcp_server_add_prompt

        self._check_alive()
        prompt_dict = prompt.to_dict() if isinstance(prompt, McpPrompt) else prompt
        mcp_server_add_prompt(self._handle, json.dumps(prompt_dict))
        return self

    async def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Handle an incoming MCP JSON-RPC message."""
        import inspect

        from gauss._native import mcp_server_handle

        self._check_alive()
        result = mcp_server_handle(self._handle, json.dumps(message))
        if inspect.isawaitable(result):
            result = await result
        return json.loads(result)  # type: ignore[no-any-return]

    def destroy(self) -> None:
        if not self._destroyed:
            from gauss._native import destroy_mcp_server

            destroy_mcp_server(self._handle)
            self._destroyed = True

    def __enter__(self) -> McpServer:
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    def __del__(self) -> None:
        self.destroy()

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("McpServer has been destroyed")
