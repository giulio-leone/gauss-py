"""Tests for MCP (Model Context Protocol) SDK."""

import pytest
from gauss.mcp import (
    McpContent,
    McpModelHint,
    McpModelPreferences,
    McpPrompt,
    McpPromptArgument,
    McpPromptMessage,
    McpPromptResult,
    McpResource,
    McpResourceContent,
    McpSamplingMessage,
    McpSamplingRequest,
    McpSamplingResponse,
    McpServer,
)


class TestMcpServer:
    def test_create_destroy(self) -> None:
        server = McpServer("test", "1.0.0")
        assert server._handle >= 0
        server.destroy()

    def test_context_manager(self) -> None:
        with McpServer("ctx", "1.0.0") as server:
            assert server._handle >= 0
        assert server._destroyed

    def test_add_tool_chaining(self) -> None:
        server = McpServer("chain", "1.0.0")
        result = server.add_tool(
            {"name": "a", "description": "tool a", "inputSchema": {"type": "object"}}
        ).add_tool(
            {"name": "b", "description": "tool b", "inputSchema": {"type": "object"}}
        )
        assert result is server
        server.destroy()

    def test_add_resource_chaining(self) -> None:
        server = McpServer("res", "1.0.0")
        res = McpResource(uri="file:///readme.md", name="README", description="Readme")
        result = server.add_resource(res)
        assert result is server
        server.destroy()

    def test_add_resource_dict(self) -> None:
        server = McpServer("res-d", "1.0.0")
        result = server.add_resource({"uri": "file:///x", "name": "X"})
        assert result is server
        server.destroy()

    def test_add_prompt_chaining(self) -> None:
        server = McpServer("prompt", "1.0.0")
        prompt = McpPrompt(
            name="summarize",
            description="Summarize text",
            arguments=[McpPromptArgument(name="text", required=True)],
        )
        result = server.add_prompt(prompt)
        assert result is server
        server.destroy()

    def test_add_prompt_dict(self) -> None:
        server = McpServer("prompt-d", "1.0.0")
        result = server.add_prompt({"name": "greet", "arguments": []})
        assert result is server
        server.destroy()

    @pytest.mark.asyncio
    async def test_tools_list(self) -> None:
        with McpServer("tl", "1.0.0") as server:
            server.add_tool(
                {"name": "calc", "description": "Calc", "inputSchema": {"type": "object"}}
            )
            resp = await server.handle_message(
                {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
            )
            assert len(resp["result"]["tools"]) == 1
            assert resp["result"]["tools"][0]["name"] == "calc"

    @pytest.mark.asyncio
    async def test_resources_list(self) -> None:
        with McpServer("rl", "1.0.0") as server:
            server.add_resource(McpResource(uri="file:///a", name="A"))
            server.add_resource(McpResource(uri="file:///b", name="B"))
            resp = await server.handle_message(
                {"jsonrpc": "2.0", "id": 2, "method": "resources/list"}
            )
            assert len(resp["result"]["resources"]) == 2

    @pytest.mark.asyncio
    async def test_prompts_list(self) -> None:
        with McpServer("pl", "1.0.0") as server:
            server.add_prompt(McpPrompt(name="greet", arguments=[]))
            resp = await server.handle_message(
                {"jsonrpc": "2.0", "id": 3, "method": "prompts/list"}
            )
            assert len(resp["result"]["prompts"]) == 1
            assert resp["result"]["prompts"][0]["name"] == "greet"

    @pytest.mark.asyncio
    async def test_ping(self) -> None:
        with McpServer("ping", "1.0.0") as server:
            resp = await server.handle_message(
                {"jsonrpc": "2.0", "id": 4, "method": "ping"}
            )
            assert resp["result"] == {}

    @pytest.mark.asyncio
    async def test_capabilities(self) -> None:
        with McpServer("caps", "1.0.0") as server:
            server.add_tool(
                {"name": "t", "description": "t", "inputSchema": {"type": "object"}}
            )
            server.add_resource(McpResource(uri="file:///x", name="x"))
            server.add_prompt(McpPrompt(name="p", arguments=[]))
            resp = await server.handle_message(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1.0"},
                    },
                }
            )
            caps = resp["result"]["capabilities"]
            assert "tools" in caps
            assert "resources" in caps
            assert "prompts" in caps

    def test_throws_after_destroy(self) -> None:
        server = McpServer("dead", "1.0.0")
        server.destroy()
        with pytest.raises(RuntimeError, match="destroyed"):
            server.add_resource(McpResource(uri="x", name="x"))
        with pytest.raises(RuntimeError, match="destroyed"):
            server.add_prompt(McpPrompt(name="x", arguments=[]))

    def test_full_builder(self) -> None:
        with McpServer("full", "2.0.0") as server:
            server.add_tool(
                {"name": "calc", "description": "Calc", "inputSchema": {"type": "object"}}
            ).add_resource(
                McpResource(uri="file:///data.json", name="Data")
            ).add_prompt(
                McpPrompt(
                    name="analyze",
                    description="Analyze",
                    arguments=[McpPromptArgument(name="topic", required=True)],
                )
            )


class TestMcpTypes:
    def test_mcp_resource_to_dict(self) -> None:
        r = McpResource(uri="file:///x", name="X", description="desc", mime_type="text/plain")
        d = r.to_dict()
        assert d["uri"] == "file:///x"
        assert d["mimeType"] == "text/plain"

    def test_mcp_prompt_to_dict(self) -> None:
        p = McpPrompt(
            name="greet",
            description="Greet",
            arguments=[McpPromptArgument(name="name", required=True)],
        )
        d = p.to_dict()
        assert d["name"] == "greet"
        assert d["arguments"][0]["required"] is True

    def test_mcp_content_text(self) -> None:
        c = McpContent.text_content("hello")
        assert c.type == "text"
        assert c.text == "hello"

    def test_mcp_content_image(self) -> None:
        c = McpContent.image_content("base64data", "image/png")
        assert c.type == "image"
        assert c.data == "base64data"

    def test_mcp_content_resource(self) -> None:
        rc = McpResourceContent(uri="file:///x", text="content")
        c = McpContent.resource_content(rc)
        assert c.type == "resource"
        assert c.resource is not None

    def test_mcp_content_from_dict_text(self) -> None:
        c = McpContent.from_dict({"type": "text", "text": "hi"})
        assert c.text == "hi"

    def test_mcp_content_from_dict_image(self) -> None:
        c = McpContent.from_dict({"type": "image", "data": "x", "mimeType": "image/png"})
        assert c.data == "x"

    def test_mcp_content_from_dict_resource(self) -> None:
        c = McpContent.from_dict(
            {"type": "resource", "resource": {"uri": "file:///x", "text": "t"}}
        )
        assert c.resource is not None
        assert c.resource.uri == "file:///x"

    def test_mcp_prompt_message_from_dict(self) -> None:
        m = McpPromptMessage.from_dict(
            {"role": "user", "content": {"type": "text", "text": "hello"}}
        )
        assert m.role == "user"
        assert m.content.text == "hello"

    def test_mcp_prompt_result_from_dict(self) -> None:
        r = McpPromptResult.from_dict(
            {
                "description": "Summary",
                "messages": [
                    {"role": "assistant", "content": {"type": "text", "text": "Done"}}
                ],
            }
        )
        assert r.description == "Summary"
        assert len(r.messages) == 1

    def test_mcp_model_preferences(self) -> None:
        p = McpModelPreferences(
            hints=[McpModelHint(name="claude")],
            cost_priority=0.3,
            speed_priority=0.5,
            intelligence_priority=0.9,
        )
        d = p.to_dict()
        assert d["costPriority"] == 0.3
        assert len(d["hints"]) == 1

    def test_mcp_sampling_request(self) -> None:
        req = McpSamplingRequest(
            messages=[
                McpSamplingMessage(
                    role="user", content=McpContent.text_content("Hello")
                )
            ],
            max_tokens=100,
            temperature=0.7,
        )
        d = req.to_dict()
        assert d["maxTokens"] == 100
        assert d["temperature"] == 0.7

    def test_mcp_sampling_response_from_dict(self) -> None:
        r = McpSamplingResponse.from_dict(
            {
                "role": "assistant",
                "content": {"type": "text", "text": "Hi"},
                "model": "gpt-4",
                "stopReason": "end_turn",
            }
        )
        assert r.model == "gpt-4"
        assert r.stop_reason == "end_turn"

    def test_frozen_dataclasses(self) -> None:
        r = McpResource(uri="x", name="X")
        with pytest.raises(AttributeError):
            r.uri = "y"  # type: ignore[misc]
