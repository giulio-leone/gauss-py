"""Tests for A2A protocol SDK."""

import pytest
from gauss.a2a import (
    A2aClient,
    A2aMessage,
    AgentCard,
    Artifact,
    Part,
    Task,
    TaskState,
    TaskStatus,
    agent_message,
    extract_text,
    task_text,
    text_message,
    user_message,
)

# ── Part Tests ────────────────────────────────────────────────────────────────


class TestPart:
    def test_text_part(self) -> None:
        p = Part.text_part("Hello")
        assert p.type == "text"
        assert p.text == "Hello"

    def test_data_part(self) -> None:
        p = Part.data_part({"key": "value"}, mime_type="application/json")
        assert p.type == "data"
        assert p.data == {"key": "value"}
        assert p.mime_type == "application/json"

    def test_to_dict(self) -> None:
        p = Part.text_part("Hi")
        d = p.to_dict()
        assert d == {"type": "text", "text": "Hi"}

    def test_to_dict_full(self) -> None:
        p = Part(type="data", data=42, mime_type="text/plain", metadata={"k": "v"})
        d = p.to_dict()
        assert d["type"] == "data"
        assert d["data"] == 42
        assert d["mimeType"] == "text/plain"
        assert d["metadata"] == {"k": "v"}

    def test_from_dict(self) -> None:
        d = {"type": "text", "text": "world", "metadata": {"x": 1}}
        p = Part.from_dict(d)
        assert p.type == "text"
        assert p.text == "world"
        assert p.metadata == {"x": 1}

    def test_frozen(self) -> None:
        p = Part.text_part("hello")
        with pytest.raises(AttributeError):
            p.text = "changed"  # type: ignore


# ── A2aMessage Tests ──────────────────────────────────────────────────────────


class TestA2aMessage:
    def test_user_text(self) -> None:
        msg = A2aMessage.user_text("Hello")
        assert msg.role == "user"
        assert len(msg.parts) == 1
        assert msg.parts[0].text == "Hello"

    def test_agent_text(self) -> None:
        msg = A2aMessage.agent_text("Response")
        assert msg.role == "agent"
        assert msg.parts[0].text == "Response"

    def test_text_property(self) -> None:
        msg = A2aMessage(
            role="agent",
            parts=(Part.text_part("Hello "), Part.data_part(42), Part.text_part("World")),
        )
        assert msg.text == "Hello World"

    def test_text_empty(self) -> None:
        msg = A2aMessage(role="agent", parts=(Part.data_part(42),))
        assert msg.text == ""

    def test_to_dict(self) -> None:
        msg = A2aMessage.user_text("Test")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert len(d["parts"]) == 1
        assert d["parts"][0]["text"] == "Test"

    def test_from_dict(self) -> None:
        d = {
            "role": "agent",
            "parts": [{"type": "text", "text": "Hi"}],
            "metadata": {"source": "test"},
        }
        msg = A2aMessage.from_dict(d)
        assert msg.role == "agent"
        assert msg.text == "Hi"
        assert msg.metadata == {"source": "test"}

    def test_helper_builders_and_extract(self) -> None:
        msg = text_message("hello")
        assert msg.role == "user"
        assert msg.text == "hello"
        assert user_message("world").role == "user"
        assert agent_message("done").role == "agent"
        assert extract_text({"role": "agent", "parts": [{"type": "text", "text": "ok"}]}) == "ok"


# ── TaskState Tests ───────────────────────────────────────────────────────────


class TestTaskState:
    def test_all_states(self) -> None:
        states = ["submitted", "working", "input-required", "completed", "canceled", "failed", "unknown"]
        for s in states:
            ts = TaskState(s)
            assert ts.value == s

    def test_enum_members(self) -> None:
        assert TaskState.SUBMITTED == "submitted"
        assert TaskState.WORKING == "working"
        assert TaskState.INPUT_REQUIRED == "input-required"
        assert TaskState.COMPLETED == "completed"
        assert TaskState.CANCELED == "canceled"
        assert TaskState.FAILED == "failed"
        assert TaskState.UNKNOWN == "unknown"


# ── TaskStatus Tests ──────────────────────────────────────────────────────────


class TestTaskStatus:
    def test_from_dict_minimal(self) -> None:
        ts = TaskStatus.from_dict({"state": "working"})
        assert ts.state == TaskState.WORKING
        assert ts.message is None

    def test_from_dict_with_message(self) -> None:
        ts = TaskStatus.from_dict({
            "state": "completed",
            "message": {"role": "agent", "parts": [{"type": "text", "text": "Done!"}]},
            "timestamp": "2024-01-01T00:00:00Z",
        })
        assert ts.state == TaskState.COMPLETED
        assert ts.message is not None
        assert ts.message.text == "Done!"
        assert ts.timestamp == "2024-01-01T00:00:00Z"


# ── Artifact Tests ────────────────────────────────────────────────────────────


class TestArtifact:
    def test_from_dict(self) -> None:
        a = Artifact.from_dict({
            "name": "result",
            "parts": [{"type": "text", "text": "output"}],
            "index": 0,
        })
        assert a.name == "result"
        assert len(a.parts) == 1
        assert a.index == 0


# ── AgentCard Tests ───────────────────────────────────────────────────────────


class TestAgentCard:
    def test_from_dict(self) -> None:
        card = AgentCard.from_dict({
            "name": "Test Agent",
            "url": "http://localhost:8080",
            "version": "1.0.0",
            "capabilities": {"streaming": True, "pushNotifications": False},
            "skills": [{"id": "summarize", "name": "Summarize", "tags": ["nlp"]}],
        })
        assert card.name == "Test Agent"
        assert card.url == "http://localhost:8080"
        assert card.version == "1.0.0"
        assert card.capabilities is not None
        assert card.capabilities.streaming is True
        assert len(card.skills) == 1
        assert card.skills[0].id == "summarize"


# ── Task Tests ────────────────────────────────────────────────────────────────


class TestTask:
    def test_from_dict(self) -> None:
        task = Task.from_dict({
            "id": "task-001",
            "sessionId": "session-abc",
            "status": {"state": "working"},
            "artifacts": [{"parts": [{"type": "text", "text": "out"}]}],
            "history": [
                {"role": "user", "parts": [{"type": "text", "text": "in"}]},
            ],
        })
        assert task.id == "task-001"
        assert task.session_id == "session-abc"
        assert task.status.state == TaskState.WORKING
        assert len(task.artifacts) == 1
        assert len(task.history) == 1

    def test_text_with_message(self) -> None:
        task = Task.from_dict({
            "id": "t1",
            "status": {
                "state": "completed",
                "message": {"role": "agent", "parts": [{"type": "text", "text": "Done!"}]},
            },
        })
        assert task.text == "Done!"

    def test_text_without_message(self) -> None:
        task = Task.from_dict({"id": "t2", "status": {"state": "submitted"}})
        assert task.text is None

    def test_task_text_helper(self) -> None:
        payload = {
            "id": "t3",
            "status": {
                "state": "completed",
                "message": {"role": "agent", "parts": [{"type": "text", "text": "result"}]},
            },
        }
        assert task_text(payload) == "result"


# ── A2aClient Construction Tests ─────────────────────────────────────────────


class TestA2aClient:
    def test_construct_with_url(self) -> None:
        client = A2aClient("http://localhost:8080")
        assert client.base_url == "http://localhost:8080"

    def test_construct_with_auth(self) -> None:
        client = A2aClient("http://localhost:8080", auth_token="secret")
        assert client.base_url == "http://localhost:8080"
