"""Conversation memory backed by Rust."""

from __future__ import annotations

import json
from typing import Any

from gauss._types import Message


class Memory:
    """Conversation memory with session-based storage.

    Example::

        mem = Memory()
        mem.store("user", "Hello!")
        entries = mem.recall(session_id="default")

        # Context manager
        with Memory() as mem:
            mem.store(Message("user", "Hi"))
    """

    def __init__(self) -> None:
        from gauss._native import create_memory  # type: ignore[import-not-found]

        self._handle: int = create_memory()
        self._destroyed = False

    def store(
        self,
        role_or_message: str | Message | dict[str, Any],
        content: str | None = None,
        session_id: str = "default",
    ) -> None:
        """Store a memory entry.

        Overloaded signatures::

            mem.store("user", "Hello!", session_id="s1")
            mem.store(Message("user", "Hello!"), session_id="s1")
            mem.store({"role": "user", "content": "Hello!", "sessionId": "s1"})
        """
        from gauss._native import memory_store  # type: ignore[import-not-found]

        self._check_alive()

        if isinstance(role_or_message, str) and content is not None:
            entry = {"role": role_or_message, "content": content, "sessionId": session_id}
        elif isinstance(role_or_message, Message):
            entry = {**role_or_message.to_dict(), "sessionId": session_id}
        elif isinstance(role_or_message, dict):
            entry = role_or_message
        else:
            raise TypeError(f"Expected str, Message, or dict, got {type(role_or_message)}")

        memory_store(self._handle, json.dumps(entry))

    def recall(
        self,
        session_id: str = "default",
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Recall memory entries for a session.

        Args:
            session_id: Session to recall from.
            limit: Optional max number of entries to return.
        """
        from gauss._native import memory_recall  # type: ignore[import-not-found]

        self._check_alive()
        options: dict[str, Any] = {"sessionId": session_id}
        if limit is not None:
            options["limit"] = limit
        result_json: str = memory_recall(self._handle, json.dumps(options))
        return json.loads(result_json)  # type: ignore[no-any-return]

    def clear(self, session_id: str = "default") -> None:
        """Clear memory for a session."""
        from gauss._native import memory_clear  # type: ignore[import-not-found]

        self._check_alive()
        memory_clear(self._handle, session_id)

    def destroy(self) -> None:
        """Release native resources."""
        if not self._destroyed:
            from gauss._native import destroy_memory  # type: ignore[import-not-found]

            destroy_memory(self._handle)
            self._destroyed = True

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    def __del__(self) -> None:
        self.destroy()

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("Memory has been destroyed")
