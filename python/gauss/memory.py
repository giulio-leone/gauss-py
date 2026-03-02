"""Conversation memory backed by Rust."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

MemoryEntryType = Literal["conversation", "fact", "preference", "task", "summary"]


class Memory:
    """Conversation memory with session-based storage.

    Example::

        mem = Memory()
        mem.store("conversation", "Hello!")
        entries = mem.recall(session_id="default")

        # Dict form
        mem.store({"id": "m1", "content": "Hi", "entry_type": "conversation",
                   "timestamp": "2024-01-01T00:00:00Z"})

        # Context manager
        with Memory() as mem:
            mem.store("conversation", "Hi")
    """

    def __init__(self) -> None:
        from gauss._native import create_memory

        self._handle: int = create_memory()
        self._destroyed = False

    def store(
        self,
        entry_type_or_dict: MemoryEntryType | dict[str, Any],
        content: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Store a memory entry.

        Overloaded signatures::

            mem.store("conversation", "Hello!", session_id="s1")
            mem.store({"id": "m1", "content": "Hi", "entry_type": "conversation", ...})
        """
        from gauss._native import memory_store

        self._check_alive()

        if isinstance(entry_type_or_dict, dict):
            entry = entry_type_or_dict
        elif isinstance(entry_type_or_dict, str) and content is not None:
            entry = {
                "id": str(uuid.uuid4()),
                "content": content,
                "entry_type": entry_type_or_dict,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if session_id:
                entry["session_id"] = session_id
        else:
            raise TypeError(
                f"Expected (entry_type, content) or dict, got {type(entry_type_or_dict)}"
            )

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
        from gauss._native import memory_recall

        self._check_alive()
        options: dict[str, Any] = {"sessionId": session_id}
        if limit is not None:
            options["limit"] = limit
        result_json: str = memory_recall(self._handle, json.dumps(options))
        return json.loads(result_json)  # type: ignore[no-any-return]

    def store_sync(
        self,
        content: str,
        entry_type: str = "conversation",
        session_id: str | None = None,
    ) -> None:
        """Store a memory entry with minimal arguments (agent integration helper).

        Args:
            content: The text content to store.
            entry_type: Type of memory entry (default: "conversation").
            session_id: Optional session ID for scoping.

        .. versionadded:: 1.2.0
        """
        self.store(entry_type, content, session_id=session_id)  # type: ignore[arg-type]

    def clear(self, session_id: str = "default") -> None:
        """Clear memory for a session."""
        from gauss._native import memory_clear

        self._check_alive()
        memory_clear(self._handle, session_id)

    def destroy(self) -> None:
        """Release native resources."""
        if not self._destroyed:
            from gauss._native import destroy_memory

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
