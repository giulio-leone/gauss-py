"""Human-in-the-Loop: Checkpoint/restore for agent state."""

from __future__ import annotations

import json
from typing import Any

from gauss.base import StatefulResource


class CheckpointStore(StatefulResource):
    """Save and restore agent execution state.

    Example::

        store = CheckpointStore()
        store.save({"id": "cp1", "step": 3, "data": {"progress": 0.5}})
        cp = store.load("cp1")
        latest = store.load_latest("session-1")
    """

    def __init__(self) -> None:
        super().__init__()
        from gauss._native import create_checkpoint_store

        self._handle: int = create_checkpoint_store()

    @property
    def _resource_name(self) -> str:
        return "CheckpointStore"

    def save(self, checkpoint: dict[str, Any]) -> None:
        """Save a checkpoint."""
        from gauss._native import checkpoint_save

        self._check_alive()
        checkpoint_save(self._handle, json.dumps(checkpoint))

    def load(self, checkpoint_id: str) -> dict[str, Any] | None:
        """Load a checkpoint by ID. Returns None if not found."""
        from gauss._native import checkpoint_load

        self._check_alive()
        result_json: str = checkpoint_load(self._handle, checkpoint_id)
        data = json.loads(result_json)
        return data if data else None  # type: ignore[no-any-return]

    def load_latest(self, session_id: str = "default") -> dict[str, Any] | None:
        """Load the most recent checkpoint for a session.

        NOTE: `checkpoint_load_latest` is not yet exposed in PyO3 bindings.
        This method currently falls back to `checkpoint_load` with a session prefix.
        """
        from gauss._native import checkpoint_load

        self._check_alive()
        result_json: str = checkpoint_load(self._handle, f"latest:{session_id}")
        data = json.loads(result_json)
        return data if data else None  # type: ignore[no-any-return]

    def destroy(self) -> None:
        if not self._destroyed:
            from gauss._native import destroy_checkpoint_store

            destroy_checkpoint_store(self._handle)
        super().destroy()
