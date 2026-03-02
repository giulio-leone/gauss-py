"""Human-in-the-Loop: Approval workflows."""

from __future__ import annotations

import json
from typing import Any


class ApprovalManager:
    """Manage human approval workflows for sensitive agent actions.

    Example::

        mgr = ApprovalManager()
        req_id = mgr.request("dangerous_tool", {"action": "delete"})
        mgr.approve(req_id)
        # or: mgr.deny(req_id, reason="Not safe")
        pending = mgr.list_pending()
    """

    def __init__(self) -> None:
        from gauss._native import create_approval_manager

        self._handle: int = create_approval_manager()
        self._destroyed = False

    def request(
        self,
        tool_name: str,
        args: dict[str, Any],
        session_id: str = "default",
    ) -> str:
        """Create an approval request. Returns the request ID."""
        from gauss._native import approval_request

        self._check_alive()
        return approval_request(  # type: ignore[no-any-return]
            self._handle, tool_name, json.dumps(args), session_id
        )

    def approve(self, request_id: str, modified_args: dict[str, Any] | None = None) -> None:
        """Approve a pending request, optionally with modified arguments."""
        from gauss._native import approval_approve

        self._check_alive()
        approval_approve(
            self._handle,
            request_id,
            json.dumps(modified_args) if modified_args else None,
        )

    def deny(self, request_id: str, reason: str | None = None) -> None:
        """Deny a pending request."""
        from gauss._native import approval_deny

        self._check_alive()
        approval_deny(self._handle, request_id, reason)

    def list_pending(self) -> list[dict[str, Any]]:
        """List all pending approval requests."""
        from gauss._native import approval_list_pending

        self._check_alive()
        result_json: str = approval_list_pending(self._handle)
        return json.loads(result_json)  # type: ignore[no-any-return]

    def destroy(self) -> None:
        if not self._destroyed:
            from gauss._native import destroy_approval_manager

            destroy_approval_manager(self._handle)
            self._destroyed = True

    def __enter__(self) -> ApprovalManager:
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    def __del__(self) -> None:
        self.destroy()

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("ApprovalManager has been destroyed")
