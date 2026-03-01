"""Tool input validation."""

from __future__ import annotations

import json
from typing import Any


class ToolValidator:
    """Validate tool inputs against JSON schemas with auto-coercion.

    Example::

        validator = ToolValidator(strategies=["type_cast", "null_to_default"])
        result = validator.validate(
            {"name": "John", "age": "25"},
            {"type": "object", "properties": {"age": {"type": "integer"}}}
        )
    """

    def __init__(self, strategies: list[str] | None = None) -> None:
        from gauss._native import create_tool_validator  # type: ignore[import-not-found]

        self._handle: int = create_tool_validator(strategies)
        self._destroyed = False

    def validate(self, input_data: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        """Validate input against schema. Returns validation result."""
        from gauss._native import tool_validator_validate  # type: ignore[import-not-found]

        self._check_alive()
        result_json: str = tool_validator_validate(
            self._handle, json.dumps(input_data), json.dumps(schema)
        )
        return json.loads(result_json)  # type: ignore[no-any-return]

    def destroy(self) -> None:
        if not self._destroyed:
            from gauss._native import destroy_tool_validator  # type: ignore[import-not-found]

            destroy_tool_validator(self._handle)
            self._destroyed = True

    def __enter__(self) -> ToolValidator:
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    def __del__(self) -> None:
        self.destroy()

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("ToolValidator has been destroyed")
