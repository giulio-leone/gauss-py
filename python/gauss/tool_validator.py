"""Tool input validation."""

from __future__ import annotations

import functools
import json
from typing import Any

from gauss.base import StatefulResource


class ToolValidator(StatefulResource):
    """Validate tool inputs against JSON schemas with auto-coercion.

    Example::

        validator = ToolValidator(strategies=["type_cast", "null_to_default"])
        result = validator.validate(
            {"name": "John", "age": "25"},
            {"type": "object", "properties": {"age": {"type": "integer"}}}
        )
    """

    def __init__(self, strategies: list[str] | None = None) -> None:
        super().__init__()
        from gauss._native import create_tool_validator

        self._handle: int = create_tool_validator(strategies)

    @functools.cached_property
    def _resource_name(self) -> str:
        return "ToolValidator"

    def validate(self, input_data: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
        """Validate input against schema. Returns validation result."""
        from gauss._native import tool_validator_validate

        self._check_alive()
        result_json: str = tool_validator_validate(
            self._handle, json.dumps(input_data), json.dumps(schema)
        )
        return json.loads(result_json)  # type: ignore[no-any-return]

    def destroy(self) -> None:
        if not self._destroyed:
            from gauss._native import destroy_tool_validator

            destroy_tool_validator(self._handle)
        super().destroy()
