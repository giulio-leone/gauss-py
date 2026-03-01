"""Streaming utilities."""

from __future__ import annotations

import json
from typing import Any


def parse_partial_json(text: str) -> Any:
    """Parse incomplete/streaming JSON into a valid Python object.

    Example::

        obj = parse_partial_json('{"name": "Jo')
        # Returns: {"name": "Jo"}
    """
    from gauss._native import py_parse_partial_json  # type: ignore[import-not-found]

    result_json: str = py_parse_partial_json(text)
    return json.loads(result_json)
