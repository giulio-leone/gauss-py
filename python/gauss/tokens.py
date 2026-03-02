"""Token counting utilities."""

from __future__ import annotations

import json
from typing import Any


def count_tokens(text: str) -> int:
    """Count tokens in a text string.

    Example::

        n = count_tokens("Hello, world!")
    """
    from gauss._native import count_tokens as _native

    return _native(text)  # type: ignore[no-any-return]


def count_tokens_for_model(text: str, model: str) -> int:
    """Count tokens for a specific model.

    Example::

        n = count_tokens_for_model("Hello", "gpt-4o")
    """
    from gauss._native import count_tokens_for_model as _native

    return _native(text, model)  # type: ignore[no-any-return]


def count_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Count tokens in a message list.

    Example::

        n = count_message_tokens([{"role": "user", "content": "Hi"}])
    """
    from gauss._native import count_message_tokens as _native

    return _native(json.dumps(messages))  # type: ignore[no-any-return]


def get_context_window_size(model: str) -> int:
    """Get the context window size for a model.

    Example::

        size = get_context_window_size("gpt-4o")  # 128000
    """
    from gauss._native import get_context_window_size as _native

    return _native(model)  # type: ignore[no-any-return]
