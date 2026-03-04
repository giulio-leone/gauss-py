"""Configuration parsing utilities."""

from __future__ import annotations

__all__ = ["parse_agent_config", "resolve_env"]

def parse_agent_config(json_str: str) -> str:
    """Parse and validate agent configuration from JSON string.

    Example::

        config = parse_agent_config('{"name": "test", "model": "gpt-4o"}')
    """
    from gauss._native import agent_config_from_json

    return agent_config_from_json(json_str)  # type: ignore[no-any-return]


def resolve_env(template: str) -> str:
    """Resolve environment variable references in a string.

    Example::

        value = resolve_env("${OPENAI_API_KEY}")
    """
    from gauss._native import agent_config_resolve_env

    return agent_config_resolve_env(template)  # type: ignore[no-any-return]
