"""Structured error hierarchy for Gauss SDK.

All Gauss errors inherit from :class:`GaussError` to enable typed exception handling::

    try:
        result = agent.run("hello")
    except DisposedError:
        print("Agent was destroyed")
    except ProviderError as e:
        print(f"Provider failed: {e.provider}")

.. versionadded:: 2.1.0
"""

from __future__ import annotations


class GaussError(Exception):
    """Base error for all Gauss SDK errors.

    Attributes:
        code: Machine-readable error code for programmatic matching.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class DisposedError(GaussError):
    """Raised when an operation is attempted on a destroyed resource.

    Attributes:
        resource_type: Type of resource (Agent, Team, Graph).
        resource_name: Name of the destroyed resource.
    """

    def __init__(self, resource_type: str, resource_name: str) -> None:
        super().__init__(
            "RESOURCE_DISPOSED",
            f'{resource_type} "{resource_name}" has been destroyed. Create a new instance.',
        )
        self.resource_type = resource_type
        self.resource_name = resource_name


class ProviderError(GaussError):
    """Raised when provider initialization or communication fails.

    Attributes:
        provider: Name of the provider that failed.
    """

    def __init__(self, provider: str, message: str) -> None:
        super().__init__("PROVIDER_ERROR", f"[{provider}] {message}")
        self.provider = provider


class ToolExecutionError(GaussError):
    """Raised when tool execution fails.

    Attributes:
        tool_name: Name of the tool that failed.
    """

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__("TOOL_EXECUTION_ERROR", f'Tool "{tool_name}" failed: {message}')
        self.tool_name = tool_name


class ValidationError(GaussError):
    """Raised when configuration validation fails.

    Attributes:
        field: Name of the invalid field (optional).
    """

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(
            "VALIDATION_ERROR",
            f'Invalid "{field}": {message}' if field else message,
        )
        self.field = field
