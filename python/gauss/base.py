"""Abstract base class for resources with lifecycle management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from gauss.errors import DisposedError


class StatefulResource(ABC):
    """Abstract base class for resources with lifecycle management.

    Provides a unified pattern for ``_destroyed`` tracking, ``_check_alive``
    validation, ``destroy()`` cleanup, and context-manager support.

    Subclasses must implement the :attr:`_resource_name` property and may
    override :meth:`destroy` to add resource-specific teardown (always call
    ``super().destroy()`` at the end).
    """

    def __init__(self) -> None:
        self._destroyed: bool = False

    @property
    @abstractmethod
    def _resource_name(self) -> str:
        """Name of the resource for error messages."""
        ...

    def _check_alive(self) -> None:
        if self._destroyed:
            raise DisposedError(self._resource_name, self._resource_name.lower())

    def destroy(self) -> None:
        """Release resources. Safe to call multiple times."""
        self._destroyed = True

    def __del__(self) -> None:
        if not self._destroyed:
            self.destroy()

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *args: Any) -> None:
        self.destroy()
