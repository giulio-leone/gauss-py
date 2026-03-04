"""Middleware chain for request/response processing."""

from __future__ import annotations

import functools

from gauss.base import StatefulResource

__all__ = ["MiddlewareChain"]

class MiddlewareChain(StatefulResource):
    """Chainable middleware pipeline for LLM requests.

    Example::

        from gauss import MiddlewareChain

        chain = (
            MiddlewareChain()
            .use_logging()
            .use_caching(ttl_ms=60_000)
        )
    """

    def __init__(self) -> None:
        super().__init__()
        from gauss._native import create_middleware_chain

        self._handle: int = create_middleware_chain()

    @functools.cached_property
    def _resource_name(self) -> str:
        return "MiddlewareChain"

    def use_logging(self) -> MiddlewareChain:
        """Add logging middleware. Returns self for chaining."""
        from gauss._native import middleware_use_logging

        self._check_alive()
        middleware_use_logging(self._handle)
        return self

    def use_caching(self, ttl_ms: int = 60_000) -> MiddlewareChain:
        """Add caching middleware with a TTL. Returns self for chaining.

        Args:
            ttl_ms: Cache time-to-live in milliseconds. Defaults to 60 seconds.
        """
        from gauss._native import middleware_use_caching

        self._check_alive()
        middleware_use_caching(self._handle, ttl_ms)
        return self

    def use_rate_limit(
        self,
        requests_per_minute: int,
        burst: int | None = None,
    ) -> MiddlewareChain:
        """Add token-bucket rate limiting middleware. Returns self for chaining."""
        from gauss._native import middleware_use_rate_limit

        self._check_alive()
        middleware_use_rate_limit(self._handle, requests_per_minute, burst)
        return self

    @property
    def handle(self) -> int:
        """Return the native handle."""
        self._check_alive()
        return self._handle

    def destroy(self) -> None:
        """Release native resources."""
        if not self._destroyed:
            from gauss._native import destroy_middleware_chain

            destroy_middleware_chain(self._handle)
        super().destroy()
