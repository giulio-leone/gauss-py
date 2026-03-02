"""Middleware chain for request/response processing."""

from __future__ import annotations

from typing import Any


class MiddlewareChain:
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
        from gauss._native import create_middleware_chain

        self._handle: int = create_middleware_chain()
        self._destroyed = False

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
            self._destroyed = True

    def __enter__(self) -> MiddlewareChain:
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    def __del__(self) -> None:
        self.destroy()

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("MiddlewareChain has been destroyed")
