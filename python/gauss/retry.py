"""
Retry utilities — exponential backoff and configurable retry logic.

Example::

    from gauss import with_retry, retryable

    result = with_retry(lambda: agent.run("Hello"), max_retries=3)

    # Or wrap an agent:
    run = retryable(agent, max_retries=5)
    result = run("Hello")
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3).
        backoff: Backoff strategy: "fixed", "linear", or "exponential" (default).
        base_delay_s: Base delay in seconds (default: 1.0).
        max_delay_s: Maximum delay in seconds (default: 30.0).
        jitter: Jitter factor 0–1 (default: 0.1).
        retry_if: Optional predicate — retry only if this returns True.
        on_retry: Called on each retry with (error, attempt, delay_s).
    """

    max_retries: int = 3
    backoff: str = "exponential"
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    jitter: float = 0.1
    retry_if: Callable[[Exception, int], bool] | None = None
    on_retry: Callable[[Exception, int, float], None] | None = None


def _compute_delay(config: RetryConfig, attempt: int) -> float:
    if config.backoff == "fixed":
        delay = config.base_delay_s
    elif config.backoff == "linear":
        delay = config.base_delay_s * attempt
    else:  # exponential
        delay = config.base_delay_s * (2 ** (attempt - 1))

    jitter_range = delay * config.jitter
    delay += random.uniform(-jitter_range, jitter_range)
    return min(max(0.0, delay), config.max_delay_s)


def with_retry(
    fn: Callable[[], T],
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> T:
    """Execute a function with retry logic.

    Example::

        result = with_retry(lambda: agent.run("Hello"), max_retries=3)

    Args:
        fn: Zero-argument callable to retry.
        config: RetryConfig instance, or pass keyword args.
        **kwargs: Shorthand for RetryConfig fields.

    Returns:
        The return value of fn on success.

    Raises:
        The last exception after all retries are exhausted.
    """
    cfg = config or RetryConfig(**kwargs)
    last_error: Exception | None = None

    for attempt in range(cfg.max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e

            if attempt == cfg.max_retries:
                break

            if cfg.retry_if and not cfg.retry_if(e, attempt + 1):
                break

            delay = _compute_delay(cfg, attempt + 1)
            if cfg.on_retry:
                cfg.on_retry(e, attempt + 1, delay)
            time.sleep(delay)

    raise last_error  # type: ignore[misc]


def retryable(
    agent: Any,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> Callable[[str], Any]:
    """Wrap an agent's run() with retry logic.

    Example::

        run = retryable(agent, max_retries=5, backoff="exponential")
        result = run("Hello")

    Args:
        agent: An Agent instance with a .run() method.
        config: RetryConfig or keyword args.

    Returns:
        A callable that accepts a prompt string.
    """
    cfg = config or RetryConfig(**kwargs)

    def _run(prompt: str) -> Any:
        return with_retry(lambda: agent.run(prompt), cfg)

    return _run
