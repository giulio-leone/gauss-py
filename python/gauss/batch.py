"""Batch execution — run multiple prompts through an agent with concurrency control.

Example::

    from gauss import batch

    results = batch(["Translate: Hello", "Translate: World"], concurrency=2)
    for r in results:
        print(r.result.text if r.result else r.error)
"""

from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING, Any

from gauss._types import AgentConfig, AgentResult

if TYPE_CHECKING:
    from collections.abc import Sequence


class BatchItem:
    """Result of a single batch prompt."""

    __slots__ = ("input", "result", "error")

    def __init__(self, input: str) -> None:
        self.input = input
        self.result: AgentResult | None = None
        self.error: Exception | None = None

    def __repr__(self) -> str:
        status = "ok" if self.result else f"error: {self.error}"
        return f"BatchItem(input={self.input!r}, {status})"


def batch(
    prompts: Sequence[str],
    *,
    concurrency: int = 5,
    **kwargs: Any,
) -> list[BatchItem]:
    """Run multiple prompts through an agent with concurrency control.

    Uses a thread pool to execute prompts in parallel.

    Args:
        prompts: List of string prompts.
        concurrency: Max parallel executions (default: 5).
        **kwargs: Passed to AgentConfig.

    Returns:
        List of BatchItem, one per prompt.
    """
    from gauss.agent import Agent

    items = [BatchItem(p) for p in prompts]
    agent = Agent(AgentConfig(**kwargs))

    def _run(idx: int, item: BatchItem) -> None:
        try:
            item.result = agent.run(item.input)
        except Exception as e:
            item.error = e

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_run, i, item) for i, item in enumerate(items)]
            concurrent.futures.wait(futures)
    finally:
        agent.destroy()

    return items
