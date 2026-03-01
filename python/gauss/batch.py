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
    """Container for the result of a single prompt within a batch run.

    After :func:`batch` completes, each ``BatchItem`` holds either a
    successful :class:`AgentResult` in ``result`` or the caught
    exception in ``error``.

    Attributes:
        input: The original prompt string.
        result: The :class:`AgentResult` on success, or ``None``.
        error: The caught :class:`Exception` on failure, or ``None``.

    Example:
        >>> item = BatchItem("Hello")
        >>> item.result is None
        True
    """

    __slots__ = ("input", "result", "error")

    def __init__(self, input: str) -> None:
        """Create a new ``BatchItem`` for the given prompt.

        Args:
            input: The prompt string to be executed.
        """
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

    Creates a single :class:`Agent` and dispatches all prompts in
    parallel via a thread pool.  Each prompt is executed independently;
    failures are captured in ``BatchItem.error`` without aborting the
    remaining prompts.

    Args:
        prompts: List of prompt strings to execute.
        concurrency: Maximum number of prompts executed in parallel.
            Defaults to ``5``.
        **kwargs: Forwarded to :class:`AgentConfig` (e.g. ``model``,
            ``temperature``, ``provider``).

    Returns:
        A list of :class:`BatchItem` objects, one per input prompt, in
        the same order.  Inspect ``item.result`` for the
        :class:`AgentResult` or ``item.error`` for the exception.

    Example:
        >>> results = batch(["Translate: Hello", "Translate: World"])
        >>> for r in results:
        ...     print(r.result.text if r.result else r.error)
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
