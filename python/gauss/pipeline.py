"""
Pipeline — compose agent operations into clean data flows.

Example::

    from gauss import pipe, map_async, filter_async

    results = await map_async(
        ["apple", "banana", "cherry"],
        lambda fruit: agent.run(f"Describe {fruit}"),
        concurrency=2,
    )
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

__all__ = ["map_sync", "filter_sync", "reduce_sync", "compose"]

T = TypeVar("T")
R = TypeVar("R")


async def pipe(input: Any, *steps: Callable[..., Any]) -> Any:
    """Compose async operations into a pipeline.

    Example::

        result = await pipe(
            "Hello",
            lambda s: agent.run(s),
            lambda r: r.text.upper(),
        )
    """
    result = input
    for step in steps:
        out = step(result)
        if asyncio.iscoroutine(out) or asyncio.isfuture(out):
            result = await out
        else:
            result = out
    return result


def map_sync(
    items: Sequence[T],
    fn: Callable[[T], R],
    *,
    concurrency: int | None = None,
) -> list[R]:
    """Map a function over items with optional thread concurrency.

    Example::

        results = map_sync(
            ["a", "b", "c"],
            lambda item: agent.run(item),
            concurrency=2,
        )
    """
    if concurrency is None or concurrency >= len(items):
        return [fn(item) for item in items]

    results: list[R | None] = [None] * len(items)

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return results  # type: ignore[return-value]


async def map_async(
    items: Sequence[T],
    fn: Callable[[T], Awaitable[R]],
    *,
    concurrency: int | None = None,
) -> list[R]:
    """Map an async function over items with concurrency control.

    Example::

        results = await map_async(
            ["a", "b", "c"],
            lambda item: agent.arun(item),
            concurrency=2,
        )
    """
    if concurrency is None or concurrency >= len(items):
        return list(await asyncio.gather(*(fn(item) for item in items)))

    semaphore = asyncio.Semaphore(concurrency)

    async def _limited(item: T) -> R:
        async with semaphore:
            return await fn(item)

    return list(await asyncio.gather(*(_limited(item) for item in items)))


async def filter_async(
    items: Sequence[T],
    predicate: Callable[[T], Awaitable[bool]],
    *,
    concurrency: int | None = None,
) -> list[T]:
    """Filter items using an async predicate.

    Example::

        valid = await filter_async(
            items,
            lambda item: agent.acheck(item),
            concurrency=3,
        )
    """
    flags = await map_async(items, predicate, concurrency=concurrency)
    return [item for item, flag in zip(items, flags) if flag]


def filter_sync(
    items: Sequence[T],
    predicate: Callable[[T], bool],
    *,
    concurrency: int | None = None,
) -> list[T]:
    """Filter items using a sync predicate with optional thread concurrency."""
    flags = map_sync(items, predicate, concurrency=concurrency)
    return [item for item, flag in zip(items, flags) if flag]


async def reduce_async(
    items: Sequence[T],
    reducer: Callable[[R, T], Awaitable[R]],
    initial: R,
) -> R:
    """Reduce items using an async reducer (sequential).

    Example::

        summary = await reduce_async(
            documents,
            lambda acc, doc: agent.arun(f"Combine: {acc}\\n{doc}"),
            "",
        )
    """
    result = initial
    for item in items:
        result = await reducer(result, item)
    return result


async def tap_async(
    items: Sequence[T],
    fn: Callable[[T, int], Awaitable[None] | None],
) -> list[T]:
    """Execute a side-effect for each item (sequential), returning items unchanged."""
    for idx, item in enumerate(items):
        maybe_awaitable = fn(item, idx)
        if asyncio.iscoroutine(maybe_awaitable) or asyncio.isfuture(maybe_awaitable):
            await maybe_awaitable
    return list(items)


def reduce_sync(
    items: Sequence[T],
    reducer: Callable[[R, T], R],
    initial: R,
) -> R:
    """Reduce items using a sync reducer (sequential)."""
    result = initial
    for item in items:
        result = reducer(result, item)
    return result


def compose(*fns: Callable[[T], T]) -> Callable[[T], T]:
    """Compose multiple functions into one (sync).

    Example::

        enhance = compose(
            lambda text: f"[System] {text}",
            lambda text: text.strip(),
        )
        result = enhance("  hello  ")  # "[System]   hello  ".strip() is wrong
    """

    def _composed(input: T) -> T:
        result = input
        for fn in fns:
            result = fn(result)
        return result

    return _composed


async def compose_async(*fns: Callable[[T], Awaitable[T]]) -> Callable[[T], Awaitable[T]]:
    """Compose multiple async functions into one.

    Example::

        enhance = await compose_async(
            lambda text: agent.arun(f"Improve: {text}"),
            lambda text: agent.arun(f"Summarize: {text}"),
        )
    """

    async def _composed(input: T) -> T:
        result = input
        for fn in fns:
            result = await fn(result)
        return result

    return _composed
