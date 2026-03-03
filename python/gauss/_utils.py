"""Shared internal utilities for the Gauss SDK."""

from __future__ import annotations

from typing import Any


def _run_native(func: Any, *args: Any) -> Any:
    """Call a native function that may return a coroutine or a plain value.

    PyO3 functions using ``pyo3_async_runtimes::tokio::future_into_py`` require
    a running asyncio event loop at call-time (they return a coroutine).  This
    helper transparently handles the three execution contexts:

    1. No event loop running → create one with ``asyncio.run``.
    2. Event loop already running (e.g. Jupyter) → offload to a thread.
    3. Mock / sync return → pass through immediately.
    """
    import asyncio
    import inspect

    async def _call() -> Any:
        res = func(*args)
        if inspect.isawaitable(res):
            return await res
        return res

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, _call()).result()

    # Fast path: try sync first (mock environment)
    try:
        result = func(*args)
        if not inspect.isawaitable(result):
            return result
    except RuntimeError:
        pass

    return asyncio.run(_call())
