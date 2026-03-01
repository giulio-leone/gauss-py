"""Code execution and image generation — standalone utility functions.

These functions don't require an Agent instance.

Example::

    from gauss.code_execution import execute_code, available_runtimes

    result = execute_code("python", "print(42)")
    print(result.stdout)  # "42\\n"
"""

from __future__ import annotations

import json
from typing import Any

from gauss._types import (
    CodeExecutionResult,
    GeneratedImageData,
    ImageGenerationResult,
    ProviderType,
    detect_provider,
    resolve_api_key,
    _default_model,
)


def _run_native(func: Any, *args: Any) -> Any:
    """Call a native function that may return a coroutine or a plain value."""
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

    try:
        result = func(*args)
        if not inspect.isawaitable(result):
            return result
    except RuntimeError:
        pass

    return asyncio.run(_call())


def execute_code(
    language: str,
    code: str,
    *,
    timeout: int = 30,
    sandbox: str = "default",
) -> CodeExecutionResult:
    """Execute code in a sandboxed runtime.

    Args:
        language: "python", "javascript", or "bash"
        code: Source code to execute
        timeout: Max execution time in seconds (default: 30)
        sandbox: "default", "strict", or "permissive"

    Returns:
        CodeExecutionResult with stdout, stderr, exit_code, etc.

    Example::

        result = execute_code("python", "print(42)")
        assert result.stdout.strip() == "42"
    """
    from gauss._native import execute_code as _exec  # type: ignore[import-not-found]

    result_json = _run_native(_exec, language, code, timeout, None, sandbox)
    data = json.loads(result_json)
    return CodeExecutionResult(
        stdout=data.get("stdout", ""),
        stderr=data.get("stderr", ""),
        exit_code=data.get("exit_code", -1),
        timed_out=data.get("timed_out", False),
        runtime=data.get("runtime", language),
        success=data.get("success", False),
    )


def available_runtimes() -> list[str]:
    """Check which code execution runtimes are available on this system.

    Returns:
        List of runtime names, e.g. ["python", "bash"]
    """
    from gauss._native import available_runtimes as _runtimes  # type: ignore[import-not-found]

    result_json = _run_native(_runtimes)
    return json.loads(result_json)


def generate_image(
    prompt: str,
    *,
    provider: ProviderType | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    size: str | None = None,
    quality: str | None = None,
    style: str | None = None,
    aspect_ratio: str | None = None,
    n: int | None = None,
    response_format: str | None = None,
) -> ImageGenerationResult:
    """Generate images using a provider's image generation API.

    Example::

        result = generate_image("A sunset over mountains", model="dall-e-3")
        print(result.images[0].url)
    """
    from gauss._native import (  # type: ignore[import-not-found]
        create_provider,
        destroy_provider,
        generate_image as _gen_image,
    )

    resolved_provider = provider or detect_provider()
    resolved_model = model or _default_model(resolved_provider)
    resolved_key = api_key or resolve_api_key(resolved_provider)
    prov_opts: dict[str, Any] = {"api_key": resolved_key}
    if base_url:
        prov_opts["base_url"] = base_url
    handle = _run_native(create_provider, resolved_provider.value, resolved_model, json.dumps(prov_opts))
    try:
        result_json = _run_native(
            _gen_image,
            handle,
            prompt,
            model,
            size,
            quality,
            style,
            aspect_ratio,
            n,
            response_format,
        )
        data = json.loads(result_json)
        images = [
            GeneratedImageData(
                url=img.get("url"),
                base64=img.get("base64"),
                mime_type=img.get("mime_type") or img.get("mimeType"),
            )
            for img in data.get("images", [])
        ]
        return ImageGenerationResult(
            images=images,
            revised_prompt=data.get("revised_prompt") or data.get("revisedPrompt"),
        )
    finally:
        _run_native(destroy_provider, handle)
