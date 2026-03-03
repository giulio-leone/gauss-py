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
    _default_model,
    detect_provider,
    resolve_api_key,
)
from gauss._utils import _run_native


def execute_code(
    language: str,
    code: str,
    *,
    timeout: int = 30,
    sandbox: str = "default",
) -> CodeExecutionResult:
    """Execute code in a sandboxed runtime and return the result.

    Runs the given source code using a local runtime managed by the
    Gauss native engine.  No :class:`Agent` instance is required.

    Args:
        language: Runtime identifier — ``"python"``, ``"javascript"``,
            or ``"bash"``.
        code: Source code to execute.
        timeout: Maximum execution time in seconds.  Defaults to ``30``.
        sandbox: Sandbox policy — ``"default"``, ``"strict"``, or
            ``"permissive"``.

    Returns:
        :class:`CodeExecutionResult` with ``stdout``, ``stderr``,
        ``exit_code``, ``timed_out``, ``runtime``, and ``success``
        fields.

    Example:
        >>> result = execute_code("python", "print(42)")
        >>> assert result.stdout.strip() == "42"
    """
    from gauss._native import execute_code as _exec

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
    """Return the code-execution runtimes available on this system.

    Queries the native engine for installed language runtimes that
    can be used with :func:`execute_code`.

    Returns:
        A list of runtime name strings, e.g. ``["python", "bash"]``.

    Example:
        >>> runtimes = available_runtimes()
        >>> "python" in runtimes
        True
    """
    from gauss._native import available_runtimes as _runtimes

    result_json = _run_native(_runtimes)
    return json.loads(result_json)


def version() -> str:
    """Return the native gauss-core runtime version."""
    from gauss._native import version as _version

    return str(_run_native(_version))


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
    """Generate images from a text prompt using a provider's image API.

    Creates a temporary provider connection, sends the generation
    request, and returns the resulting images.  No :class:`Agent`
    instance is required.

    Args:
        prompt: Text description of the desired image.
        provider: LLM provider to use.  Auto-detected from environment
            when ``None``.
        model: Model name (e.g. ``"dall-e-3"``).  Defaults to the
            provider's default model.
        api_key: API key override.  Resolved from environment when
            ``None``.
        base_url: Custom base URL for the provider API.
        size: Image dimensions, e.g. ``"1024x1024"``.
        quality: Quality preset (provider-specific, e.g. ``"hd"``).
        style: Style preset (provider-specific, e.g. ``"vivid"``).
        aspect_ratio: Aspect ratio string, e.g. ``"16:9"`` (Gemini).
        n: Number of images to generate.
        response_format: Desired format — ``"url"`` or ``"b64_json"``.

    Returns:
        :class:`ImageGenerationResult` containing a list of
        :class:`GeneratedImageData` objects (each with ``url`` and/or
        ``base64``) and an optional ``revised_prompt``.

    Example:
        >>> result = generate_image("A sunset over mountains", model="dall-e-3")
        >>> print(result.images[0].url)
    """
    from gauss._native import (
        create_provider,
        destroy_provider,
    )
    from gauss._native import (
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
