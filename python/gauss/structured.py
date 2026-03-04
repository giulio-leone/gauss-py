"""
Structured Output — validate and extract typed JSON from LLM responses.

Example::

    from gauss import structured

    result = structured(agent, "List 3 fruits", schema={
        "type": "object",
        "properties": {"fruits": {"type": "array", "items": {"type": "string"}}}
    })
    print(result["data"]["fruits"])  # ["apple", "banana", "cherry"]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

__all__ = ["StructuredConfig", "StructuredResult", "structured"]

_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?([\s\S]*?)\n?\s*```")


@dataclass
class StructuredConfig:
    """Configuration for structured output extraction.

    Attributes:
        schema: JSON schema dict the output must conform to.
        max_parse_retries: Max retries if the model returns invalid JSON (default: 2).
        include_raw: If True, include the raw AgentResult in the output.
    """

    schema: dict[str, Any]
    max_parse_retries: int = 2
    include_raw: bool = False


@dataclass
class StructuredResult:
    """Result of a structured extraction.

    Attributes:
        data: Parsed and validated data dict.
        raw: Raw AgentResult (only if include_raw was True).
    """

    data: Any
    raw: Any = None


def _build_structured_prompt(user_prompt: str, schema: dict[str, Any]) -> str:
    schema_str = json.dumps(schema, indent=2)
    return (
        f"{user_prompt}\n\n"
        f"Respond ONLY with valid JSON matching this schema:\n{schema_str}\n\n"
        f"Do not include any text outside the JSON object."
    )


def _extract_json(text: str) -> str:
    """Extract JSON from text, handling code blocks and embedded JSON."""
    # Try markdown code blocks first
    code_match = _CODE_BLOCK_RE.search(text)
    if code_match:
        return code_match.group(1).strip()

    # Find first { or [
    obj_start = text.find("{")
    arr_start = text.find("[")

    if obj_start == -1 and arr_start == -1:
        return text.strip()

    if obj_start == -1:
        start = arr_start
    elif arr_start == -1:
        start = obj_start
    else:
        start = min(obj_start, arr_start)

    opener = text[start]
    closer = "]" if opener == "[" else "}"

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return text[start:]


def structured(
    agent: Any,
    prompt: str,
    *,
    schema: dict[str, Any] | None = None,
    config: StructuredConfig | None = None,
    max_parse_retries: int = 2,
    include_raw: bool = False,
) -> StructuredResult:
    """Run an agent with structured output extraction.

    Automatically instructs the model to output JSON matching the schema,
    extracts and parses the JSON from the response, and retries on parse failure.

    Example::

        result = structured(agent, "List 3 fruits", schema={
            "type": "object",
            "properties": {"fruits": {"type": "array", "items": {"type": "string"}}},
            "required": ["fruits"],
        })
        print(result.data["fruits"])

    Args:
        agent: An Agent instance.
        prompt: The user prompt.
        schema: JSON schema dict (alternative to config).
        config: Full StructuredConfig (alternative to schema).
        max_parse_retries: Retries on parse failure (default: 2).
        include_raw: Include raw AgentResult (default: False).

    Returns:
        StructuredResult with .data and optional .raw
    """
    if config:
        cfg = config
    elif schema:
        cfg = StructuredConfig(
            schema=schema,
            max_parse_retries=max_parse_retries,
            include_raw=include_raw,
        )
    else:
        raise ValueError("Either schema or config must be provided")

    last_error: Exception | None = None

    for attempt in range(cfg.max_parse_retries + 1):
        structured_prompt = _build_structured_prompt(prompt, cfg.schema)
        if attempt > 0 and last_error:
            structured_prompt += (
                f"\n\nPrevious attempt failed: {last_error}. Please output ONLY valid JSON."
            )

        result = agent.run(structured_prompt)

        try:
            json_str = _extract_json(result.text)
            data = json.loads(json_str)
            return StructuredResult(
                data=data,
                raw=result if cfg.include_raw else None,
            )
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e

    raise RuntimeError(
        f"Failed to extract structured output after "
        f"{cfg.max_parse_retries + 1} attempts: {last_error}"
    )
