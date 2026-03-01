"""Content safety guardrails."""

from __future__ import annotations

import json
from typing import Any


class GuardrailChain:
    """Chainable content safety guardrails.

    Example::

        chain = (
            GuardrailChain()
            .add_content_moderation(blocked_categories=["violence"])
            .add_pii_detection(action="redact")
            .add_token_limit(max_input=4000, max_output=2000)
            .add_regex_filter(patterns=[r"\\bpassword\\b"])
            .add_schema({"type": "object", "required": ["answer"]})
        )
        print(chain.list())
    """

    def __init__(self) -> None:
        from gauss._native import create_guardrail_chain  # type: ignore[import-not-found]

        self._handle: int = create_guardrail_chain()
        self._destroyed = False

    def add_content_moderation(
        self,
        blocked_categories: list[str] | None = None,
        warned_categories: list[str] | None = None,
    ) -> GuardrailChain:
        """Add content moderation guardrail. Returns self."""
        from gauss._native import (
            guardrail_chain_add_content_moderation,  # type: ignore[import-not-found]
        )

        self._check_alive()
        guardrail_chain_add_content_moderation(
            self._handle,
            json.dumps(blocked_categories or []),
            json.dumps(warned_categories or []),
        )
        return self

    def add_pii_detection(self, action: str = "redact") -> GuardrailChain:
        """Add PII detection guardrail. Returns self."""
        from gauss._native import (
            guardrail_chain_add_pii_detection,  # type: ignore[import-not-found]
        )

        self._check_alive()
        guardrail_chain_add_pii_detection(self._handle, action)
        return self

    def add_token_limit(self, max_input: int = 4000, max_output: int = 2000) -> GuardrailChain:
        """Add token limit guardrail. Returns self."""
        from gauss._native import guardrail_chain_add_token_limit  # type: ignore[import-not-found]

        self._check_alive()
        guardrail_chain_add_token_limit(self._handle, max_input, max_output)
        return self

    def add_regex_filter(self, patterns: list[str]) -> GuardrailChain:
        """Add regex-based content filter. Returns self."""
        from gauss._native import guardrail_chain_add_regex_filter  # type: ignore[import-not-found]

        self._check_alive()
        guardrail_chain_add_regex_filter(self._handle, json.dumps(patterns))
        return self

    def add_schema(self, schema: dict[str, Any]) -> GuardrailChain:
        """Add JSON schema validation guardrail. Returns self."""
        from gauss._native import guardrail_chain_add_schema  # type: ignore[import-not-found]

        self._check_alive()
        guardrail_chain_add_schema(self._handle, json.dumps(schema))
        return self

    def list(self) -> list[str]:
        """List active guardrail names."""
        from gauss._native import guardrail_chain_list  # type: ignore[import-not-found]

        self._check_alive()
        result_json: str = guardrail_chain_list(self._handle)
        return json.loads(result_json)  # type: ignore[no-any-return]

    def destroy(self) -> None:
        if not self._destroyed:
            from gauss._native import destroy_guardrail_chain  # type: ignore[import-not-found]

            destroy_guardrail_chain(self._handle)
            self._destroyed = True

    def __enter__(self) -> GuardrailChain:
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    def __del__(self) -> None:
        self.destroy()

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("GuardrailChain has been destroyed")
