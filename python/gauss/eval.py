"""Evaluation runner for LLM quality assessment."""

from __future__ import annotations

import functools
import json
from typing import Any

from gauss.base import StatefulResource


class EvalRunner(StatefulResource):
    """Run evaluations with configurable scorers and datasets.

    Example::

        runner = EvalRunner(threshold=0.8).add_scorer("exact_match")
        dataset = EvalRunner.load_dataset_jsonl("data.jsonl")
    """

    def __init__(self, threshold: float = 0.8) -> None:
        super().__init__()
        from gauss._native import create_eval_runner

        self._handle: int = create_eval_runner(threshold)

    @functools.cached_property
    def _resource_name(self) -> str:
        return "EvalRunner"

    def add_scorer(self, scorer_type: str) -> EvalRunner:
        """Add a scorer to the evaluation. Returns self for chaining."""
        from gauss._native import eval_add_scorer

        self._check_alive()
        eval_add_scorer(self._handle, scorer_type)
        return self

    @staticmethod
    def load_dataset_jsonl(data: str) -> list[dict[str, Any]]:
        """Load evaluation dataset from JSONL string."""
        from gauss._native import load_dataset_jsonl

        result_json: str = load_dataset_jsonl(data)
        return json.loads(result_json)  # type: ignore[no-any-return]

    @staticmethod
    def load_dataset_json(data: str) -> list[dict[str, Any]]:
        """Load evaluation dataset from JSON string."""
        from gauss._native import load_dataset_json

        result_json: str = load_dataset_json(data)
        return json.loads(result_json)  # type: ignore[no-any-return]

    @property
    def handle(self) -> int:
        return self._handle

    def destroy(self) -> None:
        if not self._destroyed:
            from gauss._native import destroy_eval_runner

            destroy_eval_runner(self._handle)
        super().destroy()
