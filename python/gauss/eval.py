"""Evaluation runner for LLM quality assessment."""

from __future__ import annotations

import json
from typing import Any


class EvalRunner:
    """Run evaluations with configurable scorers and datasets.

    Example::

        runner = EvalRunner(threshold=0.8).add_scorer("exact_match")
        dataset = EvalRunner.load_dataset_jsonl("data.jsonl")
    """

    def __init__(self, threshold: float = 0.8) -> None:
        from gauss._native import create_eval_runner

        self._handle: int = create_eval_runner(threshold)
        self._destroyed = False

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
            self._destroyed = True

    def __enter__(self) -> EvalRunner:
        return self

    def __exit__(self, *_: Any) -> None:
        self.destroy()

    def __del__(self) -> None:
        self.destroy()

    def _check_alive(self) -> None:
        if self._destroyed:
            raise RuntimeError("EvalRunner has been destroyed")
