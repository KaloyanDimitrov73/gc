"""Retrieval metrics used as the DSPy optimization objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hublink.evaluation import hit_at_k


@dataclass(frozen=True)
class HitAtKTripleMetric:
    """DSPy-compatible Hit@k score over ranked retrieved triples."""

    k: int = 10

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("k must be greater than zero.")

    def __call__(
        self,
        gold: Any,
        pred: Any,
        trace: Any = None,
        pred_name: str | None = None,
        pred_trace: Any = None,
    ) -> float:
        del trace, pred_name, pred_trace
        if not getattr(pred, "valid_output", True):
            return 0.0

        return hit_at_k(
            getattr(pred, "retrieved_triples", []),
            getattr(gold, "expected_triples", []),
            k=self.k,
        )
