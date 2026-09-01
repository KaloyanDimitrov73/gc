"""Pure retrieval metrics shared by experiments and prompt optimization."""

from __future__ import annotations

from collections.abc import Sequence


def _as_list(value: Sequence[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    return list(value)


def hit_at_k(
    contexts: Sequence[str] | str | None,
    golden: Sequence[str] | str | None,
    *,
    k: int,
) -> float:
    """Return the fraction of gold strings found in the first ``k`` contexts."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    context_values = _as_list(contexts)
    golden_values = _as_list(golden)
    if not golden_values:
        return 1.0 if not context_values else 0.0
    if not context_values:
        return 0.0

    top_contexts = [context.lower() for context in context_values[:k]]
    hits = sum(
        any(gold.lower() in context for context in top_contexts)
        for gold in golden_values
    )
    return hits / len(golden_values)
