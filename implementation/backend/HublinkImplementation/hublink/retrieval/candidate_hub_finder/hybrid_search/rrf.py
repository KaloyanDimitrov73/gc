from collections import defaultdict
from typing import Iterable, Sequence


def rrf_score_from_ranks(
        ranks: Iterable[int],
        k: int = 60) -> float:
    """Returns an RRF score for zero-based ranks from comparable rankings."""
    if k <= 0:
        raise ValueError("RRF k must be greater than zero.")
    ranks = list(ranks)
    if any(rank < 0 for rank in ranks):
        raise ValueError("RRF ranks must be zero or greater.")
    return sum(1.0 / (k + rank + 1) for rank in ranks)


def normalize_rrf_score(
        score: float,
        channel_count: int,
        k: int = 60) -> float:
    """Normalizes an RRF score by the best score across all channels."""
    if k <= 0:
        raise ValueError("RRF k must be greater than zero.")
    if channel_count <= 0:
        raise ValueError("RRF channel_count must be greater than zero.")
    return score / (channel_count / (k + 1))


def two_way_rrf(
        wrapped_ranking: Sequence[str],
        sparse_ranking: Sequence[str],
        k: int = 60) -> list[str]:
    """Fuses two ID rankings using deterministic Reciprocal Rank Fusion."""
    if k <= 0:
        raise ValueError("RRF k must be greater than zero.")

    rankings = (
        wrapped_ranking,
        sparse_ranking,
    )
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}

    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] += 1.0 / (k + rank + 1)
            if item_id not in first_seen:
                first_seen[item_id] = len(first_seen)

    return sorted(
        scores,
        key=lambda item_id: (-scores[item_id], first_seen[item_id])
    )
