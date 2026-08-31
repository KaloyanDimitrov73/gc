from typing import Iterable


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
