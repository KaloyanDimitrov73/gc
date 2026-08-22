from collections import defaultdict
from typing import Sequence

from .sparse_search_result import SparsePathHit, SparseSearchResult


def build_sparse_search_result(
        scores: Sequence[float],
        hub_ids: list[str],
        path_hashes: list[str],
        top_hubs: int,
        paths_per_hub: int) -> SparseSearchResult:
    """Aggregates aligned path scores into a hub ranking and path hits."""
    if not (len(scores) == len(hub_ids) == len(path_hashes)):
        raise ValueError(
            "Sparse scores, hub IDs and path hashes must have equal lengths."
        )

    rows_by_hub: dict[str, list[tuple[int, float]]] = defaultdict(list)

    for row_index, raw_score in enumerate(scores):
        hub_id = hub_ids[row_index]
        score = float(raw_score)
        if score <= 0:
            continue
        rows_by_hub[hub_id].append((row_index, score))

    ranked_hub_ids = sorted(
        rows_by_hub,
        key=lambda hub_id: max(
            score for _, score in rows_by_hub[hub_id]
        ),
        reverse=True
    )[:top_hubs]

    ranked_rows_by_hub = {
        hub_id: sorted(
            rows_by_hub[hub_id],
            key=lambda row: row[1],
            reverse=True
        )[:paths_per_hub]
        for hub_id in ranked_hub_ids
    }
    globally_ranked_rows = sorted(
        (
            row
            for ranked_rows in ranked_rows_by_hub.values()
            for row in ranked_rows
        ),
        key=lambda row: (-row[1], row[0])
    )
    global_rank_by_row = {
        row_index: rank
        for rank, (row_index, _) in enumerate(globally_ranked_rows)
    }

    hits_by_hub: dict[str, list[SparsePathHit]] = {}
    for hub_id in ranked_hub_ids:
        ranked_rows = ranked_rows_by_hub[hub_id]
        hits_by_hub[hub_id] = [
            SparsePathHit(
                path_hash=path_hashes[row_index],
                hub_id=hub_id,
                score=score,
                rank=rank,
                global_rank=global_rank_by_row[row_index]
            )
            for rank, (row_index, score) in enumerate(ranked_rows)
        ]

    return SparseSearchResult(
        ranked_hub_ids=ranked_hub_ids,
        hits_by_hub=hits_by_hub
    )
