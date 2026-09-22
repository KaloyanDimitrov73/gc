from dataclasses import dataclass, field


@dataclass(frozen=True)
class SparsePathHit:
    """One path-level hit returned by a sparse retrieval channel."""

    path_hash: str
    hub_id: str
    score: float
    rank: int
    global_rank: int | None = None


@dataclass
class SparseSearchResult:
    """Hub ranking plus the path hits that produced that ranking."""

    ranked_hub_ids: list[str]
    hits_by_hub: dict[str, list[SparsePathHit]] = field(
        default_factory=dict
    )
