from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class SparseIndexDocument:
    """One searchable sparse-index document."""

    record_id: str
    text: str
    hub_id: str
    path_hash: str

    @property
    def metadata(self) -> Dict[str, str]:
        return {
            "record_id": self.record_id,
            "hub_id": self.hub_id,
            "path_hash": self.path_hash,
        }


@dataclass(frozen=True)
class SparseIndexHit:
    """One scored result returned by a sparse index store."""

    record_id: str
    hub_id: str
    path_hash: str
    score: float


@dataclass(frozen=True)
class SparseIndexArtifact:
    """An index representation and aligned metadata ready for persistence."""

    index: Any
    records: List[Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SparseIndexStore(ABC):
    """Backend-neutral interface for building and querying a sparse index."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the store implementation name."""
        ...

    @property
    @abstractmethod
    def index_key(self) -> str:
        """Returns the stable cache identity for this index."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Returns whether a valid index is loaded and searchable."""
        ...

    @abstractmethod
    def store(self, artifact: SparseIndexArtifact) -> bool:
        """Persists and loads a channel-specific index artifact."""
        ...

    @abstractmethod
    def search(
            self,
            query_text: str,
            limit: int | None = None) -> List[SparseIndexHit]:
        """Returns path-level hits ordered from most to least relevant."""
        ...

    @abstractmethod
    def reload(self) -> bool:
        """Reloads the persisted index and reports whether it is valid."""
        ...
