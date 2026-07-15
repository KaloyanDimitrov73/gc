from abc import ABC, abstractmethod
from dataclasses import field, dataclass
from typing import Dict, List, Optional, Any

from knowledge_base.vector_store.storage.utils.filters import WhereFilter


@dataclass
class VectorScoreResults:
    ids: List[str] = field(default_factory=list)
    metadata: List[Dict] = field(default_factory=list)
    embeddings: Optional[List[List[float]]] = None
    distances: Optional[List[float]] = None

    @property
    def is_empty(self) -> bool:
        return len(self.ids) == 0


class VectorStore(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """
        A stable identifier for this store instance (e.g. the persist directory
        name, dataset name, or index name - whatever uniquely identifies THIS
        store, as opposed to the collection/schema name, which may be shared
        across many store instances). Safe to use as e.g. a cache key prefix.
        """
        ...

    @abstractmethod
    def _initialize(self) -> None:
        """
        Sets up the underlying client/collection (or equivalent) for the vector store.
        Called once during construction. Implementations should be idempotent —
        calling it again should not recreate an already-initialized client/collection.
        """
        ...

    @abstractmethod
    def store_data(self, record_id: str, embedding: List[float],
                   metadata: Optional[Dict] = None) -> None:
        """
        Stores a single embedding under the given id. If an entry with this id
        already exists, it is overwritten (upsert semantics).

        Args:
            record_id (str): Unique identifier for this entry.
            embedding (List[float]): The embedding vector to store.
            metadata (Dict, optional): Additional metadata associated with the entry.
        """
        ...

    @abstractmethod
    def store_data_batch(self, ids: List[str], embeddings: List[List[float]],
                         metadatas: List[Dict]) -> None:
        """
        Stores multiple embeddings in a single batch operation (upsert semantics).
        All three lists must have the same length and be aligned by index
        (ids[i] corresponds to embeddings[i] and metadatas[i]).

        Args:
            ids (List[str]): Unique identifiers for each entry.
            embeddings (List[List[float]]): The embedding vectors to store.
            metadatas (List[Dict]): Metadata dict for each entry.
        """
        ...

    @abstractmethod
    def delete_records_with_filter(self, where_filter: WhereFilter) -> None:
        """
        Deletes all entries whose metadata matches the given filter condition.

        Args:
            where_filter (Dict): Metadata filter condition. Implementation-specific
                filter syntax is translated by the adapter.
        """
        ...

    @abstractmethod
    def get_records_with_metadata_by_filter(self, where_filter: WhereFilter,
                                            limit: Optional[int] = None) -> Optional[VectorScoreResults]:
        """
        Retrieves entries (without similarity search) whose metadata matches
        the given filter condition. Does not return a similarity score, since
        no query embedding is involved.

        Args:
            where_filter (Dict): Metadata filter condition.
            limit (int, optional): Maximum number of entries to return.
                If None, all matching entries are returned.

        Returns:
            VectorScoreResults: The matching entries, or None if nothing matched.
        """
        ...

    @abstractmethod
    def get_records_with_metadata_by_ids(self, ids: List[str]) -> Optional[VectorScoreResults]:
        """
        Retrieves a single entry by its unique id.

        Args:
            ids (str): The unique identifier of the entry to retrieve.

        Returns:
            VectorScoreResults: The matching entry, or None if no entry with this id exists.
        """
        ...

    @abstractmethod
    def vector_similarity_search(self, query_embeddings: List[List[float]],
                                 where_filter: Optional[WhereFilter] = None,
                                 n_results: int = 10) -> List[VectorScoreResults]:
        """
        Performs a similarity search, returning the entries most similar to the
        given query embedding(s), optionally restricted by a metadata filter.

        Args:
            query_embeddings (List[List[float]]): One or more query embedding vectors.
            where_filter (Dict, optional): Metadata filter condition to restrict the search.
            n_results (int): Maximum number of similar entries to return.

        Returns:
            List[VectorScoreResults]: One VectorScoreResults per query embedding
                (same order as query_embeddings), each ordered by similarity
                (most similar first) and including distances.
        """
        ...

    @abstractmethod
    def rebuild(self) -> None:
        """Rebuilds/defragments the underlying index, if the backend needs it.
        Pass for backends that don't require this."""
        ...

    @abstractmethod
    def ensure_distance_metric(self, distance_metric: str) -> None:
        """Ensures the index uses the given distance metric, migrating if needed."""
        ...

    @abstractmethod
    def print_stats(self, verbose: bool = True) -> None:
        """Prints diagnostic info about the index health."""
        ...

    @abstractmethod
    def _translate_filter(self, where_filter: Optional[WhereFilter]) -> Any:
        """
        Translates a backend-independent WhereFilter into the native
        filter syntax of this vector store backend.
        """
        ...