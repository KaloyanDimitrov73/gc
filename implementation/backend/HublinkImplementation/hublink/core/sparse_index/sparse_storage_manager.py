from typing import List

from hublink.core.sparse_index.sparse_ranking import (
    build_sparse_search_result,
)
from hublink.core.sparse_index.sparse_index_key import (
    compute_sparse_index_key,
)
from hublink.core.sparse_index.sparse_search_result import (
    SparseSearchResult,
)
from knowledge_base.sparse_index_store.sparse_index_store import (
    SparseIndexArtifact,
    SparseIndexHit,
    SparseIndexStore,
)
from knowledge_base.sparse_index_store.sparse_index_store_provider import (
    SparseIndexStoreProvider,
    SparseIndexStoreType,
)


class SparseStorageManager:
    """Persists and retrieves interchangeable sparse index stores."""

    def __init__(
            self,
            index_key: str,
            bm25_store: SparseIndexStore | None = None,
            splade_store: SparseIndexStore | None = None) -> None:
        self.index_key = index_key
        self.bm25_store = bm25_store
        self.splade_store = splade_store
        self._validate_store_key(bm25_store)
        self._validate_store_key(splade_store)

    @classmethod
    def from_config(
            cls,
            config,
            load_bm25: bool = True,
            load_splade: bool = True) -> "SparseStorageManager":
        index_key = compute_sparse_index_key(config)
        return cls(
            index_key=index_key,
            bm25_store=(
                SparseIndexStoreProvider.get_sparse_index_store(
                    store_type=SparseIndexStoreType.BM25.value,
                    index_key=index_key,
                )
                if load_bm25
                else None
            ),
            splade_store=(
                SparseIndexStoreProvider.get_sparse_index_store(
                    store_type=SparseIndexStoreType.SPLADE.value,
                    index_key=index_key,
                )
                if load_splade
                else None
            ),
        )

    @staticmethod
    def compute_index_key(config) -> str:
        return compute_sparse_index_key(config)

    @property
    def has_bm25(self) -> bool:
        return self.bm25_store is not None and self.bm25_store.is_available

    @property
    def has_splade(self) -> bool:
        return self.splade_store is not None and self.splade_store.is_available

    def store_bm25(self, artifact: SparseIndexArtifact) -> bool:
        if self.bm25_store is None:
            self.bm25_store = SparseIndexStoreProvider.get_sparse_index_store(
                store_type=SparseIndexStoreType.BM25.value,
                index_key=self.index_key,
                load_index=False,
            )
        return self.bm25_store.store(artifact)

    def store_splade(self, artifact: SparseIndexArtifact) -> bool:
        if self.splade_store is None:
            self.splade_store = SparseIndexStoreProvider.get_sparse_index_store(
                store_type=SparseIndexStoreType.SPLADE.value,
                index_key=self.index_key,
                load_index=False,
            )
        return self.splade_store.store(artifact)

    def reload_bm25(self) -> bool:
        return self.bm25_store is not None and self.bm25_store.reload()

    def reload_splade(self) -> bool:
        return self.splade_store is not None and self.splade_store.reload()

    def search_bm25(
            self,
            query_text: str,
            top_hubs: int,
            paths_per_hub: int) -> SparseSearchResult:
        hits = (
            self.bm25_store.search(query_text)
            if self.bm25_store is not None
            else []
        )
        return self._to_search_result(hits, top_hubs, paths_per_hub)

    def search_splade(
            self,
            query_text: str,
            top_hubs: int,
            paths_per_hub: int) -> SparseSearchResult:
        hits = (
            self.splade_store.search(query_text)
            if self.splade_store is not None
            else []
        )
        return self._to_search_result(hits, top_hubs, paths_per_hub)

    def _validate_store_key(self, store: SparseIndexStore | None) -> None:
        if store is not None and store.index_key != self.index_key:
            raise ValueError(
                f"Sparse store key '{store.index_key}' does not match manager "
                f"key '{self.index_key}'."
            )

    @staticmethod
    def _to_search_result(
            hits: List[SparseIndexHit],
            top_hubs: int,
            paths_per_hub: int) -> SparseSearchResult:
        return build_sparse_search_result(
            scores=[hit.score for hit in hits],
            hub_ids=[hit.hub_id for hit in hits],
            path_hashes=[hit.path_hash for hit in hits],
            top_hubs=top_hubs,
            paths_per_hub=paths_per_hub,
        )
