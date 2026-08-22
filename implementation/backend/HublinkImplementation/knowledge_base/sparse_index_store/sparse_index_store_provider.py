from enum import Enum

from knowledge_base.sparse_index_store.sparse_index_store import SparseIndexStore


class SparseIndexStoreType(str, Enum):
    BM25 = "bm25"
    SPLADE = "splade"

    @staticmethod
    def get_values() -> list[str]:
        return [store_type.value for store_type in SparseIndexStoreType]


class SparseIndexStoreProvider:
    """Creates sparse index stores without exposing concrete implementations."""

    @staticmethod
    def get_sparse_index_store(
            store_type: str,
            index_key: str,
            load_index: bool = True) -> SparseIndexStore:
        if store_type == SparseIndexStoreType.BM25.value:
            from knowledge_base.sparse_index_store.implementations.bm25_sparse_index_store import (
                Bm25SparseIndexStore,
            )
            return Bm25SparseIndexStore(
                index_key=index_key,
                load_index=load_index,
            )

        if store_type == SparseIndexStoreType.SPLADE.value:
            from knowledge_base.sparse_index_store.implementations.splade_sparse_index_store import (
                SpladeSparseIndexStore,
            )
            return SpladeSparseIndexStore(
                index_key=index_key,
                load_index=load_index,
            )

        raise ValueError(
            f"Unsupported sparse index store '{store_type}'. Supported stores: "
            f"{', '.join(SparseIndexStoreType.get_values())}."
        )
