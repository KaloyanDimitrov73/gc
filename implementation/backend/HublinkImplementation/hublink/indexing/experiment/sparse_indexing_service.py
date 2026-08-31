from core.logging.logging import get_logger
from hublink.core.hub_storage_manager import HubStorageManager
from hublink.core.models.hub_link_settings import HubLinkSettings
from hublink.core.sparse_index.sparse_index_key import (
    compute_sparse_index_key,
)
from hublink.core.sparse_index.sparse_storage_manager import SparseStorageManager
from hublink.indexing.sparse_index.bm25_indexer import Bm25Indexer
from hublink.indexing.sparse_index.sparse_index_documents import (
    collect_sparse_index_documents_from_paths,
)
from hublink.indexing.sparse_index.splade_indexer import SpladeIndexer

logger = get_logger(__name__)


class SparseIndexingService:
    """Builds experiment sparse indexes from paths in the dense hub cache."""

    def __init__(
            self,
            config,
            hub_storage_manager: HubStorageManager,
            sparse_storage_manager: SparseStorageManager,
            bm25_indexer: Bm25Indexer | None = None,
            splade_indexer: SpladeIndexer | None = None) -> None:
        self.settings = HubLinkSettings.from_config(config)
        self.hub_storage_manager = hub_storage_manager
        self.index_key = compute_sparse_index_key(config)
        if sparse_storage_manager.index_key != self.index_key:
            raise ValueError(
                f"Sparse storage key '{sparse_storage_manager.index_key}' "
                f"does not match config key '{self.index_key}'."
            )
        self.sparse_storage_manager = sparse_storage_manager
        self.bm25_indexer = bm25_indexer
        self.splade_indexer = splade_indexer

    def run_indexing(
            self,
            build_bm25: bool | None = None,
            build_splade: bool | None = None) -> int:
        if build_bm25 is None:
            build_bm25 = self.settings.use_bm25_hybrid_search
        if build_splade is None:
            build_splade = self.settings.use_splade_hybrid_search

        paths_by_hub = self.hub_storage_manager.get_all_hub_paths()
        documents = collect_sparse_index_documents_from_paths(paths_by_hub)
        if not documents:
            logger.warning(
                "No cached hub paths found; run dense hub indexing first."
            )
            return 0

        if build_bm25:
            bm25_indexer = self.bm25_indexer or Bm25Indexer(
                self.sparse_storage_manager
            )
            bm25_indexer.run_indexing(documents)

        if build_splade:
            splade_indexer = self.splade_indexer or SpladeIndexer(
                self.sparse_storage_manager
            )
            splade_indexer.run_indexing(documents)

        return len(documents)
