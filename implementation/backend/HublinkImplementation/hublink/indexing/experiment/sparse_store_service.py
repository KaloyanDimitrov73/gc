from hublink.core.models.hub_link_settings import HubLinkSettings
from hublink.core.sparse_index.sparse_storage_manager import SparseStorageManager


class SparseStoreService:
    """Creates the sparse index stores used by experiment runs."""

    def __init__(
            self,
            config,
            load_bm25: bool | None = None,
            load_splade: bool | None = None) -> None:
        self.config = config
        self.settings = HubLinkSettings.from_config(config)
        self.sparse_storage_manager = SparseStorageManager.from_config(
            config=config,
            load_bm25=(
                self.settings.use_bm25_hybrid_search
                if load_bm25 is None
                else load_bm25
            ),
            load_splade=(
                self.settings.use_splade_hybrid_search
                if load_splade is None
                else load_splade
            ),
        )
