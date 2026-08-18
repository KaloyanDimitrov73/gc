"""
Service layer for creating and accessing the vector store backend used by
HubLink. Backend selection is driven by configuration (``vector_store_backend``)
instead of being hardcoded, so the concrete implementation (Chroma, ...) can be
swapped without touching indexing or retrieval code.

This service is intentionally the ONLY place that constructs a ``HubStorageManager``.
It is shared (via the DI layer) between the indexing write-path and the
retrieval read-path, so those two stay decoupled from each other's internals.
"""
from typing import Any

from backend.app.config.base_settings import get_settings
from backend.app.config.hublink.config_loader import HublinkConfigLoader
from hublink.core.models.hub_link_settings import HubLinkSettings
from hublink.core.hub_storage_manager import HubStorageManager
from knowledge_base.vector_store.storage.vector_store_provider import VectorStoreProvider
from retrieval.config.kg_retrieval_config import KGRetrievalConfig


class HubStoreService:
    """
    Owns creation of the configured vector store backend and the
    ``HubStorageManager`` built on top of it.
    """

    def __init__(self, config: KGRetrievalConfig):
        self.settings = get_settings()
        self.hub_storage_manager: HubStorageManager
        self.store_name: str
        self._prepare_vector_store(config)

    def _prepare_vector_store(self, config: Any) -> None:
        """
        Loads the retrieval config, resolves the deterministic store name,
        and builds the backend vector store and initialize HubStorageManager.
        """
        hublink_settings = HubLinkSettings.from_config(config)

        store_name = VectorStoreProvider.compute_store_name(config, hublink_settings)

        vector_store = VectorStoreProvider.get_vector_store(
            backend=self.settings.vector_store_backend,
            store_name=store_name,
            distance_metric=hublink_settings.distance_metric,
        )
        self.hub_storage_manager = HubStorageManager(
            vector_store=vector_store,
            embedding_config=hublink_settings.embedding_config,
            diversity_penalty=hublink_settings.diversity_ranking_penalty,
        )
        self.store_name = store_name