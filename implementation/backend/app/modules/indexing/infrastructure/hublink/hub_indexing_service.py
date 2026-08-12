"""
Infrastructure service that talks to the HubLink HubIndexer.

This is the boundary between our async orchestration layer
(IndexingService) and the synchronous, CPU/IO-heavy HubLink indexing
code (HubIndexer / HubBuilder / HubFinder).
"""
from __future__ import annotations

from typing import Optional

from backend.app.config.base_settings import get_settings
from core.logging.logging import get_logger
from hublink.core.hub_storage_manager import HubStorageManager
from hublink.core.models.hub_link_settings import HubLinkSettings
from hublink.indexing.hub_indexer import HubIndexer, HubIndexerOptions
from hublink.indexing.util.root_entity_util import resolve_root_entities
from knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph

from backend.app.config.hublink.config_loader import HublinkConfigLoader

from backend.app.shared.exceptions import AppError

logger = get_logger(__name__)


class HubIndexingServiceError(AppError):
    """Raised when the underlying HubIndexer run fails to even start."""


class HubIndexingService:
    """
    Service for interacting with the HubLink indexing system.

    This service:
    - Receives an already-loaded knowledge graph from GraphLoadService
    - Receives HubStorageManager from VectorStoreService

    Responsibilities:
      - Create `HubIndexerOptions` from config and with HubStorageManager
      - Trigger initial indexing if VectorStore hold by HubstorageManager is empty
      - Run the hub indexing process

    Note: `run_indexing` is blocking and can take a while on large graphs.
    Use IndexingService to run it off the event loop via an executor.
    """

    def __init__(self, graph: KnowledgeGraph, hub_storage_manager: HubStorageManager):
        self.config_loader = HublinkConfigLoader()
        self.settings = get_settings()
        self.hub_storage_manager = hub_storage_manager
        self.hublink_settings: HubLinkSettings
        self.options = self._initialize_options()
        self._ensure_indexed(graph)


    def _initialize_options(self) -> HubIndexerOptions:
        """
        Loads the configs and HubStorageManager to build the indexing options.
        """
        config = self.config_loader.load_kg_retrieval_config()
        self.hublink_settings = HubLinkSettings.from_config(config)

        if self.hub_storage_manager is None:
            raise RuntimeError("No hub storage manager provided to HubIndexingService")

        return HubIndexerOptions.from_settings(
            self.hublink_settings,
            self.hub_storage_manager,
            config.llm_config,
            self.hublink_settings.embedding_config
        )

    def _ensure_indexed(self, graph: KnowledgeGraph) -> Optional[int]:
        """
        Triggers a full indexing run if the vector store is currently empty.
        Intended to be called e.g. on startup or before the first retrieval.

        Returns:
            The result of `run()` if indexing was triggered, otherwise None.
        """
        if self.hub_storage_manager is None:
            raise RuntimeError("No hub storage manager provided to HubIndexingService")

        if not self.hub_storage_manager.vector_store_is_empty():
            return None

        logger.info("Hub vector store is empty - triggering indexing run.")
        return self.run_indexing(graph=graph)

    def run_indexing(self, graph: KnowledgeGraph, force_update: bool = False) -> int:
        """
        Runs a full hub indexing pass.

        Args:
            graph: The loaded knowledgeGraph.
            force_update: Whether to force-update hubs that are already cached.

        Returns:
            Number of hubs indexed during this run.

        Raises:
            HubIndexingServiceError: if the run could not be started (e.g. no root entities resolved).
        """
        if graph is None:
            raise RuntimeError("No knowledge graph provided to HubLinkService")

        try:
            logger.info(
                "Starting indexing run force_update=%s", force_update,
            )

            root_entities = resolve_root_entities(
                graph=graph,
                root_entity_ids=self.hublink_settings.indexing_root_entity_ids,
                root_entity_types=self.hublink_settings.indexing_root_entity_types,
            )

            if not root_entities:
                return 0

            logger.info("Indexing %d root entities", len(root_entities))
            hub_indexer = HubIndexer(graph=graph, options=self.options)

            hub_indexer.run_indexing(
                root_entities=root_entities,
                force_index_update=force_update or self.hublink_settings.force_index_update
            )
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            raise HubIndexingServiceError(f"HubIndexer run failed: {e}") from e

        return len(root_entities)