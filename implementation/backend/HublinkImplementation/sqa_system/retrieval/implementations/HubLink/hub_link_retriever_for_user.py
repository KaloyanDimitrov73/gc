"""
HubLink Retriever for GUI/User Applications.

This is a modified version of HubLinkRetriever designed for interactive use cases
where users need to dynamically change LLM models and retrieval settings without
reinitializing the entire retriever.

Key Features:
- Separate LLM for indexing vs retrieval (can change retrieval LLM without re-indexing)
- Dynamic settings modification at runtime
- Per-query setting overrides
- Status and info methods for GUI feedback
"""
import threading
from typing import Optional, List, Dict, Any
from typing_extensions import override
from copy import deepcopy

from core.data.models import RetrievalAnswer
from language_model import LLMProvider
from language_model.config.llm_config import LLMConfig
from sqa_system.retrieval import KnowledgeGraphRetriever
from knowledge_base.knowledge_graph.storage import KnowledgeGraph
from core.logging.logging import get_logger

from .models.hub_link_settings import HubLinkSettings, ADDITIONAL_CONFIG_PARAMS
from .models.hub import IsHubOptions
from .utils.hub_indexer import HubIndexer, HubIndexerOptions
from .utils.vector_store import ChromaVectorStore
from .utils.hub_source_handler import HubSourceHandler
from .retrieval.base_retrieval_strategy import RetrievalStrategyData
from .retrieval.traversal_retrieval_strategy import TraversalRetrievalStrategy
from .retrieval.direct_retrieval_strategy import DirectRetrievalStrategy
from ...config.kg_retrieval_config import KGRetrievalConfig

logger = get_logger(__name__)


class HubLinkRetrieverForUser(KnowledgeGraphRetriever):
    """
    HubLink Retriever optimized for GUI/interactive applications.

    This retriever allows dynamic modification of LLM and retrieval settings
    without requiring full reinitialization. It separates the indexing LLM
    from the retrieval LLM, allowing users to switch models for answer
    generation without re-indexing.

    Args:
        config (KGRetrievalConfig): Retrieval configuration.
        graph (KnowledgeGraph): The knowledge graph instance to query.
    """
    ADDITIONAL_CONFIG_PARAMS = ADDITIONAL_CONFIG_PARAMS

    def __init__(self, config: KGRetrievalConfig, graph: KnowledgeGraph) -> None:
        super().__init__(config, graph)

        self.settings: HubLinkSettings = HubLinkSettings.from_config(config)
        self._llm_provider = LLMProvider()

        # Indexing LLM (used only for building index, should not change after indexing)
        self._indexing_llm = self._llm_provider.get_llm_adapter(config.llm_config)

        # LLM cache for per-query LLM selection (avoids recreating adapters).
        # Populated immediately with the default config so the first query never
        # needs to create an adapter under the lock.
        # Cache key is the full llm_config hash, not just the model name.
        self._llm_cache: Dict[str, Any] = {}
        self._llm_cache_lock = threading.Lock()
        _default_llm = self._llm_provider.get_llm_adapter(config.llm_config)
        self._llm_cache[config.llm_config.config_hash] = _default_llm

        # Keep _retrieval_llm for the legacy retrieve_knowledge() override path.
        self._retrieval_llm = _default_llm
        self._retrieval_llm_config = deepcopy(config.llm_config)

        # Embedding model (should not change - would require re-indexing)
        self.embedding_model = self._llm_provider.get_embeddings(
            self.settings.embedding_config)

        self._prepare_vector_store()
        self.build_index(
            root_entity_types=self.settings.indexing_root_entity_types,
            root_entity_ids=self.settings.indexing_root_entity_ids
        )
        self._prepare_source_handler()


    # ==================== Retrieval ====================
    def query(self,
            query_text: str,
            number_of_hubs: int,
            retrieval_mode: str,
            llm_config: LLMConfig,
            topic_entity_id: Optional[str] = None,
            use_direct_final_answer: bool = False,
            ) -> RetrievalAnswer:
        """
        Main method to query the retriever with optional per-query settings overrides.

        Args:
            query_text (str): The user query.
            topic_entity_id (Optional[str]): Identifier for the topic entity.
            number_of_hubs (int): Number of hubs to retrieve.
            retrieval_mode (str): The retrieval mode ('direct' or 'graph').
            llm_config (LLMConfig): Full LLM configuration for answer generation.

        Returns:
            RetrievalAnswer: An object containing both the retrieved knowledge and the final answer.
        """
        # Build per-request local settings so concurrent calls never overwrite
        # each other's number_of_hubs / retrieval_mode on the shared instance.
        local_settings = deepcopy(self.settings)
        local_settings.number_of_hubs = number_of_hubs
        local_settings.use_topic_if_given = (retrieval_mode == "graph")
        local_settings.use_direct_final_answer = use_direct_final_answer

        # Resolve LLM adapter from the thread-safe cache.
        local_llm = self._get_llm_adapter(llm_config)

        # Log the retrieval attempt
        separator = "#" * 30
        strategy_name = "GraphTraversal" if local_settings.use_topic_if_given else "DirectRetrieval"
        logger.debug("\n%s\nNew HubLink Retrieval (%s)\n%s", separator, strategy_name, separator)
        logger.debug("Question: %s", query_text)
        logger.debug("LLM: %s", llm_config.name_model)
        logger.debug("Number of hubs: %s", local_settings.number_of_hubs)
        logger.info("HubLinkRetrieverForUser")

        retrieval_data = RetrievalStrategyData(
            graph=self.graph,
            llm_adapter=local_llm,
            embedding_adapter=self.embedding_model,
            settings=local_settings,
            vector_store=self.vector_store,
            source_handler=self.hub_source_handler,
        )

        if local_settings.use_topic_if_given and topic_entity_id:
            strategy = TraversalRetrievalStrategy(
                retrieval_data=retrieval_data,
                topic_entity_id=topic_entity_id,
            )
        else:
            strategy = DirectRetrievalStrategy(retrieval_data=retrieval_data)

        return strategy.retrieval(query_text)

    @override
    def retrieve_knowledge(
        self,
        query_text: str,
        topic_entity_id: Optional[str],
        topic_entity_value: Optional[str]
    ) -> RetrievalAnswer:
        """
        Retrieves knowledge based on the user query. Two strategies are employed:
        
        - Graph Traversal Strategy: Used when a topic entity ID is provided.
        - Direct Retrieval Strategy: Used when no topic entity ID is given.
        
        Note:
            The parameter 'topic_entity_value' is not needed by this
            retriever and is ignored.
        
        Args:
            query_text (str): The user query.
            topic_entity_id (Optional[str]): Identifier for the topic entity.
            topic_entity_value (Optional[str]): (Ignored) Value for the topic entity.
        
        Returns:
            RetrievalAnswer: An object containing both the retrieved knowledge and the final answer.
        """

        # There are two types of strategies that are supported
        # depending on whether a Topic Entity is given, or not.
        if self.settings.use_topic_if_given and topic_entity_id:
            strategy = TraversalRetrievalStrategy(
                retrieval_data=RetrievalStrategyData(
                    graph=self.graph,
                    llm_adapter=self._retrieval_llm,
                    embedding_adapter=self.embedding_model,
                    settings=self.settings,
                    vector_store=self.vector_store,
                    source_handler=self.hub_source_handler
                ),
                topic_entity_id=topic_entity_id
            )
            return strategy.retrieval(query_text)

        strategy = DirectRetrievalStrategy(
            retrieval_data=RetrievalStrategyData(
                graph=self.graph,
                llm_adapter=self._retrieval_llm,
                embedding_adapter=self.embedding_model,
                settings=self.settings,
                vector_store=self.vector_store,
                source_handler=self.hub_source_handler
            )
        )
        return strategy.retrieval(query_text)


    # ==================== Indexing ====================

    def build_index(
        self,
        root_entity_types: Optional[List[str]] = None,
        root_entity_ids: Optional[List[str]] = None,
        force_update: bool = False
    ):
        """
        Builds or updates the index for hub-based retrieval.

        Args:
            root_entity_types (Optional[List[str]]): Entity types to start indexing from.
            root_entity_ids (Optional[List[str]]): Specific entity IDs to start indexing from.
            force_update (bool): If True, forces re-indexing of all hubs.
        """
        if not root_entity_types and not root_entity_ids:
            raise ValueError(
                "Either root_entity_types or root_entity_ids must be specified."
            )
        if not self.vector_store:
            raise ValueError("Vector store not initialized.")

        logger.info("Building/Checking index for HubLink retriever")

        # Use indexing LLM for building the index
        hub_indexer = HubIndexer(
            graph=self.graph,
            options=HubIndexerOptions(
                embedding_model=self.embedding_model,
                is_hub_options=IsHubOptions(
                    hub_edges=self.settings.hub_edges,
                    types=self.settings.hub_types
                ),
                llm=self._indexing_llm,  # Always use indexing LLM
                max_workers=self.settings.max_workers,
                vector_store=self.vector_store,
                max_indexing_depth=self.settings.max_indexing_depth,
                max_hub_path_length=self.settings.max_hub_path_length,
                distance_metric=self.settings.distance_metric
            )
        )

        root_entities = []
        if root_entity_ids:
            for root_entity_id in root_entity_ids:
                root_entity = self.graph.get_entity_by_id(root_entity_id)
                if root_entity:
                    root_entities.append(root_entity)
                else:
                    logger.warning(f"Root entity with ID {root_entity_id} not found")

        if root_entity_types:
            root_entities.extend(self.graph.get_entities_by_types(root_entity_types))

        if not root_entities:
            logger.warning(
                f"No root entities found for indexing. Types: {root_entity_types}"
            )
        else:
            logger.info(f"Indexing {len(root_entities)} root entities")
            hub_indexer.run_indexing(
                root_entities=root_entities,
                force_index_update=force_update or self.settings.force_index_update
            )

    # ==================== Internal Methods ====================

    def _prepare_vector_store(self):
        """
        Prepares the main vector store for the retriever which stores the
        HubPaths for each hub.
        """
        vector_store_name = (f"{self.graph.config.config_hash}_"
                             f"{self.settings.embedding_config.config_hash}"
                             f"{self._indexing_llm.llm_config.config_hash}")

        self.vector_store = ChromaVectorStore(
            store_name=vector_store_name,
            distance_metric=self.settings.distance_metric,
            diversity_penalty=self.settings.diversity_ranking_penalty
        )

    def _prepare_source_handler(self):
        """
        Sets up the source handler responsible for managing source documents used in hub linking.
        """
        self.hub_source_handler = None
        if self.settings.use_source_documents:
            self.hub_source_handler = HubSourceHandler(
                graph=self.graph,
                vector_store_config=self.settings.source_vector_store_config
            )

    def _start_logging(self, question: str, llm_model: str):
        """Initiates logging for the retrieval process."""
        separator = "#" * 30
        strategy = "GraphTraversal" if self.settings.use_topic_if_given else "DirectRetrieval"
        logger.debug(f"\n{separator}\nNew HubLink Retrieval ({strategy})\n{separator}")
        logger.debug(f"Question: {question}")
        logger.debug(f"LLM: {llm_model}")
        logger.debug(f"Number of hubs: {self.settings.number_of_hubs}")

    def _get_llm_adapter(self, llm_config: LLMConfig):
        """Returns a cached LLM adapter for llm_config, creating one if needed.

        Thread-safe: concurrent requests with the same config share the adapter;
        concurrent requests with different configs each get their own adapter
        without overwriting shared instance state.
        """
        cache_key = llm_config.config_hash
        # Fast path – read without lock (adapters are never removed from the cache).
        adapter = self._llm_cache.get(cache_key)
        if adapter is not None:
            return adapter
        with self._llm_cache_lock:
            # Re-check inside the lock to avoid double creation.
            adapter = self._llm_cache.get(cache_key)
            if adapter is None:
                adapter = self._llm_provider.get_llm_adapter(llm_config)
                self._llm_cache[cache_key] = adapter
        return adapter
