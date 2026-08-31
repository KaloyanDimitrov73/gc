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

from backend.app.contracts.schemas import MessageSchema
from core.data.models import RetrievalAnswer
from language_model import LLMProvider
from language_model.config.llm_config import LLMConfig
from retrieval import KnowledgeGraphRetriever
from knowledge_base.knowledge_graph.storage import KnowledgeGraph
from core.logging.logging import get_logger

from hublink.core.models.hub_link_settings import HubLinkSettings, ADDITIONAL_CONFIG_PARAMS
from hublink.retrieval.utils.hub_source_handler import HubSourceHandler
from hublink.retrieval.strategies.base_retrieval_strategy import RetrievalStrategyData
from hublink.retrieval.strategies.traversal_retrieval_strategy import TraversalRetrievalStrategy
from hublink.retrieval.strategies.direct_retrieval_strategy import DirectRetrievalStrategy
from retrieval.config.kg_retrieval_config import KGRetrievalConfig

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

    def __init__(self, config: KGRetrievalConfig, graph: KnowledgeGraph, hub_storage_manager) -> None:
        super().__init__(config, graph)

        self.settings: HubLinkSettings = HubLinkSettings.from_config(config)
        self.hub_storage_manager = hub_storage_manager
        self._llm_provider = LLMProvider()

        # LLM cache for per-query LLM selection (avoids recreating adapters).
        # Populated immediately with the default config so the first query never
        # needs to create an adapter under the lock.
        # Cache key is the full llm_config hash, not just the model name.
        self._llm_cache: Dict[str, Any] = {}
        self._llm_cache_lock = threading.Lock()
        _default_llm = self._llm_provider.get_llm_adapter(config.query_llm_config)
        self._llm_cache[config.query_llm_config.config_hash] = _default_llm

        # Keep _retrieval_llm for the legacy retrieve_knowledge() override path.
        self._retrieval_llm = _default_llm
        self._retrieval_llm_config = deepcopy(config.query_llm_config)

        # Embedding model (should not change - would require re-indexing)
        self.embedding_model = self._llm_provider.get_embeddings(
            self.settings.embedding_config)

        self._prepare_source_handler()

    # ==================== Retrieval ====================
    def query(self,
            query_text: str,
            number_of_hubs: int,
            retrieval_mode: str,
            llm_config: LLMConfig,
            topic_entity_id: Optional[str] = None,
            use_direct_final_answer: bool = False,
            conversation_history: Optional[List[str]] = None,
            cancel_event: Optional[threading.Event] = None,
            ) -> RetrievalAnswer:
        """
        Main method to query the retriever with optional per-query settings overrides.

        Args:
            query_text (str): The user query.
            topic_entity_id (Optional[str]): Identifier for the topic entity.
            number_of_hubs (int): Number of hubs to retrieve.
            retrieval_mode (str): The retrieval mode ('direct' or 'graph').
            llm_config (LLMConfig): Full LLM configuration for answer generation.
            use_direct_final_answer: Skipping per-hub partial answer generation.
            conversation_history: The last chat messages
            cancel_event: Canceling of retrieval process

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
        logger.info("\n%s\nNew HubLink Retrieval (%s)\n%s", separator, strategy_name, separator)
        logger.info("Question: %s", query_text)
        logger.info("LLM: %s", llm_config.name_model)
        logger.info("Number of hubs: %s", local_settings.number_of_hubs)

        retrieval_data = RetrievalStrategyData(
            graph=self.graph,
            llm_adapter=local_llm,
            embedding_adapter=self.embedding_model,
            settings=local_settings,
            hub_storage_manager=self.hub_storage_manager,
            source_handler=self.hub_source_handler,
        )

        if local_settings.use_topic_if_given and topic_entity_id:
            strategy = TraversalRetrievalStrategy(
                retrieval_data=retrieval_data,
                topic_entity_id=topic_entity_id,
            )
        else:
            strategy = DirectRetrievalStrategy(retrieval_data=retrieval_data)

        return strategy.retrieval(query_text, conversation_history, cancel_event)

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
                    hub_storage_manager=self.hub_storage_manager,
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
                hub_storage_manager=self.hub_storage_manager,
                source_handler=self.hub_source_handler
            )
        )
        return strategy.retrieval(query_text)

    # ==================== Internal Methods ====================

    def _prepare_source_handler(self):
        """
        Sets up the source handler responsible for managing source documents used in hub linking.
        """
        self.hub_source_handler = None

        logger.info("Prepare Source Handler")

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
