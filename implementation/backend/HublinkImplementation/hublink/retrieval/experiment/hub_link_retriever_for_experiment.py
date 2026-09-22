from typing import Optional

from typing_extensions import override

from core.data.models import RetrievalAnswer
from core.logging.logging import get_logger
from hublink.core.hub_storage_manager import HubStorageManager
from hublink.core.models.hub_link_settings import (
    ADDITIONAL_CONFIG_PARAMS,
    HubLinkSettings,
)
from hublink.core.sparse_index.sparse_storage_manager import SparseStorageManager
from hublink.retrieval.strategies.base_retrieval_strategy import RetrievalStrategyData
from hublink.retrieval.strategies.direct_retrieval_strategy import DirectRetrievalStrategy
from hublink.retrieval.strategies.traversal_retrieval_strategy import TraversalRetrievalStrategy
from hublink.retrieval.utils.hub_source_handler import HubSourceHandler
from knowledge_base.knowledge_graph.storage import KnowledgeGraph
from language_model import LLMProvider
from retrieval import KnowledgeGraphRetriever
from retrieval.config.kg_retrieval_config import KGRetrievalConfig

logger = get_logger(__name__)


class HubLinkRetrieverForExperiment(KnowledgeGraphRetriever):
    """Experiment HubLink retriever with no indexing responsibilities."""

    ADDITIONAL_CONFIG_PARAMS = ADDITIONAL_CONFIG_PARAMS

    def __init__(
            self,
            config: KGRetrievalConfig,
            graph: KnowledgeGraph,
            hub_storage_manager: HubStorageManager,
            sparse_storage_manager: SparseStorageManager) -> None:
        super().__init__(config, graph)
        self.settings = HubLinkSettings.from_config(config)
        self.hub_storage_manager = hub_storage_manager
        self.sparse_storage_manager = sparse_storage_manager

        llm_provider = LLMProvider()
        self.query_llm = llm_provider.get_llm_adapter(config.query_llm_config)
        self.embedding_model = llm_provider.get_embeddings(
            self.settings.embedding_config
        )
        self._prepare_source_handler()

    @override
    def retrieve_knowledge(
            self,
            query_text: str,
            topic_entity_id: Optional[str],
            topic_entity_value: Optional[str]) -> RetrievalAnswer:
        self._start_logging(query_text)

        retrieval_data = RetrievalStrategyData(
            graph=self.graph,
            llm_adapter=self.query_llm,
            embedding_adapter=self.embedding_model,
            settings=self.settings,
            hub_storage_manager=self.hub_storage_manager,
            source_handler=self.hub_source_handler,
        )

        if self.settings.use_topic_if_given and topic_entity_id:
            strategy = TraversalRetrievalStrategy(
                retrieval_data=retrieval_data,
                topic_entity_id=topic_entity_id,
            )
            return strategy.retrieval(query_text)

        direct_strategy = DirectRetrievalStrategy
        if self.settings.use_cross_encoder:
            from hublink.retrieval.strategies.cross_encoder_direct_retrieval_strategy import (
                CrossEncoderDirectRetrievalStrategy,
            )
            direct_strategy = CrossEncoderDirectRetrievalStrategy

        strategy = direct_strategy(
            retrieval_data=retrieval_data,
            sparse_storage_manager=self.sparse_storage_manager,
        )
        return strategy.retrieval(query_text)

    def _prepare_source_handler(self) -> None:
        self.hub_source_handler = None
        if self.settings.use_source_documents:
            self.hub_source_handler = HubSourceHandler(
                graph=self.graph,
                vector_store_config=self.settings.source_vector_store_config,
            )

    def _start_logging(self, question: str) -> None:
        separator = "#" * 30
        strategy = (
            "GraphTraversal"
            if self.settings.use_topic_if_given
            else "DirectRetrieval"
        )
        logger.debug(
            "\n%s\nNew HubLink Retrieval with %s Strategy\n%s",
            separator,
            strategy,
            separator,
        )
        logger.debug("Question: %s", question)
        logger.debug("Parameters: %s", self.settings)
