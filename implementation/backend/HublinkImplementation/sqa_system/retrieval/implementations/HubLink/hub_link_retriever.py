from typing import Optional, List
from typing_extensions import override

from sqa_system.core.config.models import KGRetrievalConfig
from sqa_system.core.language_model.llm_provider import LLMProvider
from sqa_system.retrieval import KnowledgeGraphRetriever
from sqa_system.core.data.models import RetrievalAnswer
from sqa_system.knowledge_base.knowledge_graph.storage import KnowledgeGraph
from sqa_system.core.logging.logging import get_logger

from .models.hub_link_settings import HubLinkSettings, ADDITIONAL_CONFIG_PARAMS
from .models.hub import IsHubOptions
from .utils.hub_indexer import HubIndexer, HubIndexerOptions
from .utils.vector_store import ChromaVectorStore
from .utils.hub_source_handler import HubSourceHandler
from .retrieval.base_retrieval_strategy import RetrievalStrategyData
from .retrieval.traversal_retrieval_strategy import TraversalRetrievalStrategy
from .retrieval.direct_retrieval_strategy import DirectRetrievalStrategy

logger = get_logger(__name__)


class HubLinkRetriever(KnowledgeGraphRetriever):
    """
    Our new retrieval approach.
    
    This retriever identifies hub entities in the graph, links them to source documents, and generates partial answers for each hub. These partial answers are then consolidated into a final answer using a language model. The approach supports both graph traversal (when a topic entity is provided) and direct retrieval strategies.

    Args:
        config (KGRetrievalConfig): Retrieval configuration.
        graph (KnowledgeGraph): The knowledge graph instance to query.
    """
    ADDITIONAL_CONFIG_PARAMS = ADDITIONAL_CONFIG_PARAMS

    def __init__(self, config: KGRetrievalConfig, graph: KnowledgeGraph) -> None:
        super().__init__(config, graph)
        self.settings: HubLinkSettings = HubLinkSettings.from_config(config)
        self.llm = LLMProvider().get_llm_adapter(config.llm_config)
        self.embedding_model = LLMProvider().get_embeddings(
            self.settings.embedding_config)

        self._prepare_vector_store()
        self.build_index(
            root_entity_types=self.settings.indexing_root_entity_types,
            root_entity_ids=self.settings.indexing_root_entity_ids
        )
        self._prepare_source_handler()

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
        self._start_logging(query_text)

        # There are two types of strategies that are supported
        # depending on whether a Topic Entity is given, or not.
        if self.settings.use_topic_if_given and topic_entity_id:
            strategy = TraversalRetrievalStrategy(
                retrieval_data=RetrievalStrategyData(
                    graph=self.graph,
                    llm_adapter=self.llm,
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
                llm_adapter=self.llm,
                embedding_adapter=self.embedding_model,
                settings=self.settings,
                vector_store=self.vector_store,
                source_handler=self.hub_source_handler
            )
        )
        return strategy.retrieval(query_text)

    def build_index(self,
                    root_entity_types: Optional[List[str]] = None,
                    root_entity_ids: Optional[List[str]] = None):
        """
        Builds the index for the retriever by starting from specified root entity types or IDs.
        This index supports hub-based querying within the knowledge graph.
        
        Args:
            root_entity_types (Optional[List[str]]): Entity types to initiate indexing.
            root_entity_ids (Optional[List[str]]): Specific entity IDs to initiate indexing.
        """
        if not root_entity_types and not root_entity_ids:
            raise ValueError(
                "Either root_entity_types or root_entity_ids must be specified for indexing."
            )
        if not self.vector_store:
            raise ValueError(
                "Vector store is not initialized. Cannot build index."
            )

        logger.debug("Building/Checking index for the HubLink retriever")
        hub_indexer = HubIndexer(
            graph=self.graph,
            options=HubIndexerOptions(
                embedding_model=self.embedding_model,
                is_hub_options=IsHubOptions(
                    hub_edges=self.settings.hub_edges,
                    types=self.settings.hub_types
                ),
                llm=self.llm,
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
                    logger.warning(
                        "Root entity with ID %s not found in the graph", root_entity_id)
        if root_entity_types:
            root_entities.extend(
                self.graph.get_entities_by_types(root_entity_types))

        if not root_entities:
            logger.warning("No root entities found for indexing. Did you specify the correct types?: %s",
                           root_entity_types)
        else:
            logger.debug("Starting indexing for root entities: %s",
                         root_entities)
            hub_indexer.run_indexing(
                root_entities=root_entities,
                force_index_update=self.settings.force_index_update)

    def _prepare_vector_store(self):
        """
        Prepares the main vector store for the retriever which stores the
        HubPaths for each hub.
        """
        vector_store_name = (f"{self.graph.config.config_hash}_"
                             f"{self.settings.embedding_config.config_hash}"
                             f"{self.llm.llm_config.config_hash}")
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

    def _start_logging(self, question: str):
        """
        Initiates logging for the retrieval process, indicating the chosen retrieval strategy.
        
        Args:
            question (str): The question that is asked.
        """
        separator = "#" * 30
        if self.settings.use_topic_if_given:
            logger.debug(
                f"SEPARATOR\n{separator}\n New HubLink Retrieval with GraphTraversal Strategy\n{separator}")
        else:
            logger.debug(
                f"SEPARATOR\n{separator}\n New HubLink Retrieval with DirectRetrieval Strategy\n{separator}")
        logger.debug("Question: %s", question)
        logger.debug("Parameters: %s", self.settings)
