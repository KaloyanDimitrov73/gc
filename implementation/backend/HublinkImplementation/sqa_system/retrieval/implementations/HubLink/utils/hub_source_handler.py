from sqa_system.knowledge_base.vector_store.storage import ChromaVectorStoreFactory
from sqa_system.knowledge_base.vector_store.storage import LangchainVectorStoreAdapter
from sqa_system.knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from sqa_system.core.config.models import VectorStoreConfig
from sqa_system.core.logging.logging import get_logger
from sqa_system.core.data.models import Knowledge

from ..models import SourceDocumentSummary, ProcessedQuestion

logger = get_logger(__name__)

class HubSourceHandler:
    """
    Handles linking hubs to their source documents and retrieving relevant source context.

    This class manages the vector store for source documents and provides methods to retrieve relevant document chunks based on a given text.

    Args:
        graph (KnowledgeGraph): The knowledge graph instance.
        vector_store_config (VectorStoreConfig): Configuration for the source document vector store.
    """

    def __init__(self,
                 graph: KnowledgeGraph,
                 vector_store_config: VectorStoreConfig):
        self.graph = graph
        self.source_vector_store = self._index_source_documents(
            vector_store_config)

    def get_source_document_summary(self,
                                    processed_question: ProcessedQuestion,
                                    hub_root_entity: Knowledge,
                                    n_results: int = 10) -> SourceDocumentSummary:
        """
        Retrieves the source document summary for a hub which is the context of
        the source document most relevant to the question.

        Args:
            processed_question (dict): The processed question.
            hub_root_entity (Knowledge): The root entity of the hub.
            n_results (int): The number of results to return.

        Returns:
            SourceDocumentSummary: A summary object for the source document.
        """
        doi = self.get_source_identifier_of_hub_entity(hub_root_entity)
        contexts = self.source_vector_store.query_with_metadata_filter(
            query_text=processed_question.question,
            metadata_filter={"doi": doi},
            n_results=n_results
        )
        if not contexts:
            logger.warning("No source documents found for hub: %s",
                           hub_root_entity.uid)
            return None
        return SourceDocumentSummary(
            source_identifier=doi,
            source_name=hub_root_entity.text,
            contexts=contexts
        )

    def get_source_identifier_of_hub_entity(self, root_entity: Knowledge) -> str:
        """
        Retrieves the identifier that indicates the source document for a 
        hub root entity.

        Args:
            root_entity (Knowledge): The root entity of the hub.

        Returns:
            str: The identifier of the source document.
        """
        root_relations = self.graph.get_relations_of_head_entity(root_entity)
        doi = ""
        for triple in root_relations:
            if "doi" in triple.predicate:
                doi = triple.entity_object.text
                break
        return doi

    def _index_source_documents(self,
                                vector_store_config: VectorStoreConfig) -> LangchainVectorStoreAdapter:
        """
        Runs the indexing process for the source documents by storing them in the vector store.

        Args:
            vector_store_config (VectorStoreConfig): The configuration for the vector store.

        Returns:
            LangchainVectorStoreAdapter: The vector store adapter for the source documents.
        """
        if not vector_store_config.dataset_config:
            logger.warning(
                "The dataset config is invalid. Could not index source documents")
            return None
        try:
            # Create vector store
            return ChromaVectorStoreFactory().create(
                config=vector_store_config)
        except Exception as e:
            logger.error(f"Could not create vector store: {e}")
        return None
