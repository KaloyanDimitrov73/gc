from abc import ABC, abstractmethod
from typing import ClassVar, List, Tuple
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_core.documents import Document

from sqa_system.core.config.models.chunking_strategy_config import ChunkingStrategyConfig
from sqa_system.core.config.models.dataset_config import DatasetConfig
from sqa_system.core.config.models.embedding_config import EmbeddingConfig
from sqa_system.core.data.models.context import Context
from sqa_system.core.config.models.knowledge_base.vector_store_config import VectorStoreConfig
from sqa_system.core.config.models.additional_config_parameter import AdditionalConfigParameter


class VectorStoreAdapter(ABC):
    """
    Class that adapts a vector store implementation from LangChain to the
    SQA system. This class is responsible for creating the vector store
    and providing methods to query it. 
    
    Args:
        vector_store (VectorStore): The vector store implementation from LangChain.
    """

    ADDITIONAL_CONFIG_PARAMS: ClassVar[List[AdditionalConfigParameter]] = []

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    @classmethod
    def create_config(cls,
                      vector_store_type: str,
                      chunking_strategy: ChunkingStrategyConfig,
                      embedding_config: EmbeddingConfig,
                      dataset_config: DatasetConfig,
                      **kwargs) -> VectorStoreConfig:
        """Create a VectorStoreConfig object with the specified parameters."""
        cls.validate_config_params(**kwargs)
        return VectorStoreConfig(vector_store_type=vector_store_type,
                                 chunking_strategy_config=chunking_strategy,
                                 embedding_config=embedding_config,
                                 dataset_config=dataset_config,
                                 additional_params=kwargs)

    @classmethod
    def validate_config_params(cls, **kwargs):
        """Validate the configuration parameters."""
        for param in cls.ADDITIONAL_CONFIG_PARAMS:
            is_in_kwargs = False
            # check if parameter is in kwargs
            for key, _ in kwargs.items():
                if key == param.name:
                    is_in_kwargs = True
                    break
                if isinstance(key, AdditionalConfigParameter):
                    if key.name == param.name:
                        is_in_kwargs = True
                        break

            if not is_in_kwargs:
                raise ValueError(f"Parameter {param.name} is required.")

    @abstractmethod
    def query(self, query_text: str, n_results: int) -> List[Context]:
        """
        Performs a query on the vector store and returns the results.

        Args:
            query_text (str): The query text.
            n_results (int): The number of results to return.

        Returns:
            List[Chunk]: The results of the query.
        """

    @abstractmethod
    def query_with_metadata_filter(self,
                                   query_text: str,
                                   n_results: int,
                                   metadata_filter: dict) -> List[Context]:
        """
        Performs a query on the vector store with a filter on its metadata.
        Only results are returned where the metadata matches the filter.

        Args:
            query_text (str): The query text.
            n_results (int): The number of results to return.
            metadata_filter (dict): The metadata to filter by. The key is the
                metadata name and the value is the metadata value.        
        """

    @abstractmethod
    def get_retriever(self) -> VectorStoreRetriever:
        """
        Returns the vector store retriever.

        Returns:
            VectorStoreRetriever: The vector store retriever.
        """

    def convert_documents_to_contexts(self, documents: List[Tuple[Document]]) -> List[Context]:
        """
        Converts a list of tuples, each containing a Document, into a list of Context objects.

        Args:
            documents (List[Tuple[Document]]): A list of tuples, where each tuple
                is expected to contain a single Document object.

        Returns:
            List[Context]: A list of Context objects created from the Document objects
                extracted from the input tuples.
        """
        chunks = []
        for document in documents:
            chunk = Context.from_document(document)
            if chunk is not None:
                chunks.append(chunk)
        return chunks
