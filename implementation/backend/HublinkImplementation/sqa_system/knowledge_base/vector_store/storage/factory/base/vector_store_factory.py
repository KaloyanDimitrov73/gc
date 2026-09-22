from abc import ABC, abstractmethod
from typing import List, Type
from typing_extensions import override
from langchain_core.documents import Document

from sqa_system.core.config.models.dataset_config import DatasetConfig
from sqa_system.core.data.data_loader.factory.data_loader_factory import DataLoaderFactory
from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.data.models.context import Context
from sqa_system.core.data.models.publication import Publication
from sqa_system.core.data.models import PublicationDataset
from sqa_system.core.config.models.knowledge_base.vector_store_config import VectorStoreConfig
from sqa_system.core.base.base_factory import BaseFactory
from sqa_system.knowledge_base.vector_store.chunking.chunker import Chunker
from sqa_system.core.logging.logging import get_logger
from ...base.vector_store_adapter import VectorStoreAdapter
logger = get_logger(__name__)


class VectorStoreFactory(BaseFactory, ABC):
    """
    A factory that creates vector store adapters based on the specified configuration.
    """

    @classmethod
    @abstractmethod
    def get_vector_store_class(cls) -> Type[VectorStoreAdapter]:
        """Returns the class of the vector store adapter."""

    @classmethod
    def create_config(cls, **kwargs) -> VectorStoreConfig:
        """Creates a VectorStoreConfig object with the specified parameters."""
        return cls.get_vector_store_class().create_config(**kwargs)

    @override
    def create(self,
               config: VectorStoreConfig,
               **kwargs) -> VectorStoreAdapter:
        """Creates a vector store based on the specified configuration."""
        dataset_config = config.dataset_config
        if dataset_config is None:
            raise ValueError(
                "Dataset config must be provided when creating a new vector store.")

        publication_dataset = self._load_dataset(dataset_config)
        chunker = Chunker(config.chunking_strategy_config)
        return self._create_vector_store(publication_dataset, chunker, config)

    @abstractmethod
    def _create_vector_store(self,
                             publications: PublicationDataset,
                             chunker: Chunker,
                             config: VectorStoreConfig) -> VectorStoreAdapter:
        """Creates a vector store based on the specified configuration."""

    def convert_publications_to_documents(self, publications: List[Publication]) -> List[Document]:
        """
        Converts a list of Publication objects into a list of Document objects.

        Args:
            publications (List[Publication]): A list of Publication objects to be converted.

        Returns:
            List[Document]: A list of Document objects created from the input publications.

        """
        documents = []
        for publication in publications:
            document = publication.to_document()
            if document is not None:
                documents.append(document)
        return documents

    def convert_chunks_to_documents(self, chunks: List[Context]) -> List[Document]:
        """
        Converts a list of Chunk objects into a list of Document objects.

        Args:
            chunks (List[Chunk]): A list of Chunk objects to be converted.

        Returns:
            List[Document]: A list of Document objects created from the input chunks.

        """
        documents = []
        for chunk in chunks:
            document = chunk.to_document()
            if document is not None:
                documents.append(document)
        return documents

    def _load_dataset(self, config: DatasetConfig) -> PublicationDataset:
        """
        Loads a dataset based on the specified configuration using the appropriate data loader.
        This method retrieves the file path for the dataset and uses the data loader to load
        the dataset.
        
        Args:
            config (DatasetConfig): The configuration for the dataset.
        Returns:
            PublicationDataset: The loaded dataset.
        Raises:
            ValueError: If the dataset cannot be loaded.
        """
        data_loader = DataLoaderFactory.get_data_loader(config.loader)
        file_path = FilePathManager().get_path(file_name=config.file_name)
        publication_dataset = data_loader.load(dataset_name=config.name,
                                               path=file_path,
                                               limit=config.loader_limit)
        if not isinstance(publication_dataset, PublicationDataset):
            raise ValueError(
                "There was an error loading the publication dataset.")
        return publication_dataset
