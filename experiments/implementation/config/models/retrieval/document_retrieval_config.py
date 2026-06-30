from typing import Optional, Literal

from pydantic import Field
from .retrieval_config import RetrievalConfig
from ..dataset_config import DatasetConfig
from ..knowledge_base.vector_store_config import VectorStoreConfig


class DocumentRetrievalConfig(RetrievalConfig):
    """
    Configuration for retrievers that are of document retrieval type.
    """
    type: Literal["document_retrieval"] = "document_retrieval"
    retriever_type: str
    dataset_config: Optional[DatasetConfig] = Field(
        None,
        description="A configuration for the dataset to be used."
    )
    vector_store_config: Optional[VectorStoreConfig] = Field(
        None,
        description="A configuration for the vector store to be used."
    )


    def generate_name(self):

        if isinstance(self.vector_store_config, VectorStoreConfig):
            return (
                f"{self.retriever_type}_"
                f"{self.vector_store_config.generate_name()}"
            )
        if isinstance(self.dataset_config, DatasetConfig):
            return (
                f"{self.retriever_type}_"
                f"{self.dataset_config.generate_name()}"
            )
        return f"{self.retriever_type}"
