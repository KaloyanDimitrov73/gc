from enum import Enum
from typing_extensions import override

from sqa_system.core.config.models import DocumentRetrievalConfig
from sqa_system.core.base.base_factory import BaseFactory

from ..base.document_retriever import DocumentRetriever


class DocumentRetrieverType(Enum):
    """
    This is the main enum that maps a string to the retriever type.
    It is used in the configuration files to specify exactly the 
    retriever to use.
    
    If new retrievers are added, they should be added here as well.
    """
    DOCUMENTEMBED = "documentembed"
    LIGHTRAG = "lightrag"
    MICROSOFTGRAPHRAG = "microsoftgraphrag"


class DocumentRetrieverFactory(BaseFactory):
    """
    A factory class that creates document retrievers based on the specified configuration.
    """

    @override
    def create(self, config: DocumentRetrievalConfig, **kwargs) -> DocumentRetriever:
        """
        Creates a retriever based on the specified configuration.
        
        Args:
            config (DocumentRetrievalConfig): The configuration for the retriever.
            **kwargs: Additional parameters for the retriever.
        Returns:
            DocumentRetriever: The created retriever.
        """
        retriever_class = self.get_retriever_class(config.retriever_type)
        return retriever_class(config)

    @staticmethod
    def import_retriever(retriever_type: str) -> type[DocumentRetriever]:
        """
        Imports the retriever with the specified type.
        This method dynamically imports the retriever class based on the type provided.
        
        We dynamically import the retriever class so that if a retriever has specific 
        requirements but is not used, the user does not need to install the dependencies.
        
        Args:
            retriever_type: The type of retriever to import
        Returns:
            The retriever class
        Raises:
            ImportError: If required dependencies are not installed
            ValueError: If retriever type is not supported
        """
        if retriever_type == DocumentRetrieverType.LIGHTRAG.value:
            try:
                # pylint: disable=import-outside-toplevel
                from sqa_system.retrieval.implementations.LightRag.light_rag\
                    import LightRag
                return LightRag
            except ImportError as e:
                raise ImportError(
                    f"LightRag retriever requires additional dependencies: {e}"
                ) from e
        elif retriever_type == DocumentRetrieverType.DOCUMENTEMBED.value:
            try:
                # pylint: disable=import-outside-toplevel
                from sqa_system.retrieval.implementations.DocumentEmbed.document_embed_retriever\
                    import DocumentEmbedRetriever
                return DocumentEmbedRetriever
            except ImportError as e:
                raise ImportError(
                    f"DocumentEmbed retriever requires additional dependencies: {e}"
                ) from e
        elif retriever_type == DocumentRetrieverType.MICROSOFTGRAPHRAG.value:
            try:
                # pylint: disable=import-outside-toplevel
                from sqa_system.retrieval.implementations.MicrosoftGraphRAG.microsoft_graphrag_retriever\
                    import GraphRAGRetriever
                return GraphRAGRetriever
            except ImportError as e:
                raise ImportError(
                    f"MicrosoftGraphRAG retriever requires additional dependencies: {e}"
                ) from e
        raise ValueError(f"Retriever type {retriever_type} is not supported.")

    @classmethod
    def get_retriever_class(cls, retriever_type: str) -> type[DocumentRetriever]:
        """
        Returns the class of the retriever with the specified type.
        
        Args:
            retriever_type (str): The type of the retriever.
            
        Returns:
            type[DocumentRetriever]: The class of the retr
        """
        return cls.import_retriever(retriever_type)
