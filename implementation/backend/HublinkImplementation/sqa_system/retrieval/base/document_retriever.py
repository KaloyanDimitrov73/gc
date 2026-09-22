from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional
import weave

from sqa_system.core.data.models import RetrievalAnswer
from sqa_system.core.config.models import (
    DocumentRetrievalConfig,
    AdditionalConfigParameter
)

from .retriever import Retriever


class DocumentRetriever(Retriever, ABC):
    """
    Retriever class that retrieves related document chunks from Publications.
    """

    ADDITIONAL_CONFIG_PARAMS: ClassVar[List[AdditionalConfigParameter]] = []

    def __init__(self, config: DocumentRetrievalConfig) -> None:
        super().__init__(config)

    @classmethod
    def create_config(cls,
                      retriever_type: str,
                      name: Optional[str] = None,
                      **kwargs) -> DocumentRetrievalConfig:
        """
        Creates a DocumentRetrievalConfig object with the specified parameters.

        Args:
            retriever_type (str): The type of the retriever.
            name (Optional[str]): The name of the retriever.
            **kwargs: Additional parameters for the configuration.

        Returns:
            DocumentRetrievalConfig: The created configuration object.
        """
        try:
            dataset_config = kwargs.pop('dataset_config')
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e.args[0]}") from e

        cls.validate_config_params(**kwargs)

        if name:
            name = DocumentRetrievalConfig.prepare_name_for_config(name)
            return DocumentRetrievalConfig(
                retriever_type=retriever_type,
                dataset_config=dataset_config,
                name=name,
                additional_params=kwargs
            )

        # Return the config without a name
        return DocumentRetrievalConfig(
            retriever_type=retriever_type,
            dataset_config=dataset_config,
            additional_params=kwargs
        )

    @abstractmethod
    @weave.op()
    def retrieve(self,
                 query_text: str) -> RetrievalAnswer:
        """
        Retrieves related document chunks from the knowledge base

        Args:
            query_text (str): The question that is used to retrieve relevant 
            document chunks.


        Returns:
            List[Context]: A list of related document chunks
            Optional[str]: The answer of the retriever if they provide it.
        """
