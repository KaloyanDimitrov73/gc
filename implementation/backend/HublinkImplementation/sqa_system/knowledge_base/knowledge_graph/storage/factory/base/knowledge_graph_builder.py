from abc import ABC, abstractmethod
from typing import Type, List, ClassVar
from typing_extensions import override

from sqa_system.core.config.models.additional_config_parameter import AdditionalConfigParameter
from sqa_system.core.config.models import DatasetConfig, LLMConfig
from sqa_system.core.base.base_factory import BaseFactory
from sqa_system.core.data.models import PublicationDataset
from sqa_system.knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from sqa_system.core.config.models import KnowledgeGraphConfig


class KnowledgeGraphBuilder(BaseFactory, ABC):
    """
    A base class for a factory that is intended to create a knowledge graph based
    on a given dataset of publications.  
    """

    ADDITIONAL_CONFIG_PARAMS: ClassVar[List[AdditionalConfigParameter]] = []

    @override
    # Here we disable the warning to allow extending the
    # method with additional parameters
    # pylint: disable=arguments-differ
    def create(self,
               config: KnowledgeGraphConfig,
               publications: PublicationDataset,
               **kwargs) -> KnowledgeGraph:
        """
        The main method for creating a knowledge graph based on the specified
        configuration and dataset of publications.
        
        Args:
            config: The configuration for the knowledge graph
            publications: The dataset of publications to be used for creating or
                populating the knowledge graph
            **kwargs: Additional parameters for the knowledge graph creation
            
        Returns:
            KnowledgeGraph: The created knowledge graph object
        """
        super().create(config, **kwargs)
        return self._create_knowledge_graph(publications, config)

    @classmethod
    @abstractmethod
    def get_knowledge_graph_class(cls) -> Type[KnowledgeGraph]:
        """
        Returns the type of graph the factory creates.
        
        Returns:
            Type[KnowledgeGraph]: The type of graph the factory creates.
        """

    @classmethod
    def create_config(cls,
                      graph_type: str,
                      dataset_config: DatasetConfig,
                      extraction_llm_config: LLMConfig,
                      **kwargs) -> KnowledgeGraphConfig:
        """
        Creates a Knowledge Graph Config object with the specified parameters.

        Args:
            graph_type: The type of knowledge graph to create
            dataset_config: The dataset configuration
            extraction_llm_config: The LLM configuration for the extraction
                of fulltext from the publications in the dataset
            **kwargs: Additional configuration parameters
            
        Returns:
            KnowledgeGraphConfig: The created knowledge graph configuration
        """
        cls.validate_config_params(**kwargs)
        return KnowledgeGraphConfig(
            graph_type=graph_type,
            dataset_config=dataset_config,
            extraction_llm=extraction_llm_config,
            additional_params=kwargs)

    @abstractmethod
    def _create_knowledge_graph(self,
                                publications: PublicationDataset,
                                config: KnowledgeGraphConfig) -> KnowledgeGraph:
        """
        Internal method to create a knowledge graph based on the specified
        configuration and dataset of publications.
        
        Args:
            publications: The dataset of publications to be used for creating or
                populating the knowledge graph
            config: The configuration for the knowledge graph
            
        Returns:
            KnowledgeGraph: The created knowledge graph object
        """

    @classmethod
    def validate_config_params(cls, **kwargs):
        """
        A Knowledge Graph might have additional parameters beyond the default ones
        specified in the KnowledgeGraphConfig. This method checks if all
        required parameters are present in the kwargs.
        
        Args:
            **kwargs: Additional configuration parameters
        Raises:
            ValueError: If any required parameter is missing
        """
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
