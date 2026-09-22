from abc import ABC, abstractmethod
from typing import Type, List, ClassVar
from typing_extensions import override

from sqa_system.core.config.models.additional_config_parameter import AdditionalConfigParameter
from sqa_system.core.base.base_factory import BaseFactory
from sqa_system.knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from sqa_system.core.config.models import KnowledgeGraphConfig


class KnowledgeGraphLoader(BaseFactory, ABC):
    """
    A factory class for loading existing knowledge graphs.
    The intention is not to create data in the graph but just to read from it.  
    """
    
    ADDITIONAL_CONFIG_PARAMS: ClassVar[List[AdditionalConfigParameter]] = []

    @override
    # Here we disable the warning to allow extending the
    # method with additional parameters
    # pylint: disable=arguments-differ
    def create(self,
               config: KnowledgeGraphConfig,
               **kwargs) -> KnowledgeGraph:
        super().create(config, **kwargs)
        return self._load_knowledge_graph(config)

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
                      **kwargs) -> KnowledgeGraphConfig:
        """
        Creates a Knowledge Graph Config object with the specified parameters.

        Args:
            graph_type: The type of knowledge graph to create
            **kwargs: Additional configuration parameters
            
        Returns:
            KnowledgeGraphConfig: The created knowledge graph configuration object
        """
        cls.validate_config_params(**kwargs)
        return KnowledgeGraphConfig(
            graph_type=graph_type,
            additional_params=kwargs)

    @abstractmethod
    def _load_knowledge_graph(self,
                                config: KnowledgeGraphConfig) -> KnowledgeGraph:
        """
        Creates a knowledge graph based on the specified configuration.
        
        Args:
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
