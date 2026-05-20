from abc import ABC, abstractmethod
from typing import ClassVar, List, Optional
import weave

from sqa_system.core.data.models import RetrievalAnswer
from sqa_system.knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from sqa_system.core.config.models.additional_config_parameter import AdditionalConfigParameter
from sqa_system.core.config.models import KGRetrievalConfig

from .retriever import Retriever


class KnowledgeGraphRetriever(Retriever, ABC):
    """
    Retriever class that retrieves related knowledge entities from a knowledge graph.
    """

    ADDITIONAL_CONFIG_PARAMS: ClassVar[List[AdditionalConfigParameter]] = []

    def __init__(self, config: KGRetrievalConfig, graph: KnowledgeGraph) -> None:
        super().__init__(config)
        self.graph = graph

    @classmethod
    def create_config(cls,
                      retriever_type: str,
                      name: Optional[str] = None,
                      **kwargs) -> KGRetrievalConfig:
        """
        Creates a KGRetrievalConfig object with the specified parameters.
        
        Args:
            retriever_type (str): The type of the retriever.
            name (Optional[str]): The name of the retriever.
            **kwargs: Additional parameters for the configuration.
            
        Returns:
            KGRetrievalConfig: The created configuration object.
        """
        try:
            knowledge_graph_config = kwargs.pop('knowledge_graph_config')
            llm_config = kwargs.pop('llm_config')
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e.args[0]}") from e

        cls.validate_config_params(**kwargs)

        if name:
            name = KGRetrievalConfig.prepare_name_for_config(name)
            return KGRetrievalConfig(
                retriever_type=retriever_type,
                knowledge_graph_config=knowledge_graph_config,
                llm_config=llm_config,
                name=name,
                additional_params=kwargs
            )

        # Return the config without a name
        return KGRetrievalConfig(
            retriever_type=retriever_type,
            knowledge_graph_config=knowledge_graph_config,
            llm_config=llm_config,
            additional_params=kwargs
        )

    @abstractmethod
    @weave.op()
    def retrieve_knowledge(self,
                           query_text: str,
                           topic_entity_id: Optional[str],
                           topic_entity_value: Optional[str]) -> RetrievalAnswer:
        """
        Retrieves related context from the knowledge graph and returns a RetrievalAnswer object.
        This is the main method to conduct the retrieval process and should be implemented by subclasses.
        

        Args:
            query (str): The question that is used to retrieve relevant entities.
            topic_entity_id (str): The entry entity id in the graph from which the 
                search is started from.
            topic_entity_value (str): The entry entity value in the graph from which the 
                search is started from.

        Returns:
            List[Context]: A list of related knowledge entities
            Optional[str]: The answer of the retriever if they provide it.
        """
