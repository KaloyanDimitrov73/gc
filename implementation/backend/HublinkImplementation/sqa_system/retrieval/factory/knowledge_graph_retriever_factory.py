from enum import Enum
from typing_extensions import override

from sqa_system.knowledge_base.knowledge_graph.storage import KnowledgeGraph
from sqa_system.knowledge_base.knowledge_graph.storage import KnowledgeGraphManager
from sqa_system.core.config.models import KGRetrievalConfig
from sqa_system.core.base.base_factory import BaseFactory

from ..base.knowledge_graph_retriever import KnowledgeGraphRetriever


class KnowledgeGraphRetrieverType(Enum):
    """
    A enum class that represents the different types of knowledge graph retrievers
    that are available in the system.
    """
    TOG = "tog"
    STRUCTGPT = "structgpt"
    HUBLINK = "hublink"
    DIFAR = "difar"
    FIDELIS = "fidelis"
    MINDMAP = "mindmap"


class KnowledgeGraphRetrieverFactory(BaseFactory):
    """
    A factory class that creates retrievers based on the specified configuration.
    """

    @override
    def create(self, config: KGRetrievalConfig, **kwargs) -> KnowledgeGraphRetriever:
        """
        Creates a retriever based on the specified configuration.

        Args:
            config (KGRetrievalConfig): The configuration for the retriever.
            **kwargs: Additional parameters for the retriever.
        Returns:
            KnowledgeGraphRetriever: The created retriever.
        """
        retriever_class = self.get_retriever_class(config.retriever_type)

        graph = self._load_graph(config.knowledge_graph_config)
        initialized_retriever = retriever_class(config, graph)

        return initialized_retriever

    @staticmethod
    def import_retriever(retriever_type: str) -> type[KnowledgeGraphRetriever]:
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
        if retriever_type == KnowledgeGraphRetrieverType.TOG.value:
            try:
                # pylint: disable=import-outside-toplevel
                from sqa_system.retrieval.implementations.ToG.tog_retriever\
                    import ToGKnowledgeGraphRetriever
                return ToGKnowledgeGraphRetriever
            except ImportError as e:
                raise ImportError(
                    f"ToG retriever requires additional dependencies: {e}"
                ) from e
        elif retriever_type == KnowledgeGraphRetrieverType.STRUCTGPT.value:
            try:
                # pylint: disable=import-outside-toplevel
                from sqa_system.retrieval.implementations.StructGPT.\
                    struct_gpt_main import StructGPTKnowledgeGraphRetriever
                return StructGPTKnowledgeGraphRetriever
            except ImportError as e:
                raise ImportError(
                    f"StructGPT retriever requires additional dependencies: {e}"
                ) from e
        elif retriever_type == KnowledgeGraphRetrieverType.HUBLINK.value:
            try:
                # pylint: disable=import-outside-toplevel
                from sqa_system.retrieval.implementations.HubLink.hub_link_retriever\
                    import HubLinkRetriever
                return HubLinkRetriever
            except ImportError as e:
                raise ImportError(
                    f"HubLink retriever requires additional dependencies: {e}"
                ) from e
        elif retriever_type == KnowledgeGraphRetrieverType.DIFAR.value:
            try:
                # pylint: disable=import-outside-toplevel
                from sqa_system.retrieval.implementations.DiFaR.dirfar_retriever\
                    import DifarRetriever
                return DifarRetriever
            except ImportError as e:
                raise ImportError(
                    f"TripleEmbed retriever requires additional dependencies: {e}"
                ) from e
        elif retriever_type == KnowledgeGraphRetrieverType.FIDELIS.value:
            try:
                # pylint: disable=import-outside-toplevel
                from sqa_system.retrieval.implementations.FiDELIS.fidelis_retriever\
                    import FidelisRetriever
                return FidelisRetriever
            except ImportError as e:
                raise ImportError(
                    f"Fidelis retriever requires additional dependencies: {e}"
                ) from e
        elif retriever_type == KnowledgeGraphRetrieverType.MINDMAP.value:
            try:
                # pylint: disable=import-outside-toplevel
                from sqa_system.retrieval.implementations.MindMap.mind_map_retriever\
                    import MindMapRetriever
                return MindMapRetriever
            except ImportError as e:
                raise ImportError(
                    f"MindMap retriever requires additional dependencies: {e}"
                ) from e

        raise ValueError(f"Retriever type {retriever_type} is not supported.")

    @classmethod
    def get_retriever_class(cls, retriever_type: str) -> type[KnowledgeGraphRetriever]:
        """
        Returns the class of the retriever with the specified type.

        Args:
            retriever_type (str): The type of the retriever.

        Returns:
            type[KnowledgeGraphRetriever]: The class of the retriever.
        """
        return cls.import_retriever(retriever_type)

    def _load_graph(self, kg_config: KGRetrievalConfig) -> KnowledgeGraph:
        """
        Loads the knowledge graph based on the provided configuration.
        Args:
            kg_config (KGRetrievalConfig): The configuration for the knowledge graph.
        Returns:
            KnowledgeGraph: The loaded knowledge graph.
        """
        knowledge_graph_manager = KnowledgeGraphManager()
        knowlege_graph = knowledge_graph_manager.get_item(kg_config)
        return knowlege_graph
