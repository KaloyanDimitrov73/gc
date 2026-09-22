from typing import Union
from sqa_system.core.base.base_factory_registry import BaseFactoryRegistry
from .factory.base.knowledge_graph_builder import KnowledgeGraphBuilder
from .factory.base.knowledge_graph_loader import KnowledgeGraphLoader
from .factory.implementations.orkg_knowledge_graph.orkg_knowledge_graph_factory import ORKGKnowledgeGraphFactory
from .factory.implementations.local_knowledge_graph.local_knowledge_graph_factory import LocalKnowledgeGraphFactory
from .factory.implementations.rdf_file_rdf_graph.rdf_file_graph_factory import RDFFileGraphFactory
class KnowledgeGraphFactoryRegistry(BaseFactoryRegistry[Union[KnowledgeGraphBuilder, KnowledgeGraphLoader]]):
    """
    A class that stores all available knowledge graph factories.
    It uses the singleton design pattern to ensure only one instance is created.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(
                KnowledgeGraphFactoryRegistry, cls).__new__(cls)
            cls._instance.__init__()
        return cls._instance

    def __init__(self):
        # Check if _factories is already initialized
        if not hasattr(self, '_factories'):
            super().__init__()
            #
            self._register_factories()

    def _register_factories(self):
        """
        Here we initialize the factories that are available 
        in the QA system. When adding a new graph, it is crucial
        to register it in the registry either at runtime or here.
        After the factory is registered, nothing more needs to be
        done. The Knowledge Graph Manager will now recognize its
        existence.
        """
        self.register_factory("orkg", ORKGKnowledgeGraphFactory)
        self.register_factory("local_rdflib", LocalKnowledgeGraphFactory)
        self.register_factory("rdf_file", RDFFileGraphFactory)
