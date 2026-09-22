from sqa_system.core.base.base_factory_registry import BaseFactoryRegistry
from .factory.implementations.chroma_vector_store_factory import ChromaVectorStoreFactory
from .factory.base.vector_store_factory import VectorStoreFactory


class VectorStoreFactoryRegistry(BaseFactoryRegistry[VectorStoreFactory]):
    """
    A class that stores all available vector store factories.
    It uses the singleton design pattern to ensure only one instance is created.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreFactoryRegistry, cls).__new__(cls)
            cls._instance.__init__()
        return cls._instance

    def __init__(self):
        # Check if _factories is already initialized
        if not hasattr(self, '_factories'):
            super().__init__()
            self._initialize_available_factories()

    def _initialize_available_factories(self):
        """
        Initializes the available factories. This method needs to be expanded when a new
        vector store is added to the SQA system.
        """
        self.register_factory("chroma", ChromaVectorStoreFactory)
