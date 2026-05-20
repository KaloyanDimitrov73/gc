from typing_extensions import override
from sqa_system.core.config.models.knowledge_base.vector_store_config import VectorStoreConfig
from sqa_system.core.base.base_manager import BaseManager
from sqa_system.core.logging.logging import get_logger
from .base.vector_store_adapter import VectorStoreAdapter
from .vector_store_factory_registry import VectorStoreFactoryRegistry
logger = get_logger(__name__)


class VectorStoreManager(BaseManager[VectorStoreAdapter, VectorStoreConfig]):
    """
    Manager class for vector store creation and retrieval.
    It allows to create and cache vector stores.

    It uses the singleton design pattern to ensure only one instance is created.
    """
    _factory_registry = VectorStoreFactoryRegistry()

    @override
    def _create_item(self, config: VectorStoreConfig) -> VectorStoreAdapter:
        factory_class = self._factory_registry.get_factory_class(
            config.vector_store_type)

        factory = factory_class()

        return factory.create(config)
