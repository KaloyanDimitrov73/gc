from typing_extensions import override

from sqa_system.core.config.models.dataset_config import DatasetConfig
from sqa_system.core.data.data_loader.factory.data_loader_factory import DataLoaderFactory
from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.data.models import PublicationDataset
from sqa_system.core.config.models import KnowledgeGraphConfig
from sqa_system.core.base.base_manager import BaseManager
from sqa_system.core.logging.logging import get_logger

from .knowledge_graph_factory_registry import KnowledgeGraphFactoryRegistry
from .factory.base.knowledge_graph_builder import KnowledgeGraphBuilder
from .base.knowledge_graph import KnowledgeGraph
logger = get_logger(__name__)


class KnowledgeGraphManager(BaseManager[KnowledgeGraph, KnowledgeGraphConfig]):
    """
    Manager class for handling knowledge graphs.
    It uses the singleton design pattern to ensure only one instance is created.
    If the graph is local, the manager allows to create and cache the graph.
    """
    _factory_registry = KnowledgeGraphFactoryRegistry()

    @override
    def _create_item(self, config: KnowledgeGraphConfig) -> KnowledgeGraph:
        factory_class = self._factory_registry.get_factory_class(
            config.graph_type)

        if issubclass(factory_class, KnowledgeGraphBuilder):
            dataset_config = config.dataset_config
            if dataset_config is None:
                raise ValueError(
                    "Dataset config must be provided when creating a new knowledge graph.")

            publication_dataset = self._load_dataset(dataset_config)
            factory = factory_class()

            return factory.create(config=config, publications=publication_dataset)

        factory = factory_class()
        return factory.create(config=config)

    def _load_dataset(self, config: DatasetConfig) -> PublicationDataset:
        data_loader = DataLoaderFactory.get_data_loader(config.loader)
        file_path = FilePathManager().get_path(file_name=config.file_name)
        publication_dataset = data_loader.load(dataset_name=config.name,
                                               path=file_path,
                                               limit=config.loader_limit)
        if not isinstance(publication_dataset, PublicationDataset):
            raise ValueError(
                "There was an error loading the publication dataset.")
        return publication_dataset
