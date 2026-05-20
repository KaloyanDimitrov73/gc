from typing import Dict, Type, List
from sqa_system.core.config.models.base.config import Config
from ..base.configuration_manager import ConfigurationManager

from ..implementations.chunking_config_manager import ChunkingConfigManager
from ..implementations.document_retrieval_config_manager import DocumentRetrievalConfigManager
from ..implementations.embedding_config_manager import EmbeddingConfigManager
from ..implementations.pipeline_config_manager import PipelineConfigManager
from ..implementations.qa_dataset_config_manager import QADatasetConfigManager
from ..implementations.kg_retrieval_config_manager import KGRetrievalConfigManager
from ..implementations.llm_config_manager import LLMConfigManager
from ..implementations.generation_config_manager import GenerationConfigManager
from ..implementations.kg_store_config_manager import KGStoreConfigManager
from ..implementations.publication_dataset_config_manager import PublicationDatasetConfigManager
from ..implementations.vector_store_config_manager import VectorStoreConfigManager


class ConfigManagerFactory:
    """
    Class that is used to get the config manager class for a given config.
    """

    _config_managers: List[Type[ConfigurationManager]] = [
        PipelineConfigManager,
        QADatasetConfigManager,
        ChunkingConfigManager,
        KGRetrievalConfigManager,
        LLMConfigManager,
        EmbeddingConfigManager,
        GenerationConfigManager,
        KGStoreConfigManager,
        PublicationDatasetConfigManager,
        VectorStoreConfigManager,
        DocumentRetrievalConfigManager,
    ]

    _config_manager_map: Dict[Type[Config], Type[ConfigurationManager]] = {
        manager.get_config_class(): manager for manager in _config_managers
    }

    _config_name_to_class_map: Dict[str, Type[Config]] = {
        manager.get_config_class().__name__: manager.get_config_class()
        for manager in _config_managers
    }

    @staticmethod
    def get_config_manager(config: Config) -> ConfigurationManager:
        """
        Returns the config manager for the given config.

        Args:
            config (Config): The config to get the config manager for.

        Returns:
            ConfigurationManager: The config manager for the given config.
        """
        manager_class = ConfigManagerFactory._config_manager_map.get(
            type(config))
        if manager_class:
            return manager_class()

        raise ValueError(f"Config {config} not supported.")

    @staticmethod
    def get_config_manager_by_type(config_type: Type[Config]) -> ConfigurationManager:
        """
        Returns the config manager for the given config type.

        Args:
            config_type (Type[Config]): The config type to get the config manager for.

        Returns:
            ConfigurationManager: The config manager for the given config type.
        """
        manager_class = ConfigManagerFactory._config_manager_map.get(config_type)
        if manager_class:
            return manager_class()

        raise ValueError(f"Config type {config_type} not supported.")

    @staticmethod
    def get_all_configs() -> List[Type[Config]]:
        """
        Returns a list of all the supported configurations.

        Returns:
            List[Type[Config]]: A list of all the supported configurations.
        """
        return [manager.get_config_class() for manager in ConfigManagerFactory._config_managers]

    @staticmethod
    def get_config_class_by_name(name: str) -> Type[Config]:
        """
        Retrieves a Config class by its name.

        Args:
            name (str): The name of the Config class.

        Returns:
            Type[Config]: The corresponding Config class.

        Raises:
            ValueError: If the Config class name is not found.
        """
        config_class = ConfigManagerFactory._config_name_to_class_map.get(name)
        if not config_class:
            raise ValueError(f"No Config class found with name: {name}")
        return config_class
