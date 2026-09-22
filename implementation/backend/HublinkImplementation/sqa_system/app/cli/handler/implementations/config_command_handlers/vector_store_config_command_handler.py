from typing_extensions import override
from InquirerPy.base.control import Choice

from sqa_system.core.config.config_manager import PublicationDatasetConfigManager
from sqa_system.core.config.config_manager import EmbeddingConfigManager
from sqa_system.core.config.config_manager import ChunkingConfigManager
from sqa_system.core.config.config_manager import VectorStoreConfigManager
from sqa_system.knowledge_base.vector_store.storage import VectorStoreFactoryRegistry
from sqa_system.app.cli.handler.base.config_command_handler import ConfigCommandHandler
from sqa_system.core.config.factory.config_factory import ConfigFactory
from sqa_system.core.config.models.knowledge_base.vector_store_config import VectorStoreConfig
from sqa_system.app.cli.cli_command_helper import CLICommandHelper
from .publication_dataset_config_command_handler import PublicationDatasetConfigCommandHandler
from .chunk_config_command_handler import ChunkConfigCommandHandler
from .embeddings_config_command_handler import EmbeddingsConfigCommandHandler


class VectorStoreConfigCommandHandler(ConfigCommandHandler[VectorStoreConfig]):
    """Handles commands related to the vector store management"""

    def __init__(self):
        super().__init__(VectorStoreConfigManager(), "Vector Store")
        self.dataset_config_manager = PublicationDatasetConfigManager()
        self.chunking_config_manager = ChunkingConfigManager()
        self.embedding_config_manager = EmbeddingConfigManager()
        self.config_manager.load_default_configs()
        self.dataset_config_manager.load_default_configs()
        self.chunking_config_manager.load_default_configs()
        self.embedding_config_manager.load_default_configs()

    @override
    def add_config(self):

        vector_store_type = CLICommandHelper.get_selection(
            "Select vector store type:",
            [Choice(factory, factory)
             for factory in VectorStoreFactoryRegistry().get_all_factory_types()]
        )
        if CLICommandHelper.is_exit_selection(vector_store_type):
            return None

        dataset_config = CLICommandHelper.get_selection(
            "Select dataset configuration:",
            [Choice(config, config.name)
             for config in self.dataset_config_manager.get_all_configs()] +
            [Choice(None, "Add new dataset configuration")]
        )
        if dataset_config is None:
            dataset_config = PublicationDatasetConfigCommandHandler().add_config()
            if dataset_config is None:  # if the user exited the add config
                return None
        if CLICommandHelper.is_exit_selection(dataset_config):
            return None

        chunking_strategy_config = CLICommandHelper.get_selection(
            "Select chunking strategy configuration:",
            [Choice(config, config.name)
             for config in self.chunking_config_manager.get_all_configs()] +
            [Choice(None, "Add new chunking strategy configuration")]
        )
        if chunking_strategy_config is None:
            chunking_strategy_config = ChunkConfigCommandHandler().add_config()
            if chunking_strategy_config is None:  # if the user exited the add config
                return None
        if CLICommandHelper.is_exit_selection(chunking_strategy_config):
            return None

        embedding_config = CLICommandHelper.get_selection(
            "Select embedding configuration:",
            [Choice(config, config.name)
             for config in self.embedding_config_manager.get_all_configs()] +
            [Choice(None, "Add new embedding configuration")]
        )
        if embedding_config is None:
            embedding_config = EmbeddingsConfigCommandHandler().add_config()
            if embedding_config is None:  # if the user exited the add config
                return None
        if CLICommandHelper.is_exit_selection(embedding_config):
            return None

        vector_store_factory = VectorStoreFactoryRegistry(
        ).get_factory_class(vector_store_type)
        vector_store_adapter_class = vector_store_factory.get_vector_store_class()

        additional_config_params = self._get_additional_config_params(
            vector_store_adapter_class.ADDITIONAL_CONFIG_PARAMS)
        if len(additional_config_params) > 0:
            config = ConfigFactory.create_vector_store_config(
                vector_store_type=vector_store_type,
                dataset_config=dataset_config,
                chunking_strategy_config=chunking_strategy_config,
                embedding_config=embedding_config,
                **additional_config_params)
        else:
            config = ConfigFactory.create_vector_store_config(
                vector_store_type=vector_store_type,
                dataset_config=dataset_config,
                chunking_strategy_config=chunking_strategy_config,
                embedding_config=embedding_config)

        self.config_manager.add_config(config)
        if CLICommandHelper.get_confirmation("Add this" + self.config_name + " as default?"):
            self.config_manager.save_configs_as_default()

        CLICommandHelper.print(
            f"{self.config_name} added successfully.")

        return config
