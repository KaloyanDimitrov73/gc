from typing_extensions import override

from sqa_system.core.language_model.enums.llm_enums import EndpointType
from sqa_system.core.config.config_manager import EmbeddingConfigManager
from sqa_system.core.config.models.embedding_config import EmbeddingConfig
from sqa_system.core.config.factory.config_factory import ConfigFactory
from sqa_system.app.cli.cli_command_helper import CLICommandHelper, Choice
from sqa_system.app.cli.handler.base.config_command_handler import ConfigCommandHandler
from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)


class EmbeddingsConfigCommandHandler(ConfigCommandHandler[EmbeddingConfig]):
    """Handles commands related to the embeddings configuration."""

    def __init__(self):
        super().__init__(EmbeddingConfigManager(), "Embeddings")
        self.config_manager.load_default_configs()

    @override
    def add_config(self):
        endpoint = CLICommandHelper.get_selection(
            "Select endpoint:",
            [Choice(endpoint.value, endpoint.value)
             for endpoint in EndpointType],
        )
        if CLICommandHelper.is_exit_selection(endpoint):
            return None

        name_model = CLICommandHelper.get_text_input(
            "Enter the name of the model: ")

        config = ConfigFactory.create_embedding_config(
            endpoint=endpoint,
            name_model=name_model
        )
        self.config_manager.add_config(config)
        logger.info("Added embedding configuration %s", config.name)

        if CLICommandHelper.get_confirmation("Add this" + self.config_name + " as default?"):
            self.config_manager.save_configs_as_default()

        return config
