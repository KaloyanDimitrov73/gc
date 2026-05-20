from typing_extensions import override

from sqa_system.core.config.config_manager import ChunkingConfigManager
from sqa_system.core.config.models.chunking_strategy_config import ChunkingStrategyConfig
from sqa_system.core.config.factory.config_factory import ConfigFactory
from sqa_system.app.cli.cli_command_helper import CLICommandHelper, Choice
from sqa_system.app.cli.handler.base.config_command_handler import ConfigCommandHandler
from sqa_system.knowledge_base.vector_store.chunking.chunking_strategy_factory import ChunkingStrategyType
from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)


class ChunkConfigCommandHandler(ConfigCommandHandler[ChunkingStrategyConfig]):
    """Handles commandes related to the chunking configuration."""

    def __init__(self):
        super().__init__(ChunkingConfigManager(), "Chunking")
        self.config_manager.load_default_configs()

    @override
    def add_config(self):

        chunking_strategy_type = CLICommandHelper.get_selection(
            "Select chunking strategy type:",
            [Choice(strategy.value, strategy.name)
             for strategy in ChunkingStrategyType],
        )
        if CLICommandHelper.is_exit_selection(chunking_strategy_type):
            return None
        chunk_size = int(CLICommandHelper.get_text_input(
            "Enter the chunk size: ", default="300"))
        chunk_overlap = int(CLICommandHelper.get_text_input(
            "Enter the chunk overlap: ", default="15"))

        config = ConfigFactory.create_chunking_strategy_config(
            chunking_strategy_type=chunking_strategy_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap)
        self.config_manager.add_config(config)
        logger.info(
            "Chunking strategy configuration %s added successfully.", config.name)

        if CLICommandHelper.get_confirmation("Add this" + self.config_name + " as default?"):
            self.config_manager.save_configs_as_default()

        return config
