# standard imports
from typing_extensions import override

# local imports
from sqa_system.core.config.models.pipe.generation_config import GenerationConfig
from sqa_system.core.config.factory.config_factory import ConfigFactory
from sqa_system.app.cli.cli_command_helper import CLICommandHelper
from sqa_system.core.config.config_manager import (
    LLMConfigManager,
    GenerationConfigManager
)
from sqa_system.app.cli.handler.base.config_command_handler import (
    ConfigCommandHandler,
    Choice
)
from .llm_config_command_handler import LLMConfigCommandHandler


class GenerationConfigCommandHandler(ConfigCommandHandler[GenerationConfig]):
    """Class for handling commands related to the generation pipe management"""

    def __init__(self):
        super().__init__(GenerationConfigManager(), "Generation Pipe")
        self.llm_config_manager = LLMConfigManager()
        self.llm_config_manager.load_default_configs()
        self.config_manager.load_default_configs()
        

    @override
    def add_config(self):
        """Adds a new generation configuration to the manager."""
        llm_config = CLICommandHelper.get_selection(
            "Select LLM config:",
            [Choice(config, config.name)
             for config in self.llm_config_manager.get_all_configs()] +
            [Choice(None, "Add new LLM config")]
        )
        if llm_config is None:
            llm_config = LLMConfigCommandHandler().add_config()
            if llm_config is None:  # if the user exited the add config
                return None
        if CLICommandHelper.is_exit_selection(llm_config):
            return None
        config = ConfigFactory.create_generation_config(llm_config=llm_config)
        self.config_manager.add_config(config)
        if CLICommandHelper.get_confirmation("Add this" + self.config_name + " as default?"):
            self.config_manager.save_configs_as_default()

        CLICommandHelper.print(f"{self.config_name} added successfully.")

        return config
