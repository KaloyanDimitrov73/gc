from typing_extensions import override

from sqa_system.core.config.models import PreRetrievalConfig
from sqa_system.pipe.factory.pipe_factory import PreProcessingTechnique
from sqa_system.core.config.config_manager import (
    PreRetrievalConfigManager,
    LLMConfigManager
)
from sqa_system.core.config.factory.config_factory import ConfigFactory
from sqa_system.app.cli.cli_command_helper import CLICommandHelper, Choice
from sqa_system.app.cli.handler.base.config_command_handler import ConfigCommandHandler
from .llm_config_command_handler import LLMConfigCommandHandler


class PreRetrievalConfigCommandHandler(ConfigCommandHandler[PreRetrievalConfig]):
    """Handles commands related to the post retrieval processing pipe management"""

    def __init__(self):
        super().__init__(PreRetrievalConfigManager(), "Pre Retrieval Pipe")
        self.llm_config_manager = LLMConfigManager()
        self.config_manager.load_default_configs()
        self.llm_config_manager.load_default_configs()

    @override
    def add_config(self):

        technique = CLICommandHelper.get_selection(
            "Select pre retrieval technique:",
            [Choice(technique.value, technique.value)
             for technique in PreProcessingTechnique],
        )
        if CLICommandHelper.is_exit_selection(technique):
            return None

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

        config = ConfigFactory.create_pre_retrieval_config(
            llm_config=llm_config,
            technique=technique,
        )

        self.config_manager.add_config(config)
        if CLICommandHelper.get_confirmation("Add this " + self.config_name + " as default?"):
            self.config_manager.save_configs_as_default()
        CLICommandHelper.print(f"{self.config_name} added successfully.")

        return config
