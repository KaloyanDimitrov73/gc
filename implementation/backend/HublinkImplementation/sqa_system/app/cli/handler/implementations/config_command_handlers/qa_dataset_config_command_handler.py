import os
from typing_extensions import override
from sqa_system.app.cli.handler.base.config_command_handler import ConfigCommandHandler
from sqa_system.core.config.models.dataset_config import DatasetConfig
from sqa_system.core.config.config_manager import QADatasetConfigManager
from sqa_system.core.config.factory.config_factory import ConfigFactory
from sqa_system.core.data.data_loader.factory.data_loader_factory import (
    DataLoaderFactory,
    DataLoaderType
)
from sqa_system.app.cli.cli_command_helper import CLICommandHelper


class QADatasetConfigCommandHandler(ConfigCommandHandler[DatasetConfig]):
    """Class for handling commands related to the dataset management."""

    def __init__(self):
        super().__init__(QADatasetConfigManager(), "Dataset")
        self.config_manager.load_default_configs()

    @override
    def add_config(self):
        """Adds a new dataset configuration to the manager."""
        file_path = CLICommandHelper.get_text_input(
            "Enter file path (relative to the data folder):")
        file_path = os.path.join(
            self.config_manager.path_manager.DATA_DIR, file_path)
        file_path = os.path.normpath(file_path)
        file_name = self.config_manager.path_manager.get_file_name_from_path(
            file_path)
        available_loaders = DataLoaderFactory.get_all_data_loaders(
            DataLoaderType.QUESTION_ANSWERING)
        loader = CLICommandHelper.get_selection(
            "Select loader:", list(available_loaders.keys()))
        if CLICommandHelper.is_exit_selection(loader):
            return None
        loader_limit = CLICommandHelper.get_text_input(
            "Limit the amounts of questions and answers to load (-1 for unlimited):", default="-1")

        config = ConfigFactory.create_dataset_config(
            file_name=file_name,
            loader=loader,
            loader_limit=int(loader_limit))

        self.config_manager.add_config(config)
        self.config_manager.path_manager.add_path(file_path)

        if CLICommandHelper.get_confirmation("Add this dataset as default?"):
            self.config_manager.save_configs_as_default()
            self.config_manager.path_manager.save_paths_as_default()

        CLICommandHelper.print_success("Dataset added successfully.")

        return config
