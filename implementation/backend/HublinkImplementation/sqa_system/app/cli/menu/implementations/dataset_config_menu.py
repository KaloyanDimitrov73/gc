from typing_extensions import override

from sqa_system.app.cli.handler import QADatasetConfigCommandHandler
from sqa_system.app.cli.handler import PublicationDatasetConfigCommandHandler
from sqa_system.app.cli.menu.base.cli_menu import CLIMenu
from sqa_system.app.cli.cli_command_helper import CLICommandHelper, Choice


class DatasetConfigMenu(CLIMenu):
    """
    This class is used to display a menu to the user through the CLI interface
    to allow to manage the different dataset configurations of the system.
    """

    def __init__(self) -> None:
        self.dataset_config_handler = PublicationDatasetConfigCommandHandler()
        self.qa_dataset_config_handler = QADatasetConfigCommandHandler()

    @override
    def get_action(self):
        action = CLICommandHelper.get_selection(
            "Select the dataset configuration to manage",
            [
                Choice("qa", "Manage Question-Answering dataset configurations"),
                Choice("publications", "Manage Publication dataset configurations"),
                Choice("back", "Back to main menu"),
            ],
            include_exit=False
        )
        return action

    @override
    def handle_action(self, action):
        if action == "qa":
            self.qa_dataset_config_handler.handle_command()
        elif action == "publications":
            self.dataset_config_handler.handle_command()
