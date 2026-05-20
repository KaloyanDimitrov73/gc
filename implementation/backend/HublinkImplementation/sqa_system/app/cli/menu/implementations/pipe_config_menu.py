from typing_extensions import override

from sqa_system.app.cli.handler import KGRetrievalConfigCommandHandler
from sqa_system.app.cli.handler import DocumentRetrievalConfigCommandHandler
from sqa_system.app.cli.handler import GenerationConfigCommandHandler
from sqa_system.app.cli.handler import PreRetrievalConfigCommandHandler
from sqa_system.app.cli.handler import PostRetrievalConfigCommandHandler
from sqa_system.app.cli.menu.base.cli_menu import CLIMenu
from sqa_system.app.cli.cli_command_helper import CLICommandHelper, Choice


class PipeConfigMenu(CLIMenu):
    """User interaction menu for handling pipe configurations."""

    def __init__(self):
        self.generation_handler = GenerationConfigCommandHandler()
        self.kg_retrieval_config_handler = KGRetrievalConfigCommandHandler()
        self.document_retrieval_config_handler = DocumentRetrievalConfigCommandHandler()
        self.pre_retrieval_config_handler = PreRetrievalConfigCommandHandler()
        self.post_retrieval_config_handler = PostRetrievalConfigCommandHandler()

    @override
    def get_action(self):
        action = CLICommandHelper.get_selection(
            "Select the pipe configuration to manage",
            [
                Choice("generation", "Manage generation pipe configurations"),
                Choice(
                    "knowledge_graph_retrieval",
                    "Manage knowledge graph retrieval pipe configurations"),
                Choice("document_retrieval",
                       "Manage document retrieval pipe configurations"),
                Choice("pre_retrieval",
                       "Manage pre-retrieval pipe configurations"),
                Choice("post_retrieval",
                       "Manage post-retrieval pipe configurations"),
                Choice("back", "Back to main menu"),
            ],
            include_exit=False
        )
        return action

    @override
    def handle_action(self, action):
        if action == "generation":
            self.generation_handler.handle_command()
        elif action == "knowledge_graph_retrieval":
            self.kg_retrieval_config_handler.handle_command()
        elif action == "document_retrieval":
            self.document_retrieval_config_handler.handle_command()
        elif action == "pre_retrieval":
            self.pre_retrieval_config_handler.handle_command()
        elif action == "post_retrieval":
            self.post_retrieval_config_handler.handle_command()
