from typing_extensions import override

from sqa_system.app.cli.menu.base.cli_menu import CLIMenu
from sqa_system.app.cli.cli_command_helper import CLICommandHelper, Choice
from sqa_system.app.cli.handler import LLMConfigCommandHandler
from sqa_system.app.cli.menu.implementations.dataset_config_menu import DatasetConfigMenu
from sqa_system.app.cli.menu.implementations.pipe_config_menu import PipeConfigMenu
from sqa_system.app.cli.handler import VectorStoreConfigCommandHandler
from sqa_system.app.cli.handler import ChunkConfigCommandHandler
from sqa_system.app.cli.handler import EmbeddingsConfigCommandHandler
from sqa_system.app.cli.handler import PipelineConfigCommandHandler
from sqa_system.app.cli.handler import KGStorageConfigCommandHandler


class ConfigurationMenu(CLIMenu):
    """
    This class is used to display a menu through the CLI interface to allow
    users to manage the different configurations of the system.
    """

    def __init__(self):
        # llm related
        self.llm_config_handler = LLMConfigCommandHandler()

        # vector store related
        self.chunk_config_handler = ChunkConfigCommandHandler()
        self.embedding_config_handler = EmbeddingsConfigCommandHandler()
        self.vector_store_config_handler = VectorStoreConfigCommandHandler()

        # knowledge graph related
        self.kg_storage_config_handler = KGStorageConfigCommandHandler()

        # pipeline related
        self.pipeline_config_handler = PipelineConfigCommandHandler()

        # additional menus
        self.dataset_config_menu = DatasetConfigMenu()
        self.pipe_config_menu = PipeConfigMenu()

    @override
    def get_action(self):
        action = CLICommandHelper.get_selection(
            "Select the configuration to manage",
            [
                Choice("llm", "Manage LLM configurations"),
                Choice("dataset", "Manage dataset configurations"),
                Choice("chunking", "Manage chunking configurations"),
                Choice("embeddings", "Manage embedding configurations"),
                Choice("vector_store", "Manage vector store configurations"),
                Choice("knowledge_graph",
                       "Manage knowledge graph configurations"),
                Choice("pipe", "Manage pipe configurations"),
                Choice("pipeline", "Manage pipeline configurations"),
                Choice("back", "Back to main menu"),
            ],
            include_exit=False
        )
        return action

    @override
    def handle_action(self, action):
        if action == "llm":
            self.llm_config_handler.handle_command()
        elif action == "dataset":
            self.dataset_config_menu.run()
        elif action == "chunking":
            self.chunk_config_handler.handle_command()
        elif action == "embeddings":
            self.embedding_config_handler.handle_command()
        elif action == "vector_store":
            self.vector_store_config_handler.handle_command()
        elif action == "pipeline":
            self.pipeline_config_handler.handle_command()
        elif action == "knowledge_graph":
            self.kg_storage_config_handler.handle_command()
        elif action == "pipe":
            self.pipe_config_menu.run()
