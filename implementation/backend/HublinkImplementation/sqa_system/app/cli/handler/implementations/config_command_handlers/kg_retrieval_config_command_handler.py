# standard imports
from typing_extensions import override

# local imports
from sqa_system.core.config.factory.config_factory import ConfigFactory
from sqa_system.core.config.models import KGRetrievalConfig
from sqa_system.app.cli.handler.base.config_command_handler import ConfigCommandHandler
from sqa_system.core.config.config_manager import (
    LLMConfigManager,
    KGStoreConfigManager,
    KGRetrievalConfigManager
)
from sqa_system.app.cli.cli_command_helper import (
    CLICommandHelper,
    Choice
)
from sqa_system.retrieval import (
    KnowledgeGraphRetrieverType,
    KnowledgeGraphRetrieverFactory
)
from .llm_config_command_handler import LLMConfigCommandHandler
from .kg_storage_config_command_handler import KGStorageConfigCommandHandler


class KGRetrievalConfigCommandHandler(ConfigCommandHandler[KGRetrievalConfig]):
    """
    Handles commandes related to the knowledge graph retrieval management
    """

    def __init__(self):
        super().__init__(KGRetrievalConfigManager(), "KG Retrieval")
        self.llm_config_manager = LLMConfigManager()
        self.kg_config_manager = KGStoreConfigManager()
        self.kg_config_manager.load_default_configs()
        self.llm_config_manager.load_default_configs()
        self.config_manager.load_default_configs()
        

    @override
    def add_config(self):

        name = CLICommandHelper.get_text_input(
            message="Enter a name for the KG retrieval config",
            default="",
            instruction="Leave empty for automatic name generation"
        )

        retriever_type = CLICommandHelper.get_selection(
            "Select retriever type:",
            [Choice(retriever_type.value, retriever_type.value)
             for retriever_type in KnowledgeGraphRetrieverType],
        )
        if CLICommandHelper.is_exit_selection(retriever_type):
            return None
        
        retriever_class = KnowledgeGraphRetrieverFactory.get_retriever_class(retriever_type)

        kg_retriever_params = self._prepare_knowledge_graph_retriever_params()
        if kg_retriever_params is None:
            return None
        
        additional_config_params = self._get_additional_config_params(
            retriever_class.ADDITIONAL_CONFIG_PARAMS)
        if not additional_config_params:
            config = ConfigFactory.create_kg_retrieval_config(
                retriever_type=retriever_type,
                name=name,
                **kg_retriever_params)
        else:
            config = ConfigFactory.create_kg_retrieval_config(
                retriever_type=retriever_type,
                name=name,
                **kg_retriever_params,
                **additional_config_params)

        self.config_manager.add_config(config)
        if CLICommandHelper.get_confirmation("Add this " + self.config_name + " as default?"):
            self.config_manager.save_configs_as_default()

        CLICommandHelper.print(f"{self.config_name} added successfully.")

        return config

    def _prepare_knowledge_graph_retriever_params(self) -> dict:
        knowledge_graph_config = CLICommandHelper.get_selection(
            "Select knowledge graph config:",
            [Choice(config, config.name)
             for config in self.kg_config_manager.get_all_configs()] +
            [Choice(None, "Add new knowledge graph config")]
        )
        if knowledge_graph_config is None:
            knowledge_graph_config = KGStorageConfigCommandHandler().add_config()
            if knowledge_graph_config is None:  # if the user exited the add config
                return None
        if CLICommandHelper.is_exit_selection(knowledge_graph_config):
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
        
        return {
            "knowledge_graph_config": knowledge_graph_config,
            "llm_config": llm_config
        }