from typing_extensions import override

from sqa_system.core.config.config_manager import PublicationDatasetConfigManager
from sqa_system.core.config.config_manager import KGStoreConfigManager, LLMConfigManager
from sqa_system.app.cli.handler.base.config_command_handler import ConfigCommandHandler
from sqa_system.app.cli.cli_command_helper import CLICommandHelper, Choice
from sqa_system.knowledge_base.knowledge_graph.storage import (
    KnowledgeGraphFactoryRegistry, KnowledgeGraphBuilder)
from sqa_system.core.config.models.knowledge_base.knowledge_graph_config import KnowledgeGraphConfig

from .publication_dataset_config_command_handler import PublicationDatasetConfigCommandHandler
from .llm_config_command_handler import LLMConfigCommandHandler



class KGStorageConfigCommandHandler(ConfigCommandHandler[KnowledgeGraphConfig]):
    """
    Handles commandes related to the knowledge graph storage management
    """

    def __init__(self):
        super().__init__(KGStoreConfigManager(), "Knowledge Graph Database")
        self.dataset_config_manager = PublicationDatasetConfigManager()
        self.dataset_config_manager.load_default_configs()
        self.llm_config_manager = LLMConfigManager()
        self.llm_config_manager.load_default_configs()
        self.config_manager.load_default_configs()

    @override
    def add_config(self):
        graph_type = CLICommandHelper.get_selection(
            "Select graph type: ",
            [Choice(graph_type, graph_type)
             for graph_type in KnowledgeGraphFactoryRegistry().get_all_factory_types()],
        )
        if CLICommandHelper.is_exit_selection(graph_type):
            return None

        kg_class = KnowledgeGraphFactoryRegistry().get_factory_class(graph_type)
        
        if issubclass(kg_class, KnowledgeGraphBuilder):
            datset_config = CLICommandHelper.get_selection(
                "Select dataset config: ",
                [Choice(config, config.name)
                for config in self.dataset_config_manager.get_all_configs()] +
                [Choice(None, "Add new dataset config")]
            )
            if datset_config is None:
                datset_config = PublicationDatasetConfigCommandHandler().add_config()
                if datset_config is None:  # if the user exited the add config
                    return None
            if CLICommandHelper.is_exit_selection(datset_config):
                return None
        
            llm_config = CLICommandHelper.get_selection(
                "Select LLM config for extraction:",
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

        additional_config_params = self._get_additional_config_params(
            kg_class.ADDITIONAL_CONFIG_PARAMS)
        if issubclass(kg_class, KnowledgeGraphBuilder):
            config = kg_class.create_config(
                graph_type=graph_type,
                dataset_config=datset_config,
                extraction_llm_config=llm_config,
                **additional_config_params
            )
        else:
            config = kg_class.create_config(
                graph_type=graph_type,
                **additional_config_params
            )

        self.config_manager.add_config(config)
        if CLICommandHelper.get_confirmation("Add this" + self.config_name + " as default?"):
            self.config_manager.save_configs_as_default()

        CLICommandHelper.print(f"{self.config_name} added successfully.")

        return config
