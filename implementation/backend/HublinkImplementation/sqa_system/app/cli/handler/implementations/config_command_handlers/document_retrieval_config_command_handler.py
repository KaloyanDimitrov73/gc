# standard imports
from typing_extensions import override

# local imports
from sqa_system.core.config.factory.config_factory import ConfigFactory
from sqa_system.core.config.models import DocumentRetrievalConfig
from sqa_system.app.cli.handler.base.config_command_handler import ConfigCommandHandler
from sqa_system.core.config.config_manager import (
    DocumentRetrievalConfigManager,
    PublicationDatasetConfigManager
)
from sqa_system.app.cli.cli_command_helper import (
    CLICommandHelper,
    Choice
)
from sqa_system.retrieval import (
    DocumentRetrieverType,
    DocumentRetrieverFactory
)
from .publication_dataset_config_command_handler import PublicationDatasetConfigCommandHandler


class DocumentRetrievalConfigCommandHandler(ConfigCommandHandler[DocumentRetrievalConfig]):
    """
    Handles commandes related to the knowledge graph retrieval management
    """

    def __init__(self):
        super().__init__(DocumentRetrievalConfigManager(), "Document Retrieval")
        self.dataset_config_manager = PublicationDatasetConfigManager()
        self.dataset_config_manager.load_default_configs()
        self.config_manager.load_default_configs()

    @override
    def add_config(self):

        name = CLICommandHelper.get_text_input(
            message="Enter a name for the Document retrieval config",
            default="",
            instruction="Leave empty for automatic name generation"
        )

        retriever_type = CLICommandHelper.get_selection(
            "Select retriever type:",
            [Choice(retriever_type.value, retriever_type.value)
             for retriever_type in DocumentRetrieverType],
        )
        if CLICommandHelper.is_exit_selection(retriever_type):
            return None
        
        retriever_class = DocumentRetrieverFactory.get_retriever_class(retriever_type)
        
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
        
        additional_config_params = self._get_additional_config_params(
            retriever_class.ADDITIONAL_CONFIG_PARAMS)
        if not additional_config_params:
            config = ConfigFactory.create_doc_retrieval_config(
                retriever_type=retriever_type,
                name=name,
                dataset_config=datset_config)
        else:
            config = ConfigFactory.create_doc_retrieval_config(
                retriever_type=retriever_type,
                name=name,
                dataset_config=datset_config,
                **additional_config_params)

        self.config_manager.add_config(config)
        if CLICommandHelper.get_confirmation("Add this " + self.config_name + " as default?"):
            self.config_manager.save_configs_as_default()

        CLICommandHelper.print(f"{self.config_name} added successfully.")

        return config