from typing_extensions import override
from InquirerPy.base.control import Choice

from sqa_system.core.config.factory.config_factory import ConfigFactory
from sqa_system.app.cli.cli_command_helper import CLICommandHelper
from sqa_system.app.cli.handler.base.config_command_handler import ConfigCommandHandler
from sqa_system.core.config.config_manager import (
    KGRetrievalConfigManager,
    GenerationConfigManager,
    PipelineConfigManager,
    PostRetrievalConfigManager,
    DocumentRetrievalConfigManager,
    PreRetrievalConfigManager
)
from sqa_system.core.config.models import (
    PipelineConfig,
    PipeConfig,
    GenerationConfig,
    PostRetrievalConfig,
    PreRetrievalConfig,
    RetrievalConfig
)

from .generation_config_command_handler import GenerationConfigCommandHandler
from .kg_retrieval_config_command_handler import KGRetrievalConfigCommandHandler
from .document_retrieval_config_command_handler import DocumentRetrievalConfigCommandHandler
from .post_retrieval_config_command_handler import PostRetrievalConfigCommandHandler
from .pre_retrieval_config_command_handler import PreRetrievalConfigCommandHandler


class PipelineConfigCommandHandler(ConfigCommandHandler[PipelineConfig]):
    """Handles commands related to the pipeline management"""

    def __init__(self):
        super().__init__(PipelineConfigManager(), "Pipeline")
        self.generation_config_manager = GenerationConfigManager()
        self.document_retrieval_config_manager = DocumentRetrievalConfigManager()
        self.knowledge_graph_retrieval_config_manager = KGRetrievalConfigManager()
        self.post_retrieval_config_manager = PostRetrievalConfigManager()
        self.pre_retrieval_config_manager = PreRetrievalConfigManager()
        self.generation_config_manager.load_default_configs()
        self.document_retrieval_config_manager.load_default_configs()
        self.knowledge_graph_retrieval_config_manager.load_default_configs()
        self.post_retrieval_config_manager.load_default_configs()
        self.pre_retrieval_config_manager.load_default_configs()
        self.config_manager.load_default_configs()

    @override
    def add_config(self):
        pipes: list[PipeConfig] = []

        name = CLICommandHelper.get_text_input(
            message="Enter a name for the pipeline config",
            default="",
            instruction="Leave empty for automatic name generation"
        )

        # First of, pre-retrieval pipes can be added
        pre_retrieval_pipe_configs = self._get_pre_retrieval_pipes()
        if pre_retrieval_pipe_configs is None:
            return None

        # Now a retrieval pipe needs to be added
        retrieval_config = self._get_retrieval_pipe()

        # Now we add the post retrieval pipe
        post_retrieval_pipe_configs = self._get_post_retrieval_pipes()
        if post_retrieval_pipe_configs is None:
            return None

        # Finally we add the generation pipe
        generation_config = CLICommandHelper.get_selection(
            "Select Generation Pipe",
            [Choice(config, config.name)
                for config in self.generation_config_manager.get_all_configs()] +
            [Choice(None, "Add new generation pipe")],
            exit_text="Stop adding pipeline"
        )
        if CLICommandHelper.is_exit_selection(generation_config):
            return None
        if generation_config is None:
            generation_config = GenerationConfigCommandHandler().add_config()
            if generation_config is None:  # Check if user exited during new config creation
                return None

        # Construct the pipe list
        if len(pre_retrieval_pipe_configs) > 0:
            pipes.extend(pre_retrieval_pipe_configs)
        pipes.append(retrieval_config)
        if len(post_retrieval_pipe_configs) > 0:
            pipes.extend(post_retrieval_pipe_configs)
        pipes.append(generation_config)

        config = ConfigFactory.create_pipeline_config(pipes=pipes, name=name)
        self.config_manager.add_config(config)
        CLICommandHelper.print("Pipeline added successfully.")

        if CLICommandHelper.get_confirmation("Add this pipeline as default?"):
            self.config_manager.save_configs_as_default()

        return config

    def _get_generation_pipe(self) -> GenerationConfig | None:
        generation_config = CLICommandHelper.get_selection(
            "Select Generation Pipe",
            [Choice(config, config.name)
                for config in self.generation_config_manager.get_all_configs()] +
            [Choice(None, "Add new generation pipe")],
            exit_text="Stop adding pipeline"
        )
        if CLICommandHelper.is_exit_selection(generation_config):
            return None
        if generation_config is None:
            generation_config = GenerationConfigCommandHandler().add_config()
            if generation_config is None:  # Check if user exited during new config creation
                return None
        return generation_config

    def _get_retrieval_pipe(self) -> RetrievalConfig | None:
        retrieval_config = CLICommandHelper.get_selection(
            "Add retrieval pipe:",
            [Choice(config, config.name)
                for config in self.document_retrieval_config_manager.get_all_configs()] +
            [Choice(config, config.name)
                for config in self.knowledge_graph_retrieval_config_manager.get_all_configs()] +
            [Choice(None, "Add new retrieval pipe")],
            exit_text="Stop adding pipeline"
        )
        if CLICommandHelper.is_exit_selection(retrieval_config):
            return None

        if retrieval_config is None:
            type_of_retrieval = CLICommandHelper.get_selection(
                "Select the type of retrieval",
                [Choice("knowledge_graph", "Knowledge Graph"),
                 Choice("document_retrieval", "Document Retrieval")],
            )
            if CLICommandHelper.is_exit_selection(type_of_retrieval):
                return None
            if type_of_retrieval == "knowledge_graph":
                retrieval_config = KGRetrievalConfigCommandHandler().add_config()
            elif type_of_retrieval == "document_retrieval":
                retrieval_config = DocumentRetrievalConfigCommandHandler().add_config()
            if retrieval_config is None:  # Check if user exited during new config creation
                return None
        return retrieval_config

    def _get_pre_retrieval_pipes(self) -> list[PreRetrievalConfig] | None:
        pre_retrieval_pipe_configs = []
        while True:
            pre_retr_pipe = CLICommandHelper.get_selection(
                "Add pre-retrieval pipe:",
                [Choice(None, "Add new pre-retrieval pipe")] +
                [Choice(False, "Finish adding pre-retrieval pipes")],
                exit_text="Stop adding pipeline"
            )
            if pre_retr_pipe is None:
                pre_retr_pipe = CLICommandHelper.get_selection(
                    "Select pre-retrieval pipe config:",
                    [Choice(config, config.name)
                        for config in self.pre_retrieval_config_manager.get_all_configs()] +
                    [Choice(None, "Add new Pre Retrieval config")],
                )
                if pre_retr_pipe is None:
                    pre_retr_pipe = PreRetrievalConfigCommandHandler().add_config()
                    if pre_retr_pipe is None:
                        continue
            if CLICommandHelper.is_exit_selection(pre_retr_pipe):
                return None
            if pre_retr_pipe is False:
                break
            pre_retrieval_pipe_configs.append(pre_retr_pipe)
        return pre_retrieval_pipe_configs

    def _get_post_retrieval_pipes(self) -> list[PostRetrievalConfig] | None:
        post_retrieval_pipe_configs = []
        while True:
            post_retr_pipe = CLICommandHelper.get_selection(
                "Add post-retrieval pipe:",
                [Choice(None, "Add new post-retrieval pipe")] +
                [Choice(False, "Finish adding post-retrieval pipes")],
                exit_text="Stop adding pipeline"
            )
            if post_retr_pipe is None:
                post_retr_pipe = CLICommandHelper.get_selection(
                    "Select post-retrieval pipe config:",
                    [Choice(config, config.name)
                        for config in self.post_retrieval_config_manager.get_all_configs()] +
                    [Choice(None, "Add new Post Retrieval config")],
                )
                if post_retr_pipe is None:
                    post_retr_pipe = PostRetrievalConfigCommandHandler().add_config()
                    if post_retr_pipe is None:
                        continue
                if CLICommandHelper.is_exit_selection(post_retr_pipe):
                    continue
            if CLICommandHelper.is_exit_selection(post_retr_pipe):
                return None
            if post_retr_pipe is False:
                break
            post_retrieval_pipe_configs.append(post_retr_pipe)
        return post_retrieval_pipe_configs
