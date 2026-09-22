import weave

from sqa_system.pipeline.factory.pipeline_factory import PipelineFactory
from sqa_system.core.config.config_manager import PipelineConfigManager
from sqa_system.app.cli.cli_command_helper import CLICommandHelper, Choice
from sqa_system.app.cli.handler.base.command_handler import CommandHandler
from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)


class PipelineRunCommandHandler(CommandHandler):
    """
    A command handler implementation that allows to start the interaction mode
    of a pipeline. The user can select a pipeline and then continiously input
    questions to the pipeline and receive answers.

    The interaction is done through the CLI.
    """

    def __init__(self) -> None:
        super().__init__()
        self.pipeline_config_manager = PipelineConfigManager()
        self.pipeline_config_manager.load_default_configs()

    @weave.op()
    def handle_command(self):
        weave.init("pipeline_run")
        pipeline_configs = self.pipeline_config_manager.get_all_configs()
        if len(pipeline_configs) == 0:
            logger.info("No pipelines found. Please add a pipeline first.")
            return
        choices = [
            Choice(config.uid, config.name) for config in pipeline_configs
        ]
        pipeline_uid = CLICommandHelper.get_selection(
            "Select which pipeline to run",
            choices
        )
        if CLICommandHelper.is_exit_selection(pipeline_uid):
            return

        pipeline_config = self.pipeline_config_manager.get_config(pipeline_uid)
        pipeline = PipelineFactory().create(pipeline_config)

        while True:
            CLICommandHelper.print("--" * 20)
            CLICommandHelper.print("Pipeline successfully loaded")
            CLICommandHelper.print("Type 'exit' to stop the pipeline")
            user_input = CLICommandHelper.get_text_input("Enter a question:")
            if user_input == "":
                CLICommandHelper.print("Please provide a question.")
                continue
            if user_input == "exit":
                break
            topic_entity_id = CLICommandHelper.get_text_input(
                "Provide a topic entity id (optional):", default="")
            topic_entity_id = topic_entity_id.strip()
            if topic_entity_id != "":
                topic_entity_value = CLICommandHelper.get_text_input(
                    "Provide a topic entity value (optional):", default="")
            else:
                topic_entity_value = ""
            pipeline_data = pipeline.run(
                user_input, topic_entity_id, topic_entity_value)
            CLICommandHelper.print(f"Answer: {pipeline_data.pipe_io_data.generated_answer}")
            CLICommandHelper.print(f"Tokens used: {pipeline_data.llm_stats.total_tokens}")
