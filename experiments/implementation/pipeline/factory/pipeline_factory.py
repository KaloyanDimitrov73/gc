from implementation.pipeline.retrieval_pipeline import RetrievalPipeline
from implementation.config.config_models.pipe.pipe_config import PipeConfig
from implementation.config.config_models.pipeline_config import PipelineConfig
from implementation.pipe.factory.pipe_factory import PipeFactory
from implementation.core.logging import get_logger
logger = get_logger(__name__)


class PipelineFactory:
    """
    A factory class that creates retrieval pipelines based on the specified
    configuration.
    """

    def create(self, config: PipelineConfig, **kwargs) -> RetrievalPipeline:
        """
        Create a retrieval pipeline based on the given configuration.

        Args:
            config (PipelineConfig): The configuration object for the pipeline.
        Returns:
            RetrievalPipeline: The created retrieval pipeline.
        """
        logger.debug("Creating pipeline with config: %s", config.model_dump())
        pipes = []
        for pipe_config in config.pipes:
            if not isinstance(pipe_config, PipeConfig):
                raise ValueError("Pipeline can only contain Pipe objects")
            pipe = PipeFactory.get_pipe(pipe_config)
            if pipe:
                pipes.append(pipe)
            else:
                raise ValueError("Pipeline pipe could not be loaded."
                                 " Is the configuration set up properly?")
        return RetrievalPipeline(pipes=pipes, config=config)
