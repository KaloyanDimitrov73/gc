from typing import List, Optional
import time
from pydantic import BaseModel, Field
import weave
from weave import Model
from langchain_core.runnables import Runnable

from implementation.pipeline.tracking.emission_tracker_manager import EmissionTrackerManager
from language_model.llm_stat_tracker import LLMStatTracker
from implementation.pipeline.tracking.pipeline_data_collector import PipelineDataCollector, PipelineData
from implementation.config.config_models.pipeline_config import PipelineConfig
from implementation.pipe.base.pipe import Pipe
from implementation.shared_models.pipe_io_data import PipeIOData
from core.progress.progress_handler import ProgressHandler
from implementation.core.logging import get_logger
logger = get_logger(__name__)


class RetrievalPipeline(Model, BaseModel):
    """
    The 'RetrievalPipeline' class is used to chain multiple 'Pipe' objects.

    It is implemented as a model from the 'weave' library which allows to
    track the data flow through the pipeline. Documentation is found here: 
    https://weave-docs.wandb.ai/guides/core-types/models

    Args:
        pipes (List[Pipe]): A list of 'Pipe' objects to be chained together.
        config (Optional[PipelineConfig]): The configuration object for the pipeline.
        answer_collector (Optional[PipelineDataCollector]): The answer collector
            object to collect the answers from the pipeline. This is required as a 
            workaround to the Weave library which does not support the 'PipeIOData' object
            as an output. 
    """
    pipes: List[Pipe] = Field(default_factory=list)
    config: Optional[PipelineConfig] = None
    answer_collector: Optional[PipelineDataCollector] = None

    def __or__(self, other):
        if isinstance(other, Pipe):
            return RetrievalPipeline(pipes=self.pipes + [other])
        if isinstance(other, RetrievalPipeline):
            return RetrievalPipeline(pipes=self.pipes + other.pipes)

        raise ValueError("Can only chain Pipe or Pipeline objects")

    @weave.op()
    def predict(self,
                question: str,
                topic_entity_id: str = "",
                topic_entity_value: str = "",
                uid: str = "",
                message_history: str = None) -> dict:
        """
        This method is necessary for the weave tool as it
        outputs the data as a dict which is required by weaver
        to display the data on the Weight&Biased dashboard:
        https://wandb.ai/
        """
        pipeline_data = self.run(
            question, topic_entity_id, topic_entity_value, uid, message_history)
        return pipeline_data.pipe_io_data.model_dump()

    def run(self,
            input_str: str,
            topic_entity_id: str = "",
            topic_entity_value: str = "",
            question_id: str = None,
            message_history: str = None) -> PipelineData:
        """
        The main method of the 'RetrievalPipeline' class. It is used to run the pipeline.

        It generates the initial 'PipeIOData' object and passes it through the pipeline.
        Each pipe then manipulates or extends the data in the 'PipeIOData' object. 
        The final output of the pipeline which is the filled 'PipeIOData' object and
        additional tracking data is returned in form of a 'PipelineData' object.

        Args:
            input_str (str): The input string to be processed by the pipeline.
            topic_entity_id (str, optional): The ID of the topic entity which is
            the entity in the graph from which the pipeline starts. Defaults to "".
            topic_entity_value (str, optional): The value of the topic entity.
            question_id (str, optional): The ID of the question that uniquely identifies
                the question.

        Returns:
            PipelineData: The output of the pipeline which is a 'PipelineData' object
            containing the filled 'PipeIOData' object and additional tracking data.
        """
        # Prepare the chain object
        chain = self._prepare_chain()

        # Prepare the progress bar
        progress_id = self._prepare_progress_bar()

        # Prepare the input data
        pipe_input = PipeIOData(initial_question=input_str,
                                retrieval_question=input_str,
                                topic_entity_id=topic_entity_id,
                                topic_entity_value=topic_entity_value,
                                progress_bar_id=progress_id,
                                question_id=question_id,
                                message_history=message_history)

        retries = 3
        while retries > 0:
            # For tracking the stats of the LLM we prepare the
            # StatTracker
            llm_stat_tracker = LLMStatTracker()
            llm_stat_tracker.reset()

            # For tracking the emissions we prepare the EmissionTracker
            emission_tracker = EmissionTrackerManager()
            emission_tracker.start()
            runtime = 0.0
            start_time = time.time()
            try:
                # Here we run the pipeline
                output = chain.invoke(pipe_input)

            except Exception as e:
                logger.error(f"Pipeline execution failed: {e}")
                logger.info(
                    f"Retrying pipeline execution ({3 - retries + 1}/3)")
                emission_tracker.stop_and_get_results()
                if retries == 1:
                    logger.error("Pipeline execution failed after 3 retries")
                    raise e
                retries -= 1
                continue
            end_time = time.time()
            runtime = end_time - start_time
            break

        pipeline_data = PipelineData(
            pipe_io_data=output,
            runtime=runtime,
            llm_stats=llm_stat_tracker.get_stats(),
            emissions_data=emission_tracker.stop_and_get_results(),
            weave_url=self._get_weave_url()
        )

        if self.answer_collector:
            self.answer_collector.add(
                data=pipeline_data
            )
        llm_stat_tracker.reset()
        logger.info(f"Pipeline Runtime: {runtime:.2f} seconds")

        ProgressHandler().finish_by_string_id(progress_id)
        ProgressHandler().update_task_by_string_id("asking_questions")

        return pipeline_data

    def _prepare_progress_bar(self) -> str:
        """
        Prepares the progress bar for the pipeline execution.

        Returns:
            str: The ID of the progress bar.
        """
        progress_id = "pipeline"
        if self.config is not None:
            progress_id = f"pipeline_{self.config.name}"
        ProgressHandler().add_task(
            progress_id,
            "Running pipeline",
            total=len(self.pipes)
        )
        return progress_id

    def _prepare_chain(self) -> Runnable:
        """Chains multiple Langchain runnable objects together."""
        if not self.pipes:
            logger.error("Pipeline cannot be empty")
            raise ValueError("Pipeline cannot be empty")
        pipes: List[Pipe] = list(self.pipes)

        chain = pipes[0].get_runnable()
        for pipe in pipes[1:]:
            if not isinstance(pipe, Pipe):
                logger.error("Pipeline can only contain Pipe objects")
                raise ValueError("Pipeline can only contain Pipe objects")
            next_pipe = pipe.get_runnable()
            if not next_pipe:
                logger.error("Pipeline contains an empty Pipe")
                raise ValueError("Pipeline contains an empty Pipe")
            chain = chain | next_pipe

        if not chain:
            logger.error("Pipeline must have at least one Pipe")
            raise ValueError("Pipeline must have at least one Pipe")

        return chain

    def _get_weave_url(self) -> str:
        """
        Returns the Weave URL of the current call.
        """
        current_call = weave.get_current_call()
        if current_call:
            return current_call.ui_url
        else:
            return ""
