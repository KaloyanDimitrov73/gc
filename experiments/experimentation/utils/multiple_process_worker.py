import os
from typing import Optional
from pydantic import BaseModel, Field

import pandas as pd

from experimentation.utils.experiment_result_processor import ExperimentResultProcessor
from experimentation.config.experiment_config import ExperimentConfig
from sqa_system.core.config.models import PipelineConfig
from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.logging.logging import get_logger

from ..evaluation.base.evaluator import Evaluator
from .experiment_runner_settings import ExperimentRunnerSettings
from .executor.parallel_evaluation_experiment_executor import ParallelEvaluationExperimentExecutor
from .executor.base.base_experiment_executor import ExperimentExecutorSettings
from .executor.sequential_experiment_executor import SequentialExperimentExecutor
from .executor.no_evaluation_experiment_executor import NoEvaluationExperimentExecutor

logger = get_logger(__name__)


class ExperimentWorkerData(BaseModel):
    """
    This class is used to pass data to the experiment worker.
    """
    config: PipelineConfig = Field(
        ..., description="The configuration of the pipeline to run.")
    settings: ExperimentRunnerSettings = Field(
        ..., description="The settings for the experiment runner.")
    experiment_config: ExperimentConfig = Field(
        ..., description="The experiment overall experiment configuration object.")
    weave_project_name: Optional[str] = Field(
        default=None, description="The name of the Weave project to use for tracking.")
    results_folder_path: str = Field(
        ..., description="The path to the folder where the results should be saved.")


def experiment_worker(strategy: str,
                      worker_data: ExperimentWorkerData,
                      prepared_dataset: pd.DataFrame,
                      evaluators: list[Evaluator],
                      result_processor: ExperimentResultProcessor) -> pd.DataFrame:
    """
    To be able to run experiments in true cpu parallelism, this function is used
    to run the experiment in a separate process.

    Args:
        strategy (str): The execution strategy to use.
        worker_data (ExperimentWorkerData): The data to pass to the worker.
        prepared_dataset (pd.DataFrame): The QA dataset to use for the experiment.
        evaluators (list[Evaluator]): The evaluators that evaluate the results.
        result_processor (ExperimentResultProcessor): The result processor that
            processes the results.
            
    Returns:
        pd.DataFrame: The results of the experiment.
    """

    # Because we are running multiple processes we need to set the logger
    # to a different file for each process.
    process_id = os.getpid()
    logger.set_log_file_location(
        FilePathManager().combine_paths(
            worker_data.results_folder_path, "experiment_pid_" +
            str(process_id) + ".log"
        )
    )

    if strategy == "sequential":
        executor = SequentialExperimentExecutor(
            number_of_workers=worker_data.settings.number_of_workers,
            dataset=prepared_dataset,
            evaluators=evaluators,
            result_processor=result_processor,
            executor_settings=ExperimentExecutorSettings(
                experiment_config=worker_data.experiment_config,
                weave_project_name=worker_data.weave_project_name
            )
        )
    elif strategy == "parallel_evaluation":
        executor = ParallelEvaluationExperimentExecutor(
            dataset=prepared_dataset,
            evaluators=evaluators,
            result_processor=result_processor,
            executor_settings=ExperimentExecutorSettings(
                experiment_config=worker_data.experiment_config,
                weave_project_name=worker_data.weave_project_name
            )
        )
    elif strategy == "no_evaluation":
        executor = NoEvaluationExperimentExecutor(
            dataset=prepared_dataset,
            evaluators=evaluators,
            result_processor=result_processor,
            executor_settings=ExperimentExecutorSettings(
                experiment_config=worker_data.experiment_config,
                weave_project_name=worker_data.weave_project_name
            )
        )
    else:
        raise ValueError(f"Unknown execution strategy: {strategy}")

    return executor.execute_experiments(
        configs=[worker_data.config],
        results_folder_path=worker_data.settings.results_folder_path,
        visualize_results=worker_data.settings.visualize_results
    )
