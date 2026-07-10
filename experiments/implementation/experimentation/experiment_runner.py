from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional
import pandas as pd
from dotenv import load_dotenv, find_dotenv


from implementation.file_path_management import FilePathManager
from implementation.experimentation.utils.experiment_result_processor import ExperimentResultProcessor
from implementation.experimentation.utils.experiment_preparer import ExperimentPreparer
from core.progress.progress_handler import ProgressHandler
from implementation.config.config_models.experiment_config import ExperimentConfig
from implementation.core.logging import get_logger

from .utils.experiment_runner_settings import ExperimentRunnerSettings, ExecutionStrategyType
from .utils.multiple_process_worker import ExperimentWorkerData, experiment_worker
from .utils.executor.base.base_experiment_executor import BaseExperimentExecutor, ExperimentExecutorSettings
from .utils.executor.parallel_evaluation_experiment_executor import ParallelEvaluationExperimentExecutor
from .utils.executor.sequential_experiment_executor import SequentialExperimentExecutor
from .utils.executor.no_evaluation_experiment_executor import NoEvaluationExperimentExecutor

logger = get_logger(__name__)


class ExperimentRunner:
    """
    The experiment runner is responsible for running a pipeline experiment.
    It is the main class that realizes the experiments in the SQA system.

    The class receives a ExperimentConfig object and then generates the
    concrete pipeline configurations based on the parameter ranges defined in the
    experiment configuration. It then runs the experiments on the generated
    configurations using the selected execution strategy. The results are
    processed and returned as a pandas DataFrame.

    Args:
        experiment_config (ExperimentConfig): The configuration object for the experiment.
        settings (ExperimentRunnerSettings): The settings for the experiment runner.
    """

    def __init__(self,
                 experiment_config: ExperimentConfig,
                 settings: ExperimentRunnerSettings = ExperimentRunnerSettings()):
        load_dotenv(find_dotenv(), override=True)
        self.settings = settings
        self.progress_handler = ProgressHandler()
        self.experiment_preparer = ExperimentPreparer(
            experiment_config, settings.qa_data_path)
        self.debugging_mode = settings.debugging
        self.log_to_results_folder = settings.log_to_results_folder
        self.execution_strategy = settings.execution_strategy
        self._prepare_results_folder_path(settings.results_folder_path)
        self._prepare_utility()
        self._prepare_experiment()

    def _prepare_results_folder_path(self, results_folder_path: Optional[str] = None):
        """
        Prepares the folder path in which the results should be saved.

        Args:
            results_folder_path (Optional[str]): The path to the folder where the results should be saved.
                If not provided a default path will be created.
        """
        self.file_path_manager = FilePathManager()
        if results_folder_path is None:
            self.results_folder_path = self.file_path_manager.create_evaluation_result_path()
        else:
            self.results_folder_path = results_folder_path

    def _prepare_utility(self):
        """
        Prepares the result processor, sets the debugging mode and may change
        the location of the log file if the user wants to log to the results folder.
        """
        self.result_processor = ExperimentResultProcessor()
        if self.debugging_mode:
            logger.logger.setLevel("DEBUG")
        else:
            logger.logger.setLevel("INFO")

        if self.log_to_results_folder:
            logger.set_log_file_location(self.file_path_manager.combine_paths(
                self.results_folder_path, "experiment.log"))

    def _prepare_experiment(self):
        self.prepared_dataset = self.experiment_preparer.prepare_dataset()
        self.questions_df = pd.DataFrame(self.prepared_dataset)
        self.evaluators = self.experiment_preparer.prepare_evaluators()
        self.generated_configs = self.experiment_preparer.prepare_pipeline_configs(
            skip_base_config=self.settings.skip_base_config)

    def run(self) -> pd.DataFrame:
        """
        Runs experiments using the selected execution strategy.

        Returns:
            pd.DataFrame: The results of the experiment as a pandas DataFrame.
        """

        self.progress_handler.add_task(
            string_id="configuration",
            description="Running Experiments...",
            total=len(self.generated_configs),
            reset=True
        )

        if self.settings.number_of_processes > 1:
            all_results = self._run_in_multiple_processes()
        else:
            experiment_executor = self._create_execution_strategy(
                self.execution_strategy)

            all_results = experiment_executor.execute_experiments(
                configs=self.generated_configs,
                results_folder_path=self.results_folder_path,
                visualize_results=self.settings.visualize_results
            )

        # Create the final dataframe
        final_df = self.result_processor.create_result_dataframe(
            all_results,
            self.questions_df
        )
        return final_df

    def _run_in_multiple_processes(self):
        """
        This method runs the experiments in multiple processes using the
        ProcessPoolExecutor. Essentially if the experiment has multiple configurations
        that are generated from the base pipeline with the ParameterRanges, we can
        run them in parallel. 
        """
        number_of_processed_needed = min(
            self.settings.number_of_processes, len(self.generated_configs))
        logger.info(
            f"Running experiments in {number_of_processed_needed} processes.")
        if not self.generated_configs:
            return []
        results = []
        with ProcessPoolExecutor(max_workers=number_of_processed_needed) as executor:
            futures = {}
            for config in self.generated_configs:
                # Submit the experiment worker to the executor
                futures[executor.submit(experiment_worker,
                                        self.execution_strategy,
                                        ExperimentWorkerData(
                                            config=config,
                                            settings=self.settings,
                                            experiment_config=self.experiment_preparer.experiment_config,
                                            weave_project_name=self.settings.weave_project_name,
                                            results_folder_path=self.results_folder_path,
                                            skip_weave=self.settings.skip_weave,
                                        ),
                                        self.prepared_dataset,
                                        self.evaluators,
                                        self.result_processor)] = config

            for fut in as_completed(futures):
                try:
                    results.extend(fut.result())
                    self.progress_handler.update_task_by_string_id(
                        "configuration")
                except KeyboardInterrupt:
                    logger.warning("Experiments interrupted by user.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception as e:
                    logger.error(f"Error in experiment worker: {e}")
                    self.progress_handler.update_task_by_string_id(
                        "configuration")
            executor.shutdown(wait=True)
        self.progress_handler.clear()
        return results

    def _create_execution_strategy(self,
                                   strategy_type: ExecutionStrategyType) -> BaseExperimentExecutor:
        """
        Create the appropriate executor based on the type.

        Args:
            strategy_type (ExecutionStrategyType): The type of execution
            strategy to create.

        Returns:
            BaseExperimentExecutor: The executor for the experiment.
        """
        if strategy_type == ExecutionStrategyType.SEQUENTIAL:
            return SequentialExperimentExecutor(
                number_of_workers=self.settings.number_of_workers,
                dataset=self.prepared_dataset,
                evaluators=self.evaluators,
                result_processor=self.result_processor,
                executor_settings=ExperimentExecutorSettings(
                    experiment_config=self.experiment_preparer.experiment_config,
                    weave_project_name=self.settings.weave_project_name,
                    skip_weave=self.settings.skip_weave,
                )
            )
        if strategy_type == ExecutionStrategyType.PARALLEL_EVALUATION:
            return ParallelEvaluationExperimentExecutor(
                dataset=self.prepared_dataset,
                evaluators=self.evaluators,
                result_processor=self.result_processor,
                executor_settings=ExperimentExecutorSettings(
                    experiment_config=self.experiment_preparer.experiment_config,
                    weave_project_name=self.settings.weave_project_name,
                    skip_weave=self.settings.skip_weave,
                )
            )
        if strategy_type == ExecutionStrategyType.NO_EVALUATION:
            return NoEvaluationExperimentExecutor(
                dataset=self.prepared_dataset,
                evaluators=self.evaluators,
                result_processor=self.result_processor,
                executor_settings=ExperimentExecutorSettings(
                    experiment_config=self.experiment_preparer.experiment_config,
                    weave_project_name=self.settings.weave_project_name,
                    skip_weave=self.settings.skip_weave,
                )
            )
        raise ValueError(f"Unknown execution strategy: {strategy_type}")
