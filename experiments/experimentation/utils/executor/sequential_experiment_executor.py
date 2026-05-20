import os
from typing import List, Dict, Any
from typing_extensions import override

from sqa_system.core.data.emission_tracker_manager import EmissionTrackerManager
from sqa_system.core.logging.logging import get_logger
from sqa_system.core.config.models.pipeline_config import PipelineConfig
from .base.base_experiment_executor import BaseExperimentExecutor

logger = get_logger(__name__)


class SequentialExperimentExecutor(BaseExperimentExecutor):
    """
    Executes pipelines and evaluations sequentially.
    Each pipeline is fully evaluated before moving to the next one.
    """

    def __init__(self,
                 number_of_workers: int,
                 **kwargs: Any):
        super().__init__(**kwargs)
        self.number_of_workers = number_of_workers

    @override
    def execute_experiments(self,
                            results_folder_path: str,
                            configs: List[PipelineConfig],
                            visualize_results: bool) -> List[Dict[str, Any]]:
        # Because the emission tracking doesnt work in parallel
        # we disable the paralism here.
        logger.info(
            f"Running Sequential Experiment Executor with {self.number_of_workers} threads.")
        os.environ["WEAVE_PARALLELISM"] = str(self.number_of_workers)
        self._disable_emission_tracker_if_necessary()
        all_results = []

        for config in configs:
            result = self._run_with_weave_evaluation(config)

            result_df = self.result_processor.create_result_dataframe(
                result,
                self.prepared_qa_dataset_df
            )
            self._save_results(
                results_folder_path=results_folder_path,
                config=config,
                data_to_save=result_df
            )

            all_results.extend(result)
            self.progress_handler.update_task_by_string_id("configuration")
        if visualize_results:
            self._visualize_results(
                results_folder_path=results_folder_path,
                baseline_config_hash=configs[0].config_hash if len(
                    configs) > 1 else None
            )
        return all_results

    def _disable_emission_tracker_if_necessary(self):
        """
        The emission tracker does not work when running in parallel.
        Therefore we disable it if we are running in parallel.
        """
        if self.number_of_workers > 1:
            logger.info("Disabling emission tracker due to parallel execution")
            EmissionTrackerManager().disable()
