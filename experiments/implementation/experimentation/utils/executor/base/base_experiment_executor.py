import os
from abc import ABC, abstractmethod
import logging
import json
import asyncio
from typing import List, Dict, Any, Optional

import weave
import pandas as pd
from pydantic import BaseModel, Field

from implementation.experimentation.utils.visualizer.experiment_visualizer import (
    ExperimentVisualizer,
    ExperimentVisualizerSettings,
    PlotType
)
from implementation.config.config_models.experiment_config import ExperimentConfig
from implementation.file_path_management import FilePathManager
from implementation.experimentation.evaluation.base.evaluator import Evaluator
from implementation.experimentation.utils.experiment_result_processor import ExperimentResultProcessor
from core.progress.progress_handler import ProgressHandler
from implementation.config.config_models.pipeline_config import PipelineConfig
from implementation.pipeline.tracking.pipeline_data_collector import PipelineData, PipelineDataCollector
from implementation.pipeline.factory.pipeline_factory import PipelineFactory
from implementation.core.logging import get_logger
logger = get_logger(__name__)


class ExperimentExecutorSettings(BaseModel):
    """
    Settings for the BaseExperimentExecutor.
    """
    experiment_config: ExperimentConfig = Field(
        ..., description="The configuration object for the experiment.")
    weave_project_name: Optional[str] = Field(
        default=None, description="The name of the Weave project to use for tracking.")
    skip_weave: bool = Field(
        default=False, description="If True, weave.init() is skipped and no W&B tracking is performed.")


class BaseExperimentExecutor(ABC):
    """
    This class is used to the execution of a experiment. It is intended to be
    subclassed by strategies that define how the experiment should be executed.

    Args:
        dataset (List[Dict]): The dataset to use for the experiment including the
            QAPairs.
        evaluators (List[Evaluator]): The list of evaluators to shuld be used to 
            compute the metrics.
        result_processor (ExperimentResultProcessor): The result processor which 
            is responsible for post processing the results.
        executor_settings (ExperimentExecutorSettings): The settings for the
            experiment executor. This includes the configuration for the experiment
            and the name of the Weave project to use for tracking.            
    """

    def __init__(self,
                 dataset: List[Dict],
                 evaluators: List[Evaluator],
                 result_processor: ExperimentResultProcessor,
                 executor_settings: ExperimentExecutorSettings,
                 ):
        self.experiment_config = executor_settings.experiment_config
        self.weave_project_name = executor_settings.weave_project_name or "kastel-sdq-meta-research/todo"
        self.skip_weave = executor_settings.skip_weave
        self.prepared_qa_dataset = dataset
        self.prepared_qa_dataset_df = pd.DataFrame(self.prepared_qa_dataset)
        self.evaluators = evaluators
        self.result_processor = result_processor
        self.progress_handler = ProgressHandler()
        self.file_path_manager = FilePathManager()
        if not self.skip_weave:
            self._prepare_weave()

    def _prepare_weave(self):
        # Here we initialize the weave experiment which tracks the run of the experiment
        # on the weights and biases dashboard: https://wandb.ai/
        weave.init(self.weave_project_name)
        # We disable weave printing call links to avoid cluttering the console
        os.environ["WEAVE_PRINT_CALL_LINK"] = "false"
        # This disables weave spamming the console with connection warnings
        logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

    @abstractmethod
    def execute_experiments(self,
                            results_folder_path: str,
                            configs: List[PipelineConfig],
                            visualize_results: bool) -> List[Dict[str, Any]]:
        """
        Execute experiments using a specific strategy. This should be implemented 
        by a subclass to realize the execution.

        Args:
            results_folder_path (str): The path where the results should be saved
            configs: List of pipeline configurations to run
            visualize_results (bool): Whether to visualize the results in diagrams.

        Returns:
            List[dict]: The results of all experiments
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _run_pipeline(self,
                      config: PipelineConfig) -> List[PipelineData]:
        """
        A helper function that simply runs for each question pipelines with the given
        configuration and returns the results without evaluating them.

        Args:
            config (PipelineConfig): The configuration to use for the pipeline.

        Returns:
            List[PipelineData]: The results of the pipeline runs.        
        """
        pipeline = PipelineFactory().create(config)
        all_results: List[PipelineData] = []

        self.progress_handler.add_task(
            string_id="asking_questions",
            description="Asking Questions...",
            total=len(self.prepared_qa_dataset),
            reset=True
        )

        for entry in self.prepared_qa_dataset:
            result = pipeline.run(
                input_str=entry["question"],
                topic_entity_id=entry["topic_entity_id"],
                topic_entity_value=entry["topic_entity_value"]
            )
            all_results.append(result)

        return all_results

    def _run_with_weave_evaluation(self, config: PipelineConfig) -> List[Dict]:
        """
        Runs the experiment configuration with the Evaluation class of weave.
        This allows tracking in the weights and biases dashboard and also directly
        evaluates the results of the pipeline.

        Args:
            config (PipelineConfig): The configuration to use for the pipeline.

        Returns:
            List[Dict]: The results of the pipeline runs which are already evaluated.        
        """
        pipeline = PipelineFactory().create(config)

        # Unfortunately weave doesn't currently allow to return the generated answers
        # programmatically. The answers are only visible on the weaver website.
        # Therefore we add this collector to later construct the answer dataframe
        answer_collector = PipelineDataCollector()
        pipeline.answer_collector = answer_collector

        try:
            self.progress_handler.add_task(
                string_id="asking_questions",
                description="Asking Questions...",
                total=len(self.prepared_qa_dataset),
                reset=True
            )

            evaluation = weave.Evaluation(dataset=self.prepared_qa_dataset,
                                          scorers=self.evaluators)
            result = asyncio.run(evaluation.evaluate(pipeline))

            logger.debug("Results for config %s: %s",
                         config.config_hash, result)

            return self.result_processor.process_result_rows(
                answer_collector.get_all_entries(),
                self.prepared_qa_dataset,
                config,
                result
            )
        except Exception as e:
            logger.error("Failed to evaluate pipeline: %s", e)
            raise

    def _save_results(self,
                      data_to_save: pd.DataFrame,
                      results_folder_path: str,
                      config: PipelineConfig):
        """
        Helper function to save the results of the experiment.

        Args:
            data_to_save (pd.DataFrame): The data to save as a csv file.
            results_folder_path (str): The path where the results should be saved.
            config (PipelineConfig): The configuration that was used to run the experiment.        
        """
        self.file_path_manager.ensure_dir_exists(results_folder_path)

        predictions_folder = self.file_path_manager.combine_paths(
            results_folder_path,
            "predictions"
        )
        self.file_path_manager.ensure_dir_exists(predictions_folder)
        name = config.config_hash
        data_to_save.to_csv(self.file_path_manager.combine_paths(
            predictions_folder, name + ".csv"))

        # Now we save the configuration that was used in the experiment into
        # a separate folder as a json. This helps to keep track of the configuration
        # that was used in the experiment and can be used to reproduce the results.
        self._save_configs(results_folder_path, [config])

    def _visualize_results(self,
                           results_folder_path: str,
                           baseline_config_hash: str = None):
        """
        Helper function to execute the generation of visualization of the results.

        Args:
            results_folder_path (str): The path where the results of the visualizations
                should be saved.
            baseline_config_hash (str): The hash of the baseline configuration to use
                for the visualizations. This is used generate plots where the baseline
                is highlighted
        """
        visualization_path = self.file_path_manager.combine_paths(
            results_folder_path,
            "visualizations")
        self.file_path_manager.ensure_dir_exists(visualization_path)

        visualizer = ExperimentVisualizer(
            experiment_settings=ExperimentVisualizerSettings(
                data_folder_path=results_folder_path,
                save_folder_path=visualization_path,
                baseline_config=baseline_config_hash,
                should_print=False,
                should_save_to_file=True
            )
        )
        visualizer.run(
            plots_to_generate=[
                PlotType.AVERAGE_METRICS_PER_CONFIG,
                PlotType.TABLE,
                PlotType.METRIC_DISTRIBUTIONS
            ]
        )

    def _save_configs(self, results_path: str, configs: List[PipelineConfig]):
        """
        Saves the configuration JSON files to the results folder.
        This allows to track the exact configuration used in each
        experiment.
        
        Args:
            results_path (str): The path where the results should be saved.
            configs (List[PipelineConfig]): The list of configurations to save.
        """
        for used_config in configs:
            config_path = self.file_path_manager.combine_paths(
                results_path,
                "configs",
                f"config_{used_config.config_hash}.json")
            self.file_path_manager.ensure_dir_exists(config_path)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(used_config.to_dict(), f, indent=4)
