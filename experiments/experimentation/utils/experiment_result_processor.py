from typing import List, Dict, Any, Optional
import pandas as pd

from sqa_system.core.data.models import PipeIOData
from sqa_system.core.config.models.pipeline_config import PipelineConfig
from sqa_system.core.data.pipeline_data_collector import PipelineData
from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)


class ExperimentResultProcessor:
    """
    Class that post-processes the results of an experiment run.
    """

    def create_result_dataframe(self,
                                row_results: List[dict],
                                questions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates the final result DataFrame from the row results and the 
        initial questions DataFrame.

        Args:
            row_results (List[dict]): The list of dictionaries containing the results.
            questions_df (pd.DataFrame): The DataFrame containing the initial questions
                and metadata about them.
        """
        results_df = pd.DataFrame(row_results)

        if results_df.empty:
            logger.warning("No results were generated.")
            return results_df

        duplicate_cols = set(results_df.columns) & set(
            questions_df.columns) - {"uid"}
        if duplicate_cols:
            results_df.drop(columns=list(duplicate_cols), inplace=True)

        final_df = pd.merge(
            results_df,
            questions_df,
            on="uid",
            how="left"
        )

        # Reordering columns
        cols = final_df.columns.tolist()
        front_cols = ["config_hash", "uid"]
        other_cols = [col for col in cols if col not in front_cols]
        final_df = final_df[front_cols + other_cols]
        return final_df

    def process_result_rows(self,
                            pipeline_datas: List[PipelineData],
                            prepared_dataset: List[dict],
                            config: PipelineConfig,
                            evaluation_data: Optional[dict] = None) -> List[dict]:
        """
        This is the main method that is responsible for the creation of the prediction results
        as they will later be stored as the prediction.csv file.
        It takes the pipeline data and the prepared dataset and processes them to create
        the final result rows.

        Args:
            pipeline_datas (List[PipelineData]): The list of pipeline data objects
                containing the results of the experiment.
            prepared_dataset (List[dict]): The prepared dataset as a list of dictionaries.
            config (PipelineConfig): The configuration of the pipeline.
            evaluation_data (Optional[dict]): The evaluation data to be added to the results.

        Returns:
            List[dict]: The list of dictionaries containing the prepared results.
        """
        all_results = []
        for index, row in enumerate(prepared_dataset):
            question_match = None
            for data in pipeline_datas:
                if data.pipe_io_data.question_id == row["uid"]:
                    pipeline_data = data
                    break
                if data.pipe_io_data.initial_question == row["question"]:
                    question_match = data

            if not pipeline_data:
                if not question_match:
                    raise ValueError(
                        f"Could not find matching question for {row['uid']}")
                pipeline_data = question_match

            pipe_io_data = pipeline_data.pipe_io_data
            row["config_hash"] = config.config_hash
            row["question"] = pipe_io_data.initial_question
            # Process answer
            if pipe_io_data.generated_answer:
                row["generated_answer"] = pipe_io_data.generated_answer
            else:
                row["generated_answer"] = "None"
            # Process context
            self._process_context(row, pipe_io_data)
            row["topic_entity_id"] = pipe_io_data.topic_entity_id
            row["topic_entity_value"] = pipe_io_data.topic_entity_value

            # Add evaluation results
            self._add_evaluation_results_to_row(
                row, index, evaluation_data
            )

            # Append the tracking data
            row["runtime"] = pipeline_data.runtime
            row["llm_cost"] = pipeline_data.llm_stats.cost
            row["llm_tokens"] = pipeline_data.llm_stats.total_tokens
            row["cpu_count"] = pipeline_data.emissions_data.cpu_count
            row["cpu_energy_consumption"] = pipeline_data.emissions_data.cpu_energy_consumption
            row["cpu_model"] = pipeline_data.emissions_data.cpu_model
            row["gpu_count"] = pipeline_data.emissions_data.gpu_count
            row["gpu_energy_consumption"] = pipeline_data.emissions_data.gpu_energy_consumption
            row["gpu_model"] = pipeline_data.emissions_data.gpu_model
            row["os"] = pipeline_data.emissions_data.os
            row["ram_energy_consumption"] = pipeline_data.emissions_data.ram_energy_consumption
            row["total_energy_consumption"] = pipeline_data.emissions_data.total_energy_consumption
            row["timestamp"] = pipeline_data.emissions_data.timestamp
            row["tracking_duration"] = pipeline_data.emissions_data.tracking_duration
            row["emissions"] = pipeline_data.emissions_data.emissions
            row["weave_url"] = pipeline_data.weave_url

            # Append the config
            for index, pipe_config in enumerate(config.pipes):
                row.update(self._flatten_dict(
                    pipe_config.model_dump(), f"pipe_{index}"))

            # Append the row to the results list
            all_results.append(row)
            index += 1
        return all_results

    def _add_evaluation_results_to_row(self,
                                       row: dict,
                                       index: int,
                                       evaluation_data: Optional[dict] = None):
        """Helper function to add evaluation results to the row."""
        if evaluation_data is None:
            evaluation_data = {}
        for _, value in evaluation_data.items():
            if (isinstance(value, dict) and
                    "row_scores" in value and
                    value["row_scores"] is not None):
                metric_dict = value["row_scores"][index]
                for key, score in metric_dict.items():
                    if isinstance(score, float):
                        row[f"{key}"] = round(score, 3)
                    else:
                        row[f"{key}"] = round(float(score), 3)

    def _process_context(self,
                         row: dict,
                         pipe_io_data: PipeIOData):
        """
        Processes the context retrieved from the PipeIOData and updates the row with it.

        Args:
            row (dict): The row to be updated with the context.
            pipe_io_data (PipeIOData): The PipeIOData object containing the context.
        """
        if pipe_io_data.retrieved_context:
            context_text = []
            for context in pipe_io_data.retrieved_context:
                if context.text:
                    context_text.append(context.text)
            row["retrieved_context"] = context_text

    def _flatten_dict(self, dictionary: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
        """
        Helper function to flatten a nested dictionary.

        Args:
            dictionary (Dict[str, Any]): The dictionary to be flattened.
            parent_key (str): The base key to use for the flattened keys.

        Returns:
            Dict[str, Any]: The flattened dictionary.
        """
        items: Dict[str, Any] = {}
        for k, v in dictionary.items():
            new_key = f"{parent_key}{'_'}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key))
            else:
                items[new_key] = v
        return items
