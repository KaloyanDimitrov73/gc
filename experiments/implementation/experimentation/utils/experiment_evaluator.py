import os
import re
import ast
import asyncio
from typing import List, Dict, Optional
import weave
from weave import Model
import pandas as pd

from implementation.data_management import FilePathManager
from implementation.experimentation.evaluation.base.evaluator import Evaluator
from implementation.core import ProgressHandler
from implementation.core import ContextType, Context, QAPair
from implementation.models.pipe_io_data import PipeIOData
from implementation.core.logging import get_logger
logger = get_logger(__name__)


class DummyModel(Model):
    """
    This implementation is a workaround because the weave implementation
    is intended to sequentially evaluate the results right after a single
    pipeline call. 

    This workaround intends to get around this limitation
    by allowing the evaluation to run after all pipeline calls have been made.
    This works by giving the weave evaluation function a dummy 'model' that
    returns the pipeline results.
    """
    data: List[PipeIOData]
    model_name: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prepared_data = {}
        for entry in self.data:
            self._prepared_data[entry.initial_question] = entry

        # set the name of the class to the model name
        if self.model_name:
            self.__class__.__name__ = self.model_name

    @weave.op()
    def predict(self, question: str) -> dict:
        """Simple predict method that returns the data based on the question."""
        return self._prepared_data[question].model_dump()


class ExperimentEvaluator:
    """
    This class evaluates the results of pipeline experiments using defined evaluators
    and integrates with the Weave framework for displaying the results in the 
    Weight&Bias dashboard.
    """

    def __init__(self):
        self.progress_handler = ProgressHandler()
        self.file_path_manager = FilePathManager()

    async def evaluate_results_with_weave(self,
                                           qa_dataset_dict: List[Dict],
                                           evaluators: List[Evaluator],
                                           pipeline_results: List[PipeIOData],
                                           model_name) -> List[Dict]:
        """
        Evaluates the output of pipeline calls using weave which also enables
        tracking those results in the Weight&Bias dashboard.

        Args:
            qa_dataset_dict (List[Dict]): The QA dataset as a dictionary.
            evaluators (List[Evaluator]): The list of evaluators to use for evaluation.
            pipeline_results (List[PipeIOData]): The results of the pipeline calls
            model_name: The name of the model being evaluated
        """
        required_keys = ["golden_answer",
                         "golden_triples", "golden_doc_chunks"]
        for entry in qa_dataset_dict:
            for key in required_keys:
                if key not in entry:
                    raise ValueError(
                        f"QA dataset entry missing required key: {key}")

        logger.info("Starting Evaluation for model %s", model_name)

        # Because we encountered issues with parallel evaluation we
        # disable it here.
        os.environ["WEAVE_PARALLELISM"] = "1"
        evaluation = weave.Evaluation(dataset=qa_dataset_dict,
                                      scorers=evaluators)

        self.progress_handler.add_task(
            string_id="evaluating_results",
            description="Evaluating Results...",
            total=len(qa_dataset_dict),
            reset=False
        )

        result = await evaluation.evaluate(
            DummyModel(
                data=pipeline_results,
                model_name=model_name
            )
        )

        return result

    def evaluate_pipeline_results_in_folder(self,
                                            data_folder_path: str,
                                            evaluators: List[Evaluator],
                                            save_folder_path: Optional[str] = None):
        """
        This function collects all pipeline results that are saved as 'csv' files
        in the given directory and its subdirectories and evaluates them using the
        given evaluators.
        It then writes the results to the given save directory.

        Args:
            data_folder_path (str): The path to the folder containing the pipeline results.
            evaluators (List[Evaluator]): The list of evaluators to use for evaluation.
            save_folder_path (Optional[str]): The path to the folder where the evaluation results 
                should be saved. If not provided, the original files will be overwritten.

        """
        if not os.path.exists(data_folder_path):
            raise ValueError(
                f"Data folder path {data_folder_path} does not exist.")
        if save_folder_path is not None and not os.path.exists(save_folder_path):
            self.file_path_manager.ensure_dir_exists(save_folder_path)

        csv_files = self.file_path_manager.get_files_in_folder(
            folder_path=data_folder_path,
            file_type="csv"
        )

        for csv_file in csv_files:
            self.run_evaluation_for_csv_file(csv_file_path=csv_file,
                                             evaluators=evaluators,
                                             save_folder_path=save_folder_path)

    def run_evaluation_for_csv_file(self,
                                    csv_file_path: str,
                                    evaluators: List[Evaluator],
                                    save_folder_path: Optional[str] = None):
        """
        Retrieves the pipeline results from the given CSV file and evaluates them 
        using the given evaluators.
        It then writes the results to the given save directory or overwrites the original file.

        Args:
            csv_file_path (str): The path to the CSV file containing the pipeline results.
            evaluators (List[Evaluator]): The list of evaluators to use for evaluation.
            save_folder_path (Optional[str]): The path to the folder where the evaluation 
                results should be saved. If not provided, the original file will be overwritten.
        """
        grouped_data = self._load_data_from_csv_file(csv_file_path)
        evaluated_data = []
        for _, data in grouped_data.items():
            evaluation_data = self._get_required_evaluation_data(data)
            evaluation_results = asyncio.run(
                self.evaluate_results_with_weave(
                    qa_dataset_dict=evaluation_data["qa_dataset_dict"],
                    evaluators=evaluators,
                    pipeline_results=evaluation_data["pipeline_data_list"],
                    model_name=evaluation_data["model_name"]
                )
            )

            self._add_evaluation_to_dict(data, evaluation_results)
            evaluated_data.extend(data)

        # Save the evaluation results
        result_df = pd.DataFrame(evaluated_data)
        if not save_folder_path:
            result_df.to_csv(
                csv_file_path,
                index=False
            )
        else:
            result_df.to_csv(
                os.path.join(save_folder_path,
                             os.path.basename(csv_file_path)),
                index=False
            )

    def _add_evaluation_to_dict(self,
                                data: List[dict],
                                evaluation_results: dict):
        """
        Adds the evaluation results to the original data.

        Args:
            data (List[dict]): The original data as a list of dictionaries.
            evaluation_results (dict): The evaluation results as a dictionary.
        """
        for index, row in enumerate(data):
            for _, value in evaluation_results.items():
                if (isinstance(value, dict) and
                        "row_scores" in value and
                        value["row_scores"] is not None):
                    metric_dict = value["row_scores"][index]
                    for key, score in metric_dict.items():
                        if isinstance(score, float):
                            row[f"{key}"] = round(score, 3)
                        else:
                            row[f"{key}"] = round(float(score), 3)

    def _load_data_from_csv_file(self, csv_file_path: str) -> dict:
        """
        Loads the data from the given CSV file and converts it to a dictionary.
        It also handles the conversion of strings to lists and NaN values.

        Args:
            csv_file_path (str): The path to the CSV file to load.

        Returns:
            dict: The loaded data as a dictionary.
        """
        if not self.file_path_manager.file_path_exists(csv_file_path):
            raise ValueError(f"File {csv_file_path} does not exist.")

        df = pd.read_csv(csv_file_path)
        data_dict = df.to_dict(orient="records")

        # Convert strings to lists and handle NaN values
        updated_data_dict = []
        for entry in data_dict:
            updated_entry = {}
            for key, value in entry.items():
                updated_entry[key] = self._try_convert_to_list(value)
                if isinstance(updated_entry[key], float) and pd.isna(updated_entry[key]):
                    updated_entry[key] = None

            updated_data_dict.append(updated_entry)

        # group by config_hash
        grouped_data = {}
        for entry in updated_data_dict:
            config_hash = entry["config_hash"]
            if config_hash not in grouped_data:
                grouped_data[config_hash] = []
            grouped_data[config_hash].append(entry)
        return grouped_data

    def _get_required_evaluation_data(self, data_dict: dict):
        """
        Prepares a dictionary with the required data for evaluation.

        Args:
            data_dict (dict): The data dictionary containing the pipeline results.
        """
        try:
            model_name = None
            qa_dataset_dict_list = []
            pipeline_data_list = []
            context_type = None
            for entry in data_dict:
                qa_dataset_dict = {}
                if not model_name:
                    for key in entry.keys():
                        if "retriever_type" in key:
                            model_name = entry[key]
                            break

                retrieved_context = entry["retrieved_context"]

                converted_context_list = []
                for context in retrieved_context:
                    if not context_type:
                        context_type = self._get_context_type(context)
                    converted_context = Context(
                        text=context,
                        context_type=context_type
                    )
                    converted_context_list.append(converted_context)

                pipeline_data_list.append(PipeIOData(
                    initial_question=entry["question"],
                    retrieval_question=entry["question"],
                    retrieved_context=converted_context_list,
                    generated_answer=entry["generated_answer"],
                    topic_entity_id=entry["topic_entity_id"],
                    topic_entity_value=entry["topic_entity_value"]
                ))
                qa_pair = QAPair.model_validate(entry)
                qa_dataset_dict.update(qa_pair.model_dump())
                qa_dataset_dict_list.append(qa_dataset_dict)

            return {
                "model_name": model_name,
                "qa_dataset_dict": qa_dataset_dict_list,
                "pipeline_data_list": pipeline_data_list
            }

        except Exception as e:
            raise ValueError(f"Error processing evaluation data: {e}") from e

    def _try_convert_to_list(self, string: str) -> list | str:
        """
        Helper function to convert a string to a list or other data type.

        Args:
            string (str): The string to convert.

        Returns:
            list | str: The converted value, either a list or the original string.
        """
        try:
            return ast.literal_eval(string)
        except (ValueError, SyntaxError):
            return string

    def _get_context_type(self, context_string: str) -> ContextType:
        """
        Returns the type of the context from a given string. Checks whether the 
        string is a triple or a document.

        Args:
            context_string (str): The context string to check.

        Returns:
            ContextType: The context type, either KG or DOC.
        """
        if re.match(r"^\(.*\)$", context_string):
            return ContextType.KG
        return ContextType.DOC
