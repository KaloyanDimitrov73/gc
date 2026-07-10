import os
import json
from typing import List, Optional
from ast import literal_eval
import pandas as pd

from implementation.experimentation.evaluation.factory.evaluator_factory import EvaluatorFactory
from implementation.experimentation.evaluation.base.evaluator import Evaluator
from implementation.file_path_management import FilePathManager
from implementation.data_loader.csv_qa_loader import CSVQALoader
from implementation.config.config_models.evaluator_config import EvaluatorConfig
from implementation.core import QAPair


class FileEvaluator:
    """
    Class that allows to evaluate prediction.csv files that are generated from experiments.
    It loads the files, evaluates them and saves the results back to the files.
    """

    def __init__(self):
        # Disable weave logging
        os.environ["WEAVE_DISABLED"] = "1"

    def update_prediction_files_in_folder(self,
                                          folder_path: str,
                                          evaluator_config_path: str,
                                          updated_qa_dataset_path: Optional[str] = None,
                                          white_list: Optional[List[str]] = None):
        """
        Method that updates all prediction files in a folder and its subfolders. It does so
        by loading the files, updating the ground truth values if necessary and recalculating
        all metrics in the files based on the given evaluator configs.

        Args:
            folder_path (str): The path to the folder containing the prediction files.
                All subfolder will be checked for prediction files.
            evaluator_config_path (str): The path to the configurations of the evaluators 
                that should be used for the evaluation.
            updated_qa_dataset_path (Optional[str]): Optionally, a path to a QA dataset can
                be provided. This will be used to update the ground truth values in the
                prediction files. 
            white_list (Optional[List[str]]): Optionally, a list of file names can be provided.
                Only those files will be updated. 
        """
        evaluator_configs = self._load_evaluators_from_path(
            evaluator_config_path)
        qa_pairs = None
        if updated_qa_dataset_path:
            qa_pairs = CSVQALoader().load("", updated_qa_dataset_path).get_all_entries()

        file_path_manager = FilePathManager()
        csv_file_paths = file_path_manager.get_files_in_folder(
            folder_path, file_type="csv")

        experiment_files = []
        for file_path in csv_file_paths:
            if white_list and not any(file_path.endswith(wl) for wl in white_list):
                continue
            print(f"Starting to update file: {file_path}")
            try:
                experiment_file = self.load_prediction_file(file_path)
                experiment_files.append(experiment_file)
            except ValueError as e:
                print(f"Error loading file {file_path}: {e}. "
                      "Maybe not a valid experiment file?")
                continue

            if updated_qa_dataset_path:
                experiment_file = self.update_prediction_file(
                    qa_pairs, experiment_file, evaluator_configs)
            else:
                experiment_file = self.recalculate_metrics(
                    experiment_file, evaluator_configs)

            experiment_file.to_csv(file_path, index=False)
            print(f"Updated prediction file: {file_path}")

    def update_prediction_file(self,
                               qa_pairs: List[QAPair],
                               prediction_file: pd.DataFrame,
                               evaluator_configs: List[EvaluatorConfig]) -> pd.DataFrame:
        """
        Updates the ground truth values in the prediction file.
        As this makes the evaluation metrics invalid, it recalculates the metrics
        from the file.

        Args:
            qa_pairs (List[QAPair]): The list of QAPairs with updates data 
                to be used for the evaluation.
            prediction_file (pd.DataFrame): The prediction file to be updated.
            evaluators_configs (List[EvaluatorConfig]): The list of evaluators to be used
                for the evaluation.

        Returns:
            pd.DataFrame: The updated prediction file with recalculated metrics.
        """
        prediction_file = prediction_file.copy()
        self._update_ground_truth_of_prediction_file(qa_pairs, prediction_file)
        prediction_file = self.recalculate_metrics(
            prediction_file, evaluator_configs)
        return prediction_file

    def _update_ground_truth_of_prediction_file(self,
                                                qa_pairs: List[QAPair],
                                                prediction_file: pd.DataFrame):
        """
        Updates the ground truth values in the prediction file.

        Args:
            qa_pairs (List[QAPair]): The list of QAPairs with updates data
                to be used for the evaluation.
            prediction_file (pd.DataFrame): The prediction file to be updated.
        """
        for qa_pair in qa_pairs:
            for index, row in prediction_file.iterrows():
                if row["uid"] == qa_pair.uid:
                    prediction_file.at[index,
                                       "golden_answer"] = qa_pair.golden_answer
                    prediction_file.at[index,
                                       "golden_triples"] = qa_pair.golden_triples
                    prediction_file.at[index,
                                       "golden_doc_chunks"] = qa_pair.golden_doc_chunks
                    prediction_file.at[index, "hops"] = qa_pair.hops
                    prediction_file.at[index,
                                       "source_ids"] = qa_pair.source_ids
                    prediction_file.at[index,
                                       "topic_entity_id"] = qa_pair.topic_entity_id
                    prediction_file.at[index,
                                       "topic_entity_value"] = qa_pair.topic_entity_value

    def recalculate_metrics(self,
                            prediction_file: pd.DataFrame,
                            evaluators_configs: List[EvaluatorConfig]) -> pd.DataFrame:
        """
        Recalculates the metrics in the prediction file given the evaluators.

        Args:
            prediction_file (pd.DataFrame): The prediction file to be updated.
            evaluator_configs (List[EvaluatorConfig]): The list of evaluators to be used
                for the evaluation.

        Returns:
            pd.DataFrame: The updated prediction file with recalculated metrics.
        """
        prediction_file = prediction_file.copy()
        updated_scores_by_question_id = {}
        for index, (_, row) in enumerate(prediction_file.iterrows()):
            print(f"Evaluating row {index + 1} of {len(prediction_file)}")

            model_output = {
                "retrieved_context": self._convert_to_list(row["retrieved_context"]),
                "initial_question": row["question"],
                "generated_answer": row["generated_answer"],
            }

            evaluation_results = {}
            # Add evaluation scores
            for evaluator in self._load_evaluators_from_configs(evaluators_configs):
                score = evaluator.score(
                    output=model_output,
                    golden_answer=row["golden_answer"],
                    golden_triples=self._convert_to_list(row["golden_triples"]),
                    golden_doc_chunks=self._convert_to_list(row["golden_doc_chunks"]))
                evaluation_results.update(score)

            updated_scores_by_question_id[row["uid"]] = evaluation_results

        # apply directly to dataframe
        for uid, scores in updated_scores_by_question_id.items():
            for metric, value in scores.items():
                prediction_file.loc[prediction_file["uid"]
                                    == uid, metric] = value

        return prediction_file

    def _load_evaluators_from_configs(self,
                                      evaluator_configs: List[EvaluatorConfig]) -> List[Evaluator]:
        """
        Returns all evaluators registered in the EvaluatorFactory.

        Args:
            evaluator_configs (List[EvaluatorConfig]): The list of evaluator configs.

        Returns:
            List[Evaluator]: The list of evaluators loaded from the configs.
        """
        evaluators = []
        for evaluator_config in evaluator_configs:
            evaluator = EvaluatorFactory.create(evaluator_config)
            evaluators.append(evaluator)
        return evaluators

    def load_prediction_files_from_folder(self,
                                          folder_path: str) -> List[pd.DataFrame]:
        """
        Loads all prediction files from a folder.

        Args:
            folder_path (str): The path to the folder containing the 
                prediction files.

        Returns:
            List[pd.DataFrame]: A list of loaded prediction files as
                pandas DataFrames.
        """
        file_path_manager = FilePathManager()
        csv_file_paths = file_path_manager.get_files_in_folder(
            folder_path, file_type="csv")

        experiment_files = []
        for file_path in csv_file_paths:
            try:
                experiment_file = self.load_prediction_file(file_path)
                experiment_files.append(experiment_file)
            except ValueError as e:
                print(f"Error loading file {file_path}: {e}. "
                      "Maybe not a valid experiment file?")
        return experiment_files

    def load_prediction_file(self, file_path: str) -> pd.DataFrame:
        """
        Tries to load a prediction csv file from a path.

        Args:
            file_path (str): The path to the prediction file.

        Returns:
            pd.DataFrame: The loaded prediction file as a pandas DataFrame.
        """
        if not file_path.endswith(".csv"):
            raise ValueError(
                "The prediction file must be a CSV file.")
        if not os.path.exists(file_path):
            raise ValueError(
                f"The prediction file does not exist: {file_path}")

        experiment_file = pd.read_csv(file_path)
        experiment_file = experiment_file.where(
            pd.notnull(experiment_file), None)

        # verify the columns
        if "retrieved_context" not in experiment_file.columns:
            raise ValueError(
                "The prediction file must contain a 'context' column.")
        if "generated_answer" not in experiment_file.columns:
            raise ValueError(
                "The prediction file must contain a 'generated_answer' column.")
        if "golden_answer" not in experiment_file.columns:
            raise ValueError(
                "The prediction file must contain a 'golden_answer' column.")
        if "golden_triples" not in experiment_file.columns:
            raise ValueError(
                "The prediction file must contain a 'golden_triples' column.")

        return experiment_file

    def _load_evaluators_from_path(self,
                                   evaluator_configs_path: str) -> List[EvaluatorConfig]:
        """
        Loads the evaluator configs from a path.

        Args:
            evaluator_configs_path (str): The path to the evaluator configs.

        Returns:
            List[EvaluatorConfig]: The loaded evaluator configs.
        """
        if not os.path.exists(evaluator_configs_path):
            raise ValueError(
                f"The evaluator config file does not exist: {evaluator_configs_path}")

        with open(evaluator_configs_path, "r", encoding="utf-8") as f:
            evaluator_configs = json.load(f)

        return [EvaluatorConfig.from_dict(config) for config in evaluator_configs]

    def _convert_to_list(self, value):
        if isinstance(value, list):
            return value
        if value is None or pd.isna(value):
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                return literal_eval(value)
            except (ValueError, SyntaxError):
                return [value]
        return [value]
