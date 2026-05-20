from typing import get_origin, get_args, Union
import ast
import json
import pandas as pd

from pydantic import Field
from sqa_system.core.data.models.qa_pair import QAPair
from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.qa_generator.question_classifier import QuestionClassifier
from sqa_system.core.language_model.llm_provider import LLMProvider
from sqa_system.core.data.cache_manager import CacheManager
from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)

# The data in the CSVs can be stored differently to the QAPair model.
# Here we map them.
QAPAIR_MAPPING = {
    "question_uid": "uid",
}


class CSVClassificator:
    """
    Class that is responsible for the classification of questions of CSV
    files that are similar to the QAPair dataset structure.

    Args:
        llm_config (dict): The configuration for the LLM model used for the
            classification.
        overwrite_cache (bool): Whether to overwrite the current qa classification
            cache.
    """

    def __init__(self, llm_config, overwrite_cache: bool = True):
        self.llm_adapter = LLMProvider().get_llm_adapter(llm_config)
        self.question_classifier = QuestionClassifier(self.llm_adapter)
        self.cache_manager = CacheManager()
        self.cache_key = "question_classification"
        self.overwrite_cache = overwrite_cache

    def classify(self, dir_path: str):
        """
        Given a path to a directory, this method retrieves all CSV files in the 
        directory and its subdirectories and then classifies the questions in the
        CSV files.

        It overwrites the CSV files with the classified questions.

        Args:
            dir_path (str): The path to the directory.
        """

        fpm = FilePathManager()
        file_paths = fpm.get_files_in_folder(dir_path, ".csv")
        if not file_paths:
            raise ValueError(f"No CSV files found in directory '{dir_path}'.")

        for file_path in file_paths:
            logger.info(f"Classifying questions in '{file_path}'.")
            df = self._load_data(file_path)
            if not self._validate_data(df):
                continue

            qa_pairs = self.try_convert_dataframe_to_qapairs(df)
            if not qa_pairs:
                logger.warning(f"No QAPairs found in '{file_path}'.")
                continue

            classified_qa_pairs = []
            for qa_pair in qa_pairs:
                if not self.overwrite_cache:
                    cached_pair = self.cache_manager.get_data(
                        meta_key=self.cache_key,
                        dict_key=qa_pair.uid
                    )
                    if cached_pair:
                        logger.info(
                            f"Using cached classification for question '{qa_pair.question}'.")
                        updated_pair = QAPair.model_validate_json(cached_pair)
                        classified_qa_pairs.append(updated_pair)
                        continue
                logger.info(f"Classifying question '{qa_pair.question}'.")
                updated_pair = self.question_classifier.classify_qa_pair(
                    qa_pair)
                classified_qa_pairs.append(updated_pair)
                self.cache_manager.add_data(
                    meta_key=self.cache_key,
                    dict_key=qa_pair.uid,
                    value=updated_pair.model_dump_json()
                )

            df = self._merge_qa_pairs_with_dataframe(classified_qa_pairs, df)
            df.to_csv(file_path, index=False)
            logger.info(f"Classified questions in '{file_path}'.")

    def _merge_qa_pairs_with_dataframe(self,
                                       qa_pairs: list[QAPair],
                                       df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds all the fields of the QAPair objects to the DataFrame and
        overwrites existing fields with the same name.

        Args:
            qa_pairs (list[QAPair]): The list of QAPair objects.
            df (pd.DataFrame): The DataFrame to merge with.

        Returns:
            pd.DataFrame: The merged DataFrame.
        """
        for qa_pair in qa_pairs:
            # get the row indices where the uid matches
            matching_rows = df[df["question_uid"] == qa_pair.uid]

            if matching_rows.empty:
                logger.warning(
                    f"Could not find row with uid '{qa_pair.uid}' in DataFrame.")
                continue

            # Overwrite the fields in the DataFrame
            qa_pair_dict = qa_pair.model_dump()
            for field, value in qa_pair_dict.items():
                actual_field = QAPAIR_MAPPING.get(field, field)
                for index in matching_rows.index:
                    df.at[index, actual_field] = str(value)

        return df

    def _load_data(self, path: str) -> pd.DataFrame:
        """
        Loads the data from the CSV file.

        Args:
            path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        try:
            df = pd.read_csv(path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File '{path}' not found: {e}") from e
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV file: {e}") from e

        return df

    def try_convert_dataframe_to_qapairs(self, df: pd.DataFrame) -> list[QAPair]:
        """
        Tries to convert the DataFrame into a list of QAPair objects.

        Args:
            df (pd.DataFrame): The DataFrame to convert.

        Returns:
            list[QAPair]: The list of QAPair objects.
        """
        qa_pairs = []
        for index, row in df.iterrows():
            qa_pair = self._try_convert_to_qa_pair(row)
            if qa_pair:
                qa_pairs.append(qa_pair)
            else:
                logger.warning(f"Failed to convert row {index} to QAPair.")

        return qa_pairs

    def _try_convert_to_qa_pair(self, row: pd.Series) -> QAPair | None:
        """
        Tries to convert the row into a QAPair object by extracting the
        required and optional fields from the row.

        Args:
            row (pd.Series): The row of the DataFrame.

        Returns:
            QAPair | None: The QAPair object or None if conversion failed.
        """
        model_fields = QAPair.model_fields
        required_fields = [name for name,
                           field in model_fields.items() if field.is_required()]
        optional_fields = [
            name for name, field in model_fields.items() if not field.is_required()]

        def get_key_by_value(value):
            for key, mapped_value in QAPAIR_MAPPING.items():
                if mapped_value == value:
                    return key
            return value

        row_data = {}
        for field in required_fields:
            actual_field = get_key_by_value(field)
            row_data[field] = row[actual_field]

        for field in optional_fields:
            actual_field = get_key_by_value(field)
            if actual_field in row:
                row_data[field] = row[actual_field]

        self._parse_string_fields(model_fields, row_data)

        try:
            qa_pair = QAPair.model_validate_json(json.dumps(row_data))
        except Exception as e:
            logger.warning(f"Failed to convert row to QAPair: {e}")
            return None

        return qa_pair

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """
        Checks whether the data is valid by checking whether all required
        fields from the QAPair model are present in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Returns:
            bool: True if the data is valid, False otherwise.
        """
        model_fields = QAPair.model_fields
        required_fields = [name for name,
                           field in model_fields.items() if field.is_required()]

        if "question_uid" not in df.columns:
            logger.warning("Missing required field: question_uid")
            return False

        for field in required_fields:
            is_inside = False
            if field in QAPAIR_MAPPING:
                actual_field = QAPAIR_MAPPING[field]
                if actual_field in df.columns:
                    is_inside = True
            elif field in df.columns:
                is_inside = True

            if not is_inside:
                logger.warning(f"Missing required field: {field}")
                return False

        logger.debug("All required fields are present.")

        return True

    def _parse_string_fields(self,
                             model_fields: dict[str, Field],
                             row_data: dict) -> dict:
        """
        Because we are loading csv files, we need to parse the fields to have the expected
        type of the qa pair model.

        Args:
            model_fields (dict[str, Field]): The model fields of the QAPair model.
            row_data (dict): The row data to parse.

        Returns:
            dict: The parsed row data.
        """
        for field, model_field in model_fields.items():
            if field in row_data and pd.isna(row_data[field]):
                row_data[field] = None

            annotation = model_field.annotation
            origin = get_origin(model_field.annotation)
            args = get_args(annotation)

            # Check if the field is a list
            is_list_field = False
            if origin is Union:
                for arg in args:
                    if arg == list or get_origin(arg) is list:
                        is_list_field = True
                        break
            elif origin is list:
                is_list_field = True

            if is_list_field:
                self._handle_lists(field, row_data)

            # Check if the field is an int
            is_int_field = False
            if origin is Union:
                for arg in args:
                    if arg == int:
                        is_int_field = True
                        break
            elif annotation == int:
                is_int_field = True

            if is_int_field and field in row_data:
                self._handle_int(field, row_data)

        return row_data

    def _handle_lists(self, field: str, row_data: dict):
        """
        Helper function to handle the parsing of list fields in the
        DataFrame. It checks if the field is a list and parses it
        accordingly. 

        Args:
            field (str): The field name.
            row_data (dict): The row data to parse.
        """
        if field in row_data:
            value = row_data[field]
            if isinstance(value, float) and pd.isna(value):
                row_data[field] = []
            elif isinstance(value, str):
                try:
                    row_data[field] = ast.literal_eval(value)
                except (ValueError, SyntaxError) as e:
                    logger.warning(
                        f"Error parsing field '{field}' as list: {e}")
                    row_data[field] = []

    def _handle_int(self, field: str, row_data: dict):
        """
        Helper function to handle the parsing of int fields in the
        DataFrame. It checks if the field is an int and parses it
        accordingly.

        Args:
            field (str): The field name.
            row_data (dict): The row data to parse.

        Raises:
            ValueError: If the field cannot be parsed as an int.
        """
        value = row_data[field]
        if pd.isna(value):
            row_data[field] = 0
            return
        try:
            row_data[field] = int(value)
        except ValueError as e:
            logger.warning(f"Error parsing field '{field}' as int: {e}")
