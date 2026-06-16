import ast
from typing import Optional, get_origin, get_args, Union

import pandas as pd
from pydantic import Field
from typing_extensions import override

from sqa_system.core.data.data_loader.base.data_loader import DataLoader
from sqa_system.core.data.models.dataset.implementations.qa_dataset import QADataset
from sqa_system.core.data.models.qa_pair import QAPair
from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)


class CSVQALoader(DataLoader):
    """
    Implementation of the KGQA data loader that we used for our experiments.
    This loader is a specialized version for the QA dataset that is created with the 
    SQA system. It is not a general purpose data loader for all CSV files.
    """

    @override
    def load(self, dataset_name: str, path: str, limit: Optional[int] = None) -> QADataset:
        try:
            df = pd.read_csv(path)
            logger.debug(f"CSV file '{path}' loaded successfully.")
        except FileNotFoundError:
            logger.error(f"File '{path}' not found.")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV file: {e}")
            raise

        model_fields = QAPair.model_fields
        required_fields = [name for name,
                           field in model_fields.items() if field.is_required()]

        self._check_if_contains_all_required_columns(df, required_fields)

        qa_pairs = {}
        for index, row in df.iterrows():

            # Skip rows with missing required data
            for field in required_fields:
                if pd.isna(row[field]):
                    logger.warning(
                        "Skipping row %d in %s because it contains missing required data.", index, path
                    )
                    continue

            row_data = self._process_row_data(
                df=df,
                model_fields=model_fields,
                index=index,
                row=row
            )

            try:
                if row_data:
                    qa_pair = QAPair(**row_data)
                    qa_pairs[qa_pair.uid] = qa_pair
            except Exception as e:
                logger.warning(
                    "Skipping row %d in %s due to error creating QAPair: %s", index, path, e
                )
                continue

        if limit is not None and limit > 0:
            qa_pairs = dict(list(qa_pairs.items())[:limit])

        logger.info("Loaded %d QA pairs from %s.", len(qa_pairs), path)
        return QADataset(name=dataset_name, data=qa_pairs)

    def _check_if_contains_all_required_columns(self,
                                                df: pd.DataFrame,
                                                required_fields: list[str]):
        """
        Checks if the whole CSV is missing required columns.
        
        Args:
            df (pd.DataFrame): The DataFrame to check.
            required_fields (list[str]): The list of required fields.
            
        Raises:
            ValueError: If any required columns are missing.
        """
        missing_columns = [
            col for col in required_fields if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {', '.join(missing_columns)}")

    def _process_row_data(self,
                          df: pd.DataFrame,
                          model_fields: dict[str, Field],
                          index: int,
                          row: pd.Series) -> dict:
        """
        Processes a single row of data and returns a dictionary of field values.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            model_fields (dict[str, Field]): The model fields to map the data to.
            index (int): The index of the current row.
            row (pd.Series): The current row of data.

        Returns:
            dict: A dictionary containing the processed row data.
        """
        row_data = {}
        for field_name, field in model_fields.items():
            if field_name not in df.columns:
                # If we skip the field the pydantic default will be used
                continue

            value = row[field_name]

            if pd.isna(value):
                continue

            # If the field expects a list, try to parse a string representation into a list
            if self.expects_list(field.annotation):
                if isinstance(value, str):
                    try:
                        parsed_value = ast.literal_eval(value)
                        if not isinstance(parsed_value, list):
                            raise ValueError(
                                f"Value for field '{field_name}' is not a list."
                            )
                        value = parsed_value
                    except (ValueError, SyntaxError) as e:
                        logger.warning(
                            "Skipping row %d due to error parsing field '%s': %s",
                            index, field_name, e
                        )
                        continue
                else:
                    continue
                
            row_data[field_name] = value
        return row_data

    def expects_list(self, expected_type: type) -> bool:
        """
        Return True if expected_type is or contains a list.
        
        Args:
            expected_type (type): The type to check for list compatibility.
            
        Returns:
            bool: True if the expected type is a list or contains a list, False otherwise.
        """
        origin = get_origin(expected_type)
        # Check if it's directly a list.
        if origin == list:
            return True
        # Check if it's a Union (like Optional[list[str]]),
        # then see if one of the types is a list.
        if origin is Union:
            return any(get_origin(arg) == list or arg == list for arg in get_args(expected_type))
        return False