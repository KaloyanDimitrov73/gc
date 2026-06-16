from enum import Enum
from typing import Dict
from ..implementations.json_publication_loader import JsonPublicationLoader
from ..implementations.csv_qa_loader import CSVQALoader
from ..base.data_loader import DataLoader


class DataLoaderType(Enum):
    """The type of the data loader."""
    PUBLICATION = "publication"
    QUESTION_ANSWERING = "question_answering"


class DataLoaderFactory:
    """A factory class that returns an instance of the specified data loader."""

    @staticmethod
    def get_data_loader(data_loader_name: str) -> DataLoader:
        """
        Retrieves a data loader instance based on the provided data loader name.

        Args:
            data_loader_name (str): The name of the data loader to retrieve.

        Returns:
            DataLoader: An instance of the specified data loader.

        Raises:
            ValueError: If the provided data loader name is invalid.
        """
        if data_loader_name == "JsonPublicationLoader":
            return JsonPublicationLoader()
        if data_loader_name == "CSVQALoader":
            return CSVQALoader()

        raise ValueError(f"Invalid data loader name: {data_loader_name}")

    @staticmethod
    def get_all_data_loaders(loader_type: DataLoaderType) -> Dict[str, DataLoader]:
        """
        Returns a dictionary containing all available data loaders.
        
        Args:
            loader_type (DataLoaderType): The type of data loader to retrieve.
            
        Returns:
            Dict[str, DataLoader]: A dictionary mapping data loader names to their instances.
        """
        if loader_type == DataLoaderType.PUBLICATION:
            return {
                "JsonPublicationLoader": JsonPublicationLoader()
            }
        if loader_type == DataLoaderType.QUESTION_ANSWERING:
            return {"CSVQALoader": CSVQALoader()}

        raise ValueError(f"Invalid data loader type: {loader_type}")
