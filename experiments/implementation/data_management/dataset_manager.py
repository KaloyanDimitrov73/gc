from typing import Dict, Optional

from implementation.config.models.dataset_config import DatasetConfig
from implementation.data_loader.csv_qa_loader import CSVQALoader
from implementation.models.qa_dataset import QADataset
from .file_path_manager import FilePathManager

_LOADER_MAP = {
    "CSVQALoader": CSVQALoader,
}


class DatasetManager:
    """
    A manager class for handling datasets across the project.
    It allows to retrieve a dataset based only on the dataset
    configuration.

    Each loaded dataset is cached in a dictionary. This way, if
    the dataset is used multiple times in a run, it is only loaded
    once.

    It is implemented as a singleton pattern to ensure that only
    one instance of the class exists at any given time.
    """

    _instance = None
    _datasets: Dict[str, QADataset] = {}
    _file_path_manager = FilePathManager()

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(DatasetManager, cls).__new__(cls)
        return cls._instance

    def get_dataset(self, config: DatasetConfig, file_path: Optional[str] = None) -> QADataset:
        """
        Retrieves a dataset based on the provided dataset configuration.

        Args:
            config (DatasetConfig): The configuration of the dataset.
            file_path (Optional[str]): The path to the dataset file.

        Returns:
            QADataset: The loaded dataset.
        """
        loader_class = _LOADER_MAP.get(config.loader)
        if loader_class is None:
            raise ValueError(
                f"Unknown loader '{config.loader}'. "
                f"Available loaders: {list(_LOADER_MAP.keys())}"
            )
        data_loader = loader_class()
        if not file_path:
            file_path = self._file_path_manager.get_path(config.file_name)
        dataset = data_loader.load(dataset_name=config.name,
                                   path=file_path,
                                   limit=config.loader_limit)
        return dataset
