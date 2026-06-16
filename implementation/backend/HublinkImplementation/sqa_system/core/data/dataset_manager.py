from typing import Dict, Optional

from sqa_system.core.config.models.dataset_config import DatasetConfig
from sqa_system.core.data.data_loader.factory.data_loader_factory import DataLoaderFactory
from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.data.models.dataset.base.dataset import Dataset


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
    _datasets: Dict[str, Dataset] = {}
    _file_path_manager = FilePathManager()

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(DatasetManager, cls).__new__(cls)
        return cls._instance

    def get_dataset(self, config: DatasetConfig, file_path: Optional[str] = None) -> Dataset:
        """
        Retrieves a dataset based on the provided dataset ID.

        Args:
            config (DatasetConfig): The configuration of the dataset.
            file_path (Optional[str]): The path to the dataset file.

        Returns:
            Dataset: The dataset with the given ID.
        """
        data_loader = DataLoaderFactory.get_data_loader(config.loader)
        if not file_path:
            file_path = self._file_path_manager.get_path(config.file_name)
        dataset = data_loader.load(dataset_name=config.name,
                                   path=file_path,
                                   limit=config.loader_limit)
        return dataset
