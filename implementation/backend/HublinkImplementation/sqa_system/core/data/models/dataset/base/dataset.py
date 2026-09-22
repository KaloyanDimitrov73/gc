from typing import Dict, Generator, List, Tuple, TypeVar, Generic
from datetime import datetime
from uuid import uuid4
import csv
import pandas as pd
from pydantic import BaseModel, Field
from sqa_system.core.data.file_path_manager import FilePathManager

T = TypeVar('T', bound=BaseModel)


class Dataset(BaseModel, Generic[T]):
    """
    The base class that represents a dataset in the SQA system. It allows the storage
    and access of various entries, each identified by a unique ID. The dataset can
    be used to manage and manipulate the entries, including adding, updating,
    deleting, and retrieving them. 
    
    Args:
        name (str): The name of the dataset.
        data (Dict[str, T]): A dictionary containing the entries in the dataset where each
            key is the entry ID and the value is the entry itself.
    """
    id: str = Field(default_factory=lambda: str(uuid4()),
                    description="The unique identifier of the dataset.")
    name: str = Field(description="The name of the dataset.")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    data: Dict[str, T] = Field(default_factory=dict,
                               description="The data in the dataset.")

    def __init__(self, name: str, data: Dict[str, T]):
        super().__init__(name=name, _data=data)
        self.name = name
        self.data = data
        self._update_timestamp()

    def add_entry(self, entry_id: str, entry: T):
        """
        Adds an entry to the dataset.

        Args:
            entry_id (str): The ID of the entry.
            entry (T): The entry to be added.

        Raises:
            KeyError: If an entry with the same ID already exists in the dataset.
        """
        if entry_id not in self.data:
            self.data[entry_id] = entry
            self._update_timestamp()
        else:
            raise KeyError(
                f"Entry with id {entry_id} already exists in dataset {self.name}.")

    def add_entries(self, entries: Dict[str, T]):
        """
        Adds multiple entries to the dataset.

        Args:
            entries (Dict[str, T]): A dictionary containing the entries to be added

        Raises:
            ValueError: If any of the entries have existing keys in the dataset.
        """
        existing_keys = set(entries.keys()) & set(self.data.keys())
        if existing_keys:
            # Remove the entries with existing keys
            for key in existing_keys:
                del entries[key]

        self.data.update(entries)
        self._update_timestamp()

    def update_entry(self, entry_id: str, entry: T):
        """
        Updates an entry in the dataset.

        Args:
            entry_id (str): The ID of the entry to update.
            entry (T): The new entry to replace the existing one with.

        Raises:
            KeyError: If the entry with the given ID does not exist in the dataset.
        """
        if entry_id in self.data:
            self.data[entry_id] = entry
            self._update_timestamp()
        else:
            raise KeyError(
                f"Entry with id {entry_id} not found in dataset {self.name}.")

    def delete_entry(self, entry_id: str):
        """
        Deletes an entry from the dataset.

        Args:
            entry_id (str): The ID of the entry to be deleted.

        Raises:
            KeyError: If the entry with the given ID does not exist in the dataset.
        """
        if entry_id in self.data:
            del self.data[entry_id]
            self._update_timestamp()
        else:
            raise KeyError(
                f"Entry with id {entry_id} not found in dataset {self.name}.")

    def get_entry(self, entry_id: str) -> T:
        """
        Retrieves an entry from the dataset based on the given entry ID.

        Args:
            entry_id (str): The ID of the entry to retrieve.

        Returns:
            T: The entry with the specified ID.

        Raises:
            KeyError: If the entry with the given ID is not found in the dataset.
        """
        if entry_id in self.data:
            return self.data[entry_id]

        raise KeyError(
            f"Entry with id {entry_id} not found in dataset {self.name}.")

    def get_all_entries(self) -> List[T]:
        """
        Returns a list of all entries in the dataset.

        Returns:
            List[T]: A list of all entries in the dataset.
        """
        return list(self.data.values())

    def iterate_entries(self) -> Generator[Tuple[str, T], None, None]:
        """
        Returns an iterator over the entries in the dataset.

        Yields:
            T: The next entry in the dataset.

        Raises:
            StopIteration: If there are no more entries in the dataset.
        """
        for entry_id, entry in self.data.items():
            yield entry_id, entry

    def to_dataframe(self):
        """
        Converts the dataset to a pandas DataFrame.
        """
        qa_pairs_dict = [qa_pair.model_dump()
                         for qa_pair in self.get_all_entries()]
        return pd.DataFrame(qa_pairs_dict)

    def save_as_csv(self, path: str) -> None:
        """
        Saves the dataset as a CSV file at the specified path.

        Args:
            path (str): The path where the CSV file should be saved.
        """
        if not self.data:
            raise ValueError(
                "Dataset is empty. Cannot save an empty dataset as CSV.")

        # Get fieldnames from the first entry
        first_entry = next(iter(self.data.values()))
        fieldnames = list(first_entry.dict().keys())

        # make sure that the directories exist
        file_path_manager = FilePathManager()
        file_path_manager.ensure_dir_exists(path)

        with open(path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self.data.values():
                writer.writerow(entry.dict())

    def _update_timestamp(self) -> None:
        self.updated_at = datetime.now()
