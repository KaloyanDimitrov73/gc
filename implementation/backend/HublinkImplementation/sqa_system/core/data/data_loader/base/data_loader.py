from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from sqa_system.core.data.models.publication import Publication
from sqa_system.core.data.models.dataset.base.dataset import Dataset


class DataLoader(ABC):
    """
    Abstract base class for loader implementations that load datasets.
    """

    @abstractmethod
    def load(self, dataset_name: str, path: str, limit: Optional[int] = None) -> Dataset:
        """
        Load the dataset from the specified path.

        Parameters:
            dataset_name (str): The name of the dataset.
            path (str): The path to the dataset.
            limit (Optional[int]): The maximum number of entries to load.

        Returns:
            Dataset: The loaded dataset.
        """

    def parse_dictionary_to_publication(self,
                                        paper: Dict[str, Any],
                                        field_mapping: Dict[str, str]) -> Publication:
        """
        Receives a dictionary and a field mapping. Based on the
        received mapping which maps each dictionary entry to its
        publication field, the function will return a Publication
        object with the data from the dictionary.

        Args:
            paper (Dict[str, Any]): The dictionary to be parsed.
            field_mapping (Dict[str, str]): The field mapping.
                Which is used to map each dictionary entry to its
                publication field. The keys map to the publication
                fields and the values are the dictionary keys.

        Returns:
            Publication: The parsed Publication object.
        """
        publication_data = {}
        for key, value in paper.items():
            if key in field_mapping.values():
                field_mapping_key = [
                    k for k, v in field_mapping.items() if v == key][0]
                if value is None:
                    continue
                if isinstance(value, str):
                    field_value = value.strip("[]{}")
                    field_value = field_value.replace("{", "").replace("}", "")
                    publication_data[field_mapping_key] = field_value
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], str):
                    publication_data[field_mapping_key] = [
                        v.strip("[](){}") for v in value]

            else:
                if "additional_fields" not in publication_data:
                    publication_data["additional_fields"] = {key: value}
                else:
                    publication_data["additional_fields"][key] = value

        if publication_data.get("year"):
            try:
                publication_data["year"] = int(publication_data["year"])
            except ValueError:
                publication_data["year"] = None

        return Publication(**publication_data)
