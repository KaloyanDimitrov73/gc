import json
import os
from typing import Dict, Any, List, Optional
from typing_extensions import override
from pydantic import ValidationError

from sqa_system.core.data.data_loader.base.data_loader import DataLoader
from sqa_system.core.data.models.dataset.implementations.publication_dataset import PublicationDataset
from sqa_system.core.data.models.publication import Publication
from sqa_system.core.logging.logging import get_logger
logger = get_logger(__name__)


class JsonPublicationLoader(DataLoader):
    """
    A data loader implementation for loading data from a JSON file and converting it
    to a PublicationDataset.
    
    This is the loader that we used for our experiments. This is not a generalized loader
    but rather a specialized loader for the JSON file that we used in our experiments.
    """

    def __init__(self, field_mapping: Optional[Dict[str, str]] = None):
        """
        In the constructor, we define the field mapping which maps the 
        fields of the JSON file to the fields of the Publication object.
        
        Args:
            field_mapping (Optional[Dict[str, str]]): A dictionary that maps the fields of the JSON file
                to the fields of the Publication object. If not provided, a default mapping is used.
        """
        self.field_mapping = field_mapping or {
            "doi": "doi",
            "authors": "author",
            "title": "title",
            "year": "year",
            "venue": "venue_name",
            "abstract": "abstract",
            "research_field": "track",
            "full_text": "fulltext",
            "publisher": "publisher",
        }

    @override
    def load(self, dataset_name: str, path: str, limit: Optional[int] = None) -> PublicationDataset:
        if not os.path.exists(path):
            logger.error(f"File not found at {path}")
            raise FileNotFoundError(f"File not found at {path}")

        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        publications: dict[str, Publication] = {}
        papers = self._get_papers(data)
        papers = self._parse_authors(papers)
        papers = self._parse_annotations(papers)

        # Here we remove unnecessary fields from the JSON file which we are not using
        fields_to_remove = ["abstract_snt",
                            "abstract",
                            "markedValid",
                            "markedChecked",
                            "booktitle",
                            "pages",
                            "bibsource",
                            "biburl",
                            "creationdate",
                            "file",
                            "url",]
        
        # if the field is in the dictionary, remove it
        for paper in papers:
            for field in fields_to_remove:
                if field in paper:
                    del paper[field]

        for paper in papers:
            try:
                pub = self.parse_dictionary_to_publication(
                    paper, self.field_mapping)
                publications[pub.doi] = pub
                if limit is not None and 0 < limit:
                    if len(publications) >= limit:
                        break
            except ValidationError as e:
                logger.warning("Skipping invalid publication: %s", e)

        if len(publications) == 0:
            logger.warning("No valid publications found in JSON file")

        return PublicationDataset(name=dataset_name, data=publications)

    def _parse_authors(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parses the authors from the papers and converts them into a list format.
        
        Args:
            papers (List[Dict[str, Any]]): A list of paper dictionaries to parse authors from.

        Returns:
            List[Dict[str, Any]]: The list of papers with authors parsed into a list format.
        """
        for paper in papers:
            authors_entry = paper.get(self.field_mapping["authors"])
            if not isinstance(authors_entry, list) and authors_entry:
                authors = authors_entry.split(" and ")
                paper[self.field_mapping["authors"]] = authors
        return papers

    def _parse_annotations(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parses the annotations from the papers and extracts relevant information.
        
        Args:
            papers (List[Dict[str, Any]]): A list of paper dictionaries to parse annotations from.

        Returns:
            List[Dict[str, Any]]: The list of papers with annotations parsed and extracted.
        """
        for paper in papers:
            classes_str = paper.get("classes")
            if classes_str:
                parsed_classes, _ = self._extract_annotations(
                    "annotations", classes_str)
                paper["annotations"] = parsed_classes
                paper.pop("classes")
            else:
                logger.warning(
                    f"No classes field found for entry {paper.get('doi', 'unknown')}.")
        return papers

    def _extract_annotations(self, current_key: str, s: str) -> tuple[Dict[str, Any], str]:
        """
        A recursive function to extract annotations from a the string which is provided
        in the json file.
        
        Args:
            current_key (str): The current key being processed.
            s (str): The string to extract annotations from.
            
            
        Returns:
            Dict: A dictionary containing the extracted annotations.
            str: The remaining string after extraction.
        """
        current_key_dict = {}
        while s:
            s = s.strip()

            if s.startswith("{"):
                s = s[1:].strip()
                continue

            if s.startswith("}"):
                s = s[1:].strip()
                return current_key_dict, s

            if s.startswith(","):
                s = s[1:].strip()
                continue

            # Find the next opening and closing braces
            opening_brace = s.find("{")
            closing_brace = s.find("}")
            next_comma = s.find(",")

            if opening_brace == -1 or (closing_brace != -1 and closing_brace < opening_brace):
                # We found a value
                value = s[:closing_brace].strip()
                # Handle list values separated by commas
                if "," in value:
                    value = [v.strip() for v in value.split(",")]
                
                s = s[closing_brace + 1:].strip()
                if current_key_dict:
                    current_key_dict[value] = True
                    return current_key_dict, s
                return value, s
            
            if next_comma != -1 and next_comma < opening_brace:
                # We found a boolean value
                key = s[:next_comma].strip()
                value = "true"
                if current_key:
                    current_key_dict[key] = value
                s = s[next_comma + 1:].strip()
                # there could still be another value
                continue

            # There is a nested structure, find the key
            key = s[:opening_brace].strip()
            s = s[opening_brace:].strip()

            # Recursively process the nested structure
            nested_dict, s = self._extract_annotations(key, s)
            current_key_dict[key] = nested_dict

        return current_key_dict, s

    def _get_papers(self, data: Any) -> List[Dict[str, Any]]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "papers" in data:
            return data["papers"]

        raise ValueError(
            "Invalid JSON structure: expected a list of papers or a dict with a 'papers' key")
