from typing import Any, Dict, List
import re

from pylatexenc.latex2text import LatexNodes2Text

from core import LLMConfig
from core.data.models.publication import Publication

from .contribution import Contribution


class ORKGTemplateBuilder:
    """
    Builds ORKG paper payloads from publications and contribution blocks.

    The output format is compatible with the ORKG v2 papers endpoint.
    """

    _PREDICATE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9:_-]+$")

    def __init__(self, llm_config: LLMConfig, context_size: int = 4000):
        self.context_size = context_size
        self.llm_config = llm_config
        self.contributions: List[Contribution] = []

    def add_contribution(self, contribution: Contribution):
        """
        Adds a contribution to the ORKG template builder used for adding data
        to the ORKG graph.

        Args:
            contribution (Contribution): The contribution to add.
        """
        self.contributions.append(contribution)

    def create_template_for_publication(
        self,
        publication: Publication,
        research_field_id: str = "R659055",
    ) -> dict:
        """
        Generates an ORKG v2-compliant paper payload for a given publication.

        Args:
            publication (Publication): The publication object containing metadata
                and optional annotations.
            research_field_id (str, optional): The ORKG research field identifier.
                Defaults to "R659055".

        Returns:
            dict: ORKG paper payload for the v2 endpoint.
        """
        temp_objects: Dict[str, Dict[str, Dict[str, Any]]] = {
            "resources": {},
            "literals": {},
            "predicates": {},
            "lists": {},
        }
        temp_counters = {
            "resource": 0,
            "literal": 0,
            "predicate": 0,
            "list": 0,
        }

        contributions = []
        for contri in self.contributions:
            contri_data = contri.get_contribution_data(publication)
            if not contri_data:
                continue
            contributions.append(
                self._convert_contribution_to_v2(
                    contribution=contri_data,
                    temp_objects=temp_objects,
                    temp_counters=temp_counters,
                )
            )

        title = LatexNodes2Text().latex_to_text(publication.title).replace("\\", "")

        authors = []
        for author in publication.authors or []:
            author_name = LatexNodes2Text().latex_to_text(author)
            if not author_name:
                continue
            authors.append(
                {
                    "id": None,
                    "name": author_name,
                    "identifiers": None,
                    "homepage": None,
                }
            )

        identifiers: Dict[str, List[str]] = {}
        doi = LatexNodes2Text().latex_to_text(publication.doi) if publication.doi else ""
        if doi:
            identifiers["doi"] = [doi]

        publication_info: Dict[str, Any] = {}
        if publication.month is not None:
            publication_info["published_month"] = publication.month
        if publication.year is not None:
            publication_info["published_year"] = publication.year
        if publication.venue:
            publication_info["published_in"] = publication.venue
        if publication.url:
            publication_info["url"] = publication.url

        contents: Dict[str, Any] = {
            "contributions": contributions,
        }
        for key in ("resources", "literals", "predicates", "lists"):
            if temp_objects[key]:
                contents[key] = temp_objects[key]

        return {
            "title": title,
            "research_fields": [research_field_id],
            "identifiers": identifiers,
            "publication_info": publication_info,
            "authors": authors,
            "organizations": [],
            "observatories": [],
            "contents": contents,
            "extraction_method": "UNKNOWN",
        }

    def _convert_contribution_to_v2(
        self,
        contribution: Dict[str, Any],
        temp_objects: Dict[str, Dict[str, Dict[str, Any]]],
        temp_counters: Dict[str, int],
    ) -> Dict[str, Any]:
        values = contribution.get("values", {}) if isinstance(contribution, dict) else {}
        statements = self._convert_values_to_statements(
            values=values,
            temp_objects=temp_objects,
            temp_counters=temp_counters,
        )

        return {
            "label": contribution.get("name", "Contribution"),
            "classes": ["Contribution"],
            "statements": statements,
        }

    def _convert_values_to_statements(
        self,
        values: Any,
        temp_objects: Dict[str, Dict[str, Dict[str, Any]]],
        temp_counters: Dict[str, int],
    ) -> Dict[str, List[Dict[str, Any]]]:
        statements: Dict[str, List[Dict[str, Any]]] = {}
        if not isinstance(values, dict):
            return statements

        for predicate, objects in values.items():
            predicate_id = self._to_predicate_id(
                predicate=predicate,
                temp_objects=temp_objects,
                temp_counters=temp_counters,
            )
            if not isinstance(objects, list):
                objects = [objects]

            converted_objects = []
            for obj in objects:
                if obj is None:
                    continue
                converted_objects.append(
                    self._to_statement_object(
                        obj=obj,
                        temp_objects=temp_objects,
                        temp_counters=temp_counters,
                    )
                )

            if converted_objects:
                statements[predicate_id] = converted_objects

        return statements

    def _to_statement_object(
        self,
        obj: Any,
        temp_objects: Dict[str, Dict[str, Dict[str, Any]]],
        temp_counters: Dict[str, int],
    ) -> Dict[str, Any]:
        if isinstance(obj, dict):
            if "@id" in obj and isinstance(obj["@id"], str):
                return {"id": obj["@id"], "statements": None}

            if "id" in obj and isinstance(obj["id"], str):
                nested = obj.get("statements")
                if isinstance(nested, dict):
                    nested = self._convert_values_to_statements(
                        values=nested,
                        temp_objects=temp_objects,
                        temp_counters=temp_counters,
                    )
                else:
                    nested = None
                return {"id": obj["id"], "statements": nested}

            if "text" in obj:
                literal_id = self._next_temp_id("literal", temp_counters)
                literal_payload: Dict[str, Any] = {"label": str(obj.get("text", ""))}
                if "data_type" in obj and obj.get("data_type"):
                    literal_payload["data_type"] = obj.get("data_type")
                temp_objects["literals"][literal_id] = literal_payload
                return {"id": literal_id, "statements": None}

            if "label" in obj:
                resource_id = self._next_temp_id("resource", temp_counters)
                classes = obj.get("classes", [])
                if not isinstance(classes, list):
                    classes = [str(classes)]
                if not classes:
                    classes = ["Thing"]

                temp_objects["resources"][resource_id] = {
                    "label": str(obj.get("label", "")),
                    "classes": classes,
                }

                nested = self._convert_values_to_statements(
                    values=obj.get("values", {}),
                    temp_objects=temp_objects,
                    temp_counters=temp_counters,
                )
                return {"id": resource_id, "statements": nested or None}

        literal_id = self._next_temp_id("literal", temp_counters)
        temp_objects["literals"][literal_id] = {"label": str(obj)}
        return {"id": literal_id, "statements": None}

    def _to_predicate_id(
        self,
        predicate: Any,
        temp_objects: Dict[str, Dict[str, Dict[str, Any]]],
        temp_counters: Dict[str, int],
    ) -> str:
        if isinstance(predicate, str) and self._PREDICATE_ID_PATTERN.match(predicate):
            return predicate

        predicate_id = self._next_temp_id("predicate", temp_counters)
        predicate_label = str(predicate)
        temp_objects["predicates"][predicate_id] = {
            "label": predicate_label,
            "description": predicate_label,
        }
        return predicate_id

    @staticmethod
    def _next_temp_id(kind: str, temp_counters: Dict[str, int]) -> str:
        temp_counters[kind] += 1
        return f"#temp_{kind}_{temp_counters[kind]}"
