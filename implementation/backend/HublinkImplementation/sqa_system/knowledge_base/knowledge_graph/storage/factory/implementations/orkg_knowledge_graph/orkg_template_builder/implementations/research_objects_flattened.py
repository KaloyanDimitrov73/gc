from abc import ABC
from typing_extensions import override

from sqa_system.core.logging.logging import get_logger

from ..base.annotations_block import AnnotationsBlock

logger = get_logger(__name__)


class ResearchObjectsFlattenedBlock(AnnotationsBlock, ABC):
    """
    A block for adding the research objects of a publication to the ORKG template.
    This implementation flattens the data meaning that there is no nesting of values.
    """
    
    def __init__(self):
        super().__init__()
        self.seen_sub_properties = set()
        self.seen_properties = set()
        self.seen_evaluation_methods = set()
        self.seen_has_eval_guideline = set()

    def _build_research_object(self, additional_fields: dict, object_prefix: str):
        if additional_fields is None or "annotations" not in additional_fields:
            return {}
        annotations = additional_fields["annotations"]
        if len(annotations) == 0:
            return {}

        research_object_data = {}
        research_object_key = f"{object_prefix} Research Object"
        for key, value in annotations.items():
            if research_object_key in key:
                research_object = value.get("Research Object", None)
                if not research_object:
                    continue

                # Add research object directly
                if isinstance(research_object, list):
                    for obj in research_object:
                        research_object_data.setdefault(
                            "P123038", []).append({"text": obj})
                elif isinstance(research_object, str):
                    research_object_data.setdefault(
                        "P123038", []).append({"text": research_object})

                # Accumulate evaluation methods
                temp_evaluation_methods = self._create_evaluation_methods(
                    value)
                for k, v in temp_evaluation_methods.items():
                    research_object_data.setdefault(k, []).extend(v)

                # Accumulate properties
                temp_properties = self._create_properties(value)
                for k, v in temp_properties.items():
                    research_object_data.setdefault(k, []).extend(v)

                # Accumulate evaluation guideline
                temp_has_eval_guideline = self._create_has_eval_guideline(
                    value)
                for k, v in temp_has_eval_guideline.items():
                    research_object_data.setdefault(k, []).extend(v)

        if not research_object_data:
            return {}

        return research_object_data

    def _create_evaluation_methods(self, value):
        temp_evaluations = {}
        for k, v in value.items():
            if isinstance(v, dict) and "evaluation" in k.lower():
                method = v.get("Evaluation Method", None)
                if not method:
                    temp_evaluations.setdefault(
                        "P59089", []).append({"text": "False"})
                else:
                    if isinstance(method, list):
                        for m in method:
                            temp_evaluations.setdefault(
                                "P59089", []).append({"text": m})
                    else:
                        temp_evaluations.setdefault(
                            "P59089", []).append({"text": method})
        # here we remove any duplicate entries and if no entries are there
        # we add a False entry
        if "P59089" in temp_evaluations:
            for item in temp_evaluations["P59089"]:
                if item["text"] not in self.seen_evaluation_methods:
                    self.seen_evaluation_methods.add(item["text"])
        else:
            temp_evaluations["P59089"] = [{"text": "False"}]
        return temp_evaluations

    def _create_properties(self, value):
        property_entries = []
        sub_property_entries = []
        for k, v in value.items():
            if isinstance(v, dict) and "evaluation" in k.lower():
                properties = v.get("Properties", None)
                if not properties:
                    property_entries.append({"text": "False"})
                    continue
                for prop_name, sub_prop in properties.items():
                    property_entries.append({"text": prop_name})
                    if not sub_prop:
                        continue
                    if isinstance(sub_prop, list):
                        for item in sub_prop:
                            sub_property_entries.append({"text": item})
                    elif isinstance(sub_prop, str):
                        sub_property_entries.append({"text": sub_prop})

        # here we remove any duplicate entries
        property_entries = [item for item in property_entries
                            if item["text"] not in self.seen_properties and not self.seen_properties.add(item["text"])]

        sub_property_entries = [item for item in sub_property_entries
                                if item["text"] not in self.seen_sub_properties and not self.seen_sub_properties.add(item["text"])]

        result = {}
        if property_entries:
            result["P168001"] = property_entries
        else:
            result["P168001"] = [{"text": "False"}]
        if sub_property_entries:
            result["P168002"] = sub_property_entries
        else:
            result["P168002"] = [{"text": "False"}]
        return result

    def _create_has_eval_guideline(self, value):
        any_true = False
        for k, v in value.items():
            if isinstance(v, dict) and "evaluation" in k.lower():
                guideline = v.get("Referenced Evaluation Guideline", False)
                if guideline:
                    any_true = True
                    break
        return {"P168003": [{"text": "True" if any_true else "False"}]}


class AllResearchObjectsFlattenedBlock(ResearchObjectsFlattenedBlock):
    """
    Building block for adding all research object annotations to a single
    contribution.
    This implementation flattens the data meaning that there is no nesting of values.
    """

    @override
    def build(self, additional_fields: dict) -> dict:
        return self._build_research_object(additional_fields, "")


class FirstResearchObjectFlattenedBlock(ResearchObjectsFlattenedBlock):
    """
    Building block for adding the first research object annotation to the
    contribution.
    This implementation flattens the data meaning that there is no nesting of values.
    """

    @override
    def build(self, additional_fields: dict) -> dict:
        return self._build_research_object(additional_fields, "First")


class SecondResearchObjectFlattenedBlock(ResearchObjectsFlattenedBlock):
    """
    Building block for adding the second research object annotation to the
    contribution.
    This implementation flattens the data meaning that there is no nesting of values.
    """

    @override
    def build(self, additional_fields: dict) -> dict:
        return self._build_research_object(additional_fields, "Second")
