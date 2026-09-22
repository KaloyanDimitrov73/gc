from abc import ABC
from typing_extensions import override
from sqa_system.core.logging.logging import get_logger

from ..base.annotations_block import AnnotationsBlock

logger = get_logger(__name__)


class ResearchObjectBlock(AnnotationsBlock, ABC):
    """
    A block for adding the research objects of a publication to the ORKG template.
    """

    def _build_research_object(self, additional_fields: dict, object_prefix: str):
        if additional_fields is None or "annotations" not in additional_fields:
            return {}
        annotations = additional_fields["annotations"]
        if len(annotations) == 0:
            return {}

        research_objects = []
        research_object_key = f"{object_prefix} Research Object"
        for key, value in annotations.items():
            if research_object_key in key:
                research_object = value.get("Research Object", None)
                if not research_object:
                    continue

                temp_evaluations = self._create_evaluations(value)
                temp_research_object = self._create_object(
                    research_object, temp_evaluations)
                research_objects.append(temp_research_object)

        if not research_objects:
            return {}

        return {
            "P123038": research_objects
        }

    def _create_evaluations(self, value):
        temp_evaluations = []
        for k, v in value.items():
            if isinstance(v, dict) and "evaluation" in k.lower():
                method = v.get("Evaluation Method", None)

                guideline = v.get(
                    "Referenced Evaluation Guideline", False)

                properties = v.get("Properties", None)
                if not properties:
                    properties = {}

                temp_properties = []
                for prop_key, prop_value in properties.items():
                    property_value = self._create_property(
                        prop_key, prop_value)
                    temp_properties.append(property_value)
                evaluation = self._create_evaluation(
                    method, guideline, temp_properties)

                temp_evaluations.append(evaluation)
        return temp_evaluations

    def _create_property(self, value, list_of_sub_properties):
        sub_properties = []
        if list_of_sub_properties and isinstance(list_of_sub_properties, list):
            for sub_prop in list_of_sub_properties:
                sub_property = self._create_property(sub_prop, None)
                sub_properties.append(sub_property)
        elif list_of_sub_properties:
            sub_property = self._create_property(list_of_sub_properties, None)
            sub_properties.append(sub_property)
        return {
            "classes": ["C103018"],
            "values": {
                "P15191": [
                    {
                        "text": value
                    }
                ],
                "P162024": sub_properties
            },
            "label": "Property"
        }

    def _create_evaluation(self, method, guideline, properties):

        eval_method_data = []
        if method:
            if isinstance(method, list):
                for m in method:
                    eval_method_data.extend(self.create_entity_with_description(
                        [{"name": m}],
                        label="Evaluation Method"))
            else:
                eval_method_data.extend(self.create_entity_with_description(
                    [{"name": method}],
                    label="Evaluation Method"))
        else:
            eval_method_data = [{"text": "False"}]

        values = {
            "P59089": eval_method_data,   # Evaluation Method
            "P162023": [  # Evaluation Guideline
                {
                    "text": "True" if guideline else "False"
                }
            ],
        }
        if properties:
            values["P41769"] = properties
        else:
            values["P41769"] = [{"text": "False"}]

        return {
            "classes": ["C103017"],
            "values": values,
            "label": "Evaluation"
        }

    def _create_object(self, research_object, evaluations):

        if not evaluations or len(evaluations) == 0:
            evaluations = [self._create_evaluation(None, None, None)]

        research_object_data = []
        if isinstance(research_object, list):
            for name in research_object:
                research_object_data.extend(
                    self.create_entity_with_description([{"name": name}],
                                                        label="Research Object"))
        else:
            research_object_data = self.create_entity_with_description(
                [{"name": research_object}],
                label="Research Object")

        return {
            "classes": ["C103016"],
            "values": {
                "P47032": research_object_data,
                "P34": evaluations
            },
            "label": "Research Object"
        }


class AllResearchObjectsBlock(ResearchObjectBlock):
    """
    Building block for adding all research object annotations to to a 
    single contribution.
    """
    @override
    def build(self, additional_fields: dict) -> dict:
        return self._build_research_object(additional_fields, "")


class FirstResearchObjectBlock(ResearchObjectBlock):
    """
    Building block for adding the first research object annotation to the
    contribution.
    """

    @override
    def build(self, additional_fields: dict) -> dict:
        return self._build_research_object(additional_fields, "First")


class SecondResearchObjectBlock(ResearchObjectBlock):
    """
    Building block for adding the second research object annotation to the
    contribution.
    """

    @override
    def build(self, additional_fields: dict) -> dict:
        return self._build_research_object(additional_fields, "Second")
