from typing_extensions import override
from ..base.annotations_block import AnnotationsBlock


class ResearchLevelBlock(AnnotationsBlock):
    """
    A block for adding the research level of a publication to the ORKG template.
    """

    @override
    def build(self, additional_fields: dict):
        """
        Builds a dictionary conforming to the ORKG API for adding contributions 
        to the ORKG graph. 

        Args:
            additional_fields: A dictionary containing the annotation classes
                and their values. 

        Returns:
            A dictionary that that conforms to the ORKG API for adding contributions
            to the ORKG graph. 
        """
        if additional_fields is None or "annotations" not in additional_fields:
            return {}
        annotations = additional_fields["annotations"]
        if len(annotations) == 0:
            return {}

        meta_data = annotations.get("Meta Data", None)
        if not meta_data:
            return {}

        research_level = meta_data.get("Research Level", None)
        if research_level is None:
            return {}

        mapping = {
            "primary_research": False,
            "secondary_research": False,
        }

        ids = {
            "primary_research": "P162009",
            "secondary_research": "P57021",
        }

        if isinstance(research_level, list):
            for level in research_level:
                key = level.lower().replace(" ", "_")
                if key in mapping:
                    mapping[key] = True
        else:
            key = research_level.lower().replace(" ", "_")
            if key in mapping:
                mapping[key] = True

        values_dict = {}
        for key, value_id in ids.items():
            values_dict[value_id] = [
                {
                    "text": "True" if mapping[key] else "False"
                }
            ]

        return {
            "P162008": [  # Research Level
                {
                    "classes": ["C103004"],
                    "values": {
                        **values_dict
                    },
                    "label": "Research Level"
                }
            ]
        }
