from typing_extensions import override
from ..base.annotations_block import AnnotationsBlock


class ResearchLevelFlattenedBlock(AnnotationsBlock):
    """
    A block for adding the research level of a publication to the ORKG template.
    This implementation flattens the data meaning that there is no nesting of values.
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

        # Prepare mapping for research level and ids
        mapping = {
            "primary_research": False,
            "secondary_research": False,
        }

        mapping_to_name = {
            "primary_research": "Primary Research",
            "secondary_research": "Secondary Research",
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

        result = {}
        for name, truth in mapping.items():
            if not truth:
                continue
            result.setdefault("P162008", []).append(
                {"text": mapping_to_name[name]})

        return result
