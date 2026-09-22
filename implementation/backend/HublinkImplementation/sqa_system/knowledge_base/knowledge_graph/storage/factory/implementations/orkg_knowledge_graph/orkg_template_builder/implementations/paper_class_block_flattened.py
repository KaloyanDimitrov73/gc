from typing_extensions import override
from ..base.annotations_block import AnnotationsBlock


class PaperClassFlattenedBlock(AnnotationsBlock):
    """
    A block for adding the paper class of a publication to the ORKG template.
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

        paper_class = meta_data.get("Paper class", None)
        if paper_class is None:
            return {}

        # Prepare mapping for paper class and ids
        mappings = {
            "evaluation_research": False,
            "philosophical_papers": False,
            "opinion_paper": False,
            "proposal_of_solution": False,
            "personal_experience_papers": False,
            "validation_research": False,
        }

        mapping_to_name = {
            "evaluation_research": "Evaluation Research",
            "philosophical_papers": "Philosophical Papers",
            "opinion_paper": "Opinion Paper",
            "proposal_of_solution": "Proposal of Solution",
            "personal_experience_papers": "Personal Experience Papers",
            "validation_research": "Validation Research",
        }

        if isinstance(paper_class, list):
            for pc in paper_class:
                key = pc.lower().replace(" ", "_")
                if key in mappings:
                    mappings[key] = True
        else:
            key = paper_class.lower().replace(" ", "_")
            if key in mappings:
                mappings[key] = True

        if not any(mappings.values()):
            raise ValueError("No valid paper class found.")

        result = {}
        for key, truth in mappings.items():
            if not truth:
                continue
            result.setdefault("P154073", []).append(
                {"text": mapping_to_name[key]})
        return result
