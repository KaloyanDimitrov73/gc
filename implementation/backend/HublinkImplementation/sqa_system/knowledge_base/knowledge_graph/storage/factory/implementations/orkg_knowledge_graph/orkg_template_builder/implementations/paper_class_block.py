from typing_extensions import override
from ..base.annotations_block import AnnotationsBlock


class PaperClassBlock(AnnotationsBlock):
    """
    A block for adding the paper class of a publication to the ORKG template.
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
        paper_class_list = {
            "evaluation_research": False,
            "philosophical_papers": False,
            "opinion_paper": False,
            "proposal_of_solution": False,
            "personal_experience_papers": False,
            "validation_research": False,
        }

        ids = {
            "evaluation_research": "P154074",
            "philosophical_papers": "P154075",
            "opinion_paper": "P154076",
            "proposal_of_solution": "P154077",
            "personal_experience_papers": "P154078",
            "validation_research": "P154079",
        }

        if isinstance(paper_class, list):
            for pc in paper_class:
                key = pc.lower().replace(" ", "_")
                if key in ids:
                    paper_class_list[key] = True
        else:
            key = paper_class.lower().replace(" ", "_")
            if key in ids:
                paper_class_list[key] = True

        values_dict = {}
        anything_true = False
        for key, value_id in ids.items():
            values_dict[value_id] = [
                {
                    "text": "True" if paper_class_list[key] else "False"
                }
            ]
            if paper_class_list[key]:
                anything_true = True
        if not anything_true:
            return {}

        return {
            "P154073": [  # Paper Class
                {
                    "classes": ["C97001"],
                    "values": {
                        **values_dict
                    },
                    "label": "Paper Class"
                }
            ]
        }
