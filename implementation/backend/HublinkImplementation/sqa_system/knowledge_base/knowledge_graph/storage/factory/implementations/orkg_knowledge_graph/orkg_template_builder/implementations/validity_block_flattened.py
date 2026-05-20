from typing_extensions import override
from ..base.annotations_block import AnnotationsBlock

class ValidityFlattenedBlock(AnnotationsBlock):
    """
    A block for adding the validity information of a publication to the ORKG template.
    The data is flattened, meaning that there is no nesting of values.
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

        validity = annotations.get("Validity", {})

        validity_list = {
            "construct_validity": False,
            "external_validity": False,
            "internal_validity": False,
            "confirmability": False,
            "repeatability": False,
        }

        mapping_to_name = {
            "construct_validity": "Construct Validity",
            "external_validity": "External Validity",
            "internal_validity": "Internal Validity",
            "confirmability": "Confirmability",
            "repeatability": "Repeatability",
        }

        if "Threats To Validity" in validity and isinstance(validity, dict):
            if isinstance(validity["Threats To Validity"], list):
                for threat in validity["Threats To Validity"]:
                    threat_key = threat.lower().replace(" ", "_")
                    if threat_key in validity_list:
                        validity_list[threat_key] = True
            else:
                # It is a single threat
                threat_key = validity["Threats To Validity"].lower().replace(
                    " ", "_")
                if threat_key in validity_list:
                    validity_list[threat_key] = True

        result = {}

        has_validity_threats = False
        for name, truth in validity_list.items():
            if not truth:
                continue
            has_validity_threats = True
            result.setdefault("P39099", []).append(
                {"text": mapping_to_name[name]})
        if not has_validity_threats:
            result.setdefault("P39099", []).append({"text": "False"})

        self._add_additional_metadata(result, validity)
        return result

    def _add_additional_metadata(self, result: dict, validity: dict):

        if "Referenced Threats To Validity Guideline" in validity:
            result["P162071"] = [{"text": "True"}]
        else:
            result["P162071"] = [{"text": "False"}]
