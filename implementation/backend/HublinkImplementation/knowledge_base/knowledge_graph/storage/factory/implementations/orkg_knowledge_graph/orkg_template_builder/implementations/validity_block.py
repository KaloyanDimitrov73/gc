from typing_extensions import override
from ..base.annotations_block import AnnotationsBlock


class ValidityBlock(AnnotationsBlock):
    """
    A block for adding the validity information of a publication to the ORKG template.
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

        threats_ids = {
            "construct_validity": "P55037",
            "external_validity": "P55034",
            "internal_validity": "P55035",
            "confirmability": "P162070",
            "repeatability": "P97002",
        }

        # Update validity_list based on the threats present
        if "Threats To Validity" in validity and isinstance(validity, dict):
            # check if validity["Threats To Validity"] is a list
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

        # Create a single dictionary by merging all threat entries
        threats_dict = {}
        for threat_key, threat_id in threats_ids.items():
            threats_dict[threat_id] = [
                {
                    "text": "True" if validity_list[threat_key] else "False"
                }
            ]

        final_dict = {
            "P162017": [  # Validity
                {
                    "classes": ["C103014"],  # Validity Information
                    "values": {
                        "P123037": [  # Threats to validity
                            {
                                # Threat to Validity
                                "classes": ["C103025"],
                                "values": threats_dict,
                                "label": "Threat to Validity"
                            }
                        ]
                    },
                    "label": "Validity"
                }
            ]
        }

        self._add_additional_metadata(final_dict, validity)
        return final_dict

    def _add_additional_metadata(self, final_dict: dict, validity: dict):

        if "Referenced Threats To Validity Guideline" in validity:
            final_dict["P162017"][0]["values"]["P162071"] = [
                {"text": "True"}]
        else:
            final_dict["P162017"][0]["values"]["P162071"] = [
                {"text": "False"}]
