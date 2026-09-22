from typing_extensions import override
from ..base.annotations_block import AnnotationsBlock


class EvidenceBlock(AnnotationsBlock):
    """
    A block for adding the evidence information of a publication to the ORKG template.
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

        final_dict = {
            "P5004": [  # Evidence
                {
                    "classes": ["C109004"],  # Evidence Information
                    "values": {},
                    "label": "Evidence"
                }
            ]
        }

        self._add_replication_package(final_dict, validity)
        self._add_replication_package_link(final_dict, additional_fields)
        self._add_input_data(validity, final_dict)
        self._add_tool_support(validity, final_dict)
        return final_dict

    def _add_replication_package(self, final_dict: dict, validity: dict):
        if "Replication Package" in validity:
            final_dict["P5004"][0]["values"]["P162020"] = [{"text": "True"}]
        else:
            final_dict["P5004"][0]["values"]["P162020"] = [{"text": "False"}]

    def _add_tool_support(self, validity: dict, final_dict: dict):

        tool_support_list = {
            "used": False,
            "available": False,
            "none": False
        }

        tool_support_ids = {
            "used": "P20046",
            "available": "P4033",
            "none": "P169001"
        }

        if "Tool Support" in validity:
            value = validity["Tool Support"]
            if value.lower() == "used":
                tool_support_list["used"] = True
            elif value.lower() == "available":
                tool_support_list["available"] = True
        else:
            tool_support_list["none"] = True

        values_dict = {}
        for key, value_id in tool_support_ids.items():
            values_dict[value_id] = [
                {
                    "text": "True" if tool_support_list[key] else "False"
                }
            ]

        final_dict["P5004"][0]["values"]["P168009"] = [
            {
                "classes": ["C110000"],
                "values": {
                    **values_dict
                },
                "label": "Tool Support"
            }
        ]

    def _add_input_data(self, validity: dict, final_dict: dict):
        input_data_list = {
            "used": False,
            "available": False,
            "none": False
        }

        input_data_ids = {
            "used": "P20046",
            "available": "P4033",
            "none": "P169001"
        }

        if "Input Data" in validity:
            value = validity["Input Data"]
            if value.lower() == "used":
                input_data_list["used"] = True
            elif value.lower() == "available":
                input_data_list["available"] = True
        else:
            input_data_list["none"] = True

        values_dict = {}
        for key, value_id in input_data_ids.items():
            values_dict[value_id] = [
                {
                    "text": "True" if input_data_list[key] else "False"
                }
            ]

        final_dict["P5004"][0]["values"]["P169004"] = [
            {
                "classes": ["C110000"],
                "values": {
                    **values_dict
                },
                "label": "Input Data"
            }
        ]

    def _add_replication_package_link(self, final_dict: dict, additional_fields: dict):
        if ("replication_package_link" not in additional_fields or
                len(additional_fields["replication_package_link"]) == 0):
            return

        links = additional_fields["replication_package_link"]

        if not isinstance(links, list):
            links = [links]

        for link in links:
            final_dict["P5004"][0]["values"].setdefault(
                "P168010", []).append({"text": link})
