from typing_extensions import override
from ..base.annotations_block import AnnotationsBlock


class EvidenceFlattenedBlock(AnnotationsBlock):
    """
    A block for adding the evidence information of a publication to the ORKG template.
    The data is flattened, meaning that there is no nesting of values.
    """

    @override
    def build(self, additional_fields: dict) -> dict:
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

        result = {}

        if "Replication Package" in validity:
            result["P162020"] = [{"text": "True"}]
        else:
            result["P162020"] = [{"text": "False"}]

        if "Tool Support" in validity:
            value = validity["Tool Support"]
            if value == "used":
                result["P168009"] = [
                    {"text": "Tool was used but is not available"}]
            elif value == "available":
                result["P168009"] = [{"text": "Tool is available"}]
        else:
            result["P168009"] = [{"text": "No tool used"}]

        if "Input Data" in validity:
            value = validity["Input Data"]
            if value == "used":
                result["P169004"] = [
                    {"text": "Input data was used but is not available"}]
            elif value == "available":
                result["P169004"] = [{"text": "Input data is available"}]
        else:
            result["P169004"] = [{"text": "No input data used"}]

        self._add_replication_package_link(result, additional_fields)

        return result

    def _add_replication_package_link(self, result: dict, additional_fields: dict):
        """
        Adds the replication package link to the result dictionary.

        Args:
            result: The result dictionary to which the replication package link
                will be added.
            additional_fields: The additional fields dictionary containing the
                replication package link.
        """
        if ("replication_package_link" not in additional_fields or
                len(additional_fields["replication_package_link"]) == 0):
            return

        links = additional_fields["replication_package_link"]

        if not isinstance(links, list):
            links = [links]

        for link in links:
            result.setdefault(
                "P168010", []).append({"text": link})
