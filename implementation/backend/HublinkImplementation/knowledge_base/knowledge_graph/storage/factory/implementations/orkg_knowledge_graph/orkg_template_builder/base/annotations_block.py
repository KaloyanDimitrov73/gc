from abc import ABC, abstractmethod

from .orkg_template_block import OrkgTemplateBlock


class AnnotationsBlock(OrkgTemplateBlock, ABC):
    """
    In our master thesis, we prepared templates for contributions for the ORKG.
    A annotationsblock is responsible for filling these templates with data.
    """

    @abstractmethod
    def build(self, additional_fields: dict) -> dict:
        """
        Builds the template part for a publication node in the ORKG graph
        using the paper content.
        
        Args:
            additional_fields: The dictionary containing the additional fields
                of the publication which have been loaded by the loader.
        
        Returns:
            dict: The constructed template part for the publication node that
                is in a format that the API from ORKG can use to add
                contributions to the publication node.
        """
