from abc import ABC, abstractmethod

from sqa_system.core.data.models import Publication
from .orkg_template_block import OrkgTemplateBlock


class MetadataBlock(OrkgTemplateBlock, ABC):
    """
    This block is responsible for filling the metadata of a publication
    node in the ORKG graph.
    """

    @abstractmethod
    def build(self, publication: Publication) -> dict:
        """
        Responsible for filling the metadata of a publication node in the ORKG graph.
        
        Args:
            publication: The publication to build the metadata for.
            
        Returns:
            dict: The constructed dictionary that the API from ORKG can use to add
                metadata to the publication node.
        """
