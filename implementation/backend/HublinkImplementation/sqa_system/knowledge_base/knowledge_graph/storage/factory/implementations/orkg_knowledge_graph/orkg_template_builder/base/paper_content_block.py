from abc import ABC, abstractmethod
from sqa_system.core.data.extraction.paper_content_extractor import PaperContent

from .orkg_template_block import OrkgTemplateBlock


class PaperContentBlock(OrkgTemplateBlock, ABC):
    """
    This block is responsible for adding data to the orkg which has been extracted
    from the fulltext of the paper.
    """

    @abstractmethod
    def build(self, paper_content: PaperContent) -> dict:
        """
        Builds the template part for a publication node in the ORKG graph
        using the paper content.
        
        Args:
            paper_content: The content of the paper that is extracted from the fulltext.
        Returns:
            A dictionary that can be used to add the paper content to the ORKG graph which
            is in the format that is required by the ORKG API.
        """
