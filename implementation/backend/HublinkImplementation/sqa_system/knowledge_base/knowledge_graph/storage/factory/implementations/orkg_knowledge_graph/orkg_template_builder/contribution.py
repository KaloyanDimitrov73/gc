from typing import List, Optional
from pydantic import BaseModel, PrivateAttr

from sqa_system.core.config.models import LLMConfig
from sqa_system.core.data.models.publication import Publication
from sqa_system.core.data.extraction.paper_content_extractor import (
    PaperContent,
    PaperContentExtractor
)

from .base.paper_content_block import PaperContentBlock
from .base.annotations_block import AnnotationsBlock
from .base.metadata_block import MetadataBlock


class Contribution(BaseModel):
    """
    A simple base model for a ORKG contribution as it is required by the ORKG API.
    """
    llm_config: Optional[LLMConfig] = None
    context_size: int = 4000
    name: str = "Publication Overview"

    _data: dict = PrivateAttr(default={})
    _paper_content_blocks: List[PaperContentBlock] = PrivateAttr(default=[])
    _annotation_blocks: List[AnnotationsBlock] = PrivateAttr(default=[])
    _paper_content_extractor: PaperContentExtractor = PrivateAttr(default=None)
    _metadata_blocks: List[MetadataBlock] = PrivateAttr(default=[])

    # pylint: disable=arguments-differ
    def model_post_init(self, __context):
        self._data = {
            "name": self.name,
            "values": {}
        }
        self._paper_content_extractor = None

    def update_values(self, data: dict):
        """        
        Update the values in the contribution with the provided data.

        Args:
            data (dict): The data to update the values with.
        """
        self._data["values"].update(data)

    def get_contribution_data(self, publication: Publication) -> dict:
        """
        Get the data of the contribution for a given publication.
        This method builds the contribution data by combining the values from
        the paper content blocks, annotation blocks, and metadata blocks.
        
        Args:
            publication (Publication): The publication for which to get the data.

        Returns:
            dict: The data of the contribution.
        """
        if self._paper_content_blocks is not None and len(self._paper_content_blocks) > 0:
            paper_content = self._get_paper_content(publication)
            for block in self._paper_content_blocks:
                self.update_values(block.build(paper_content))

        for block in self._annotation_blocks:
            self.update_values(block.build(publication.additional_fields))

        if self._metadata_blocks is not None and len(self._metadata_blocks) > 0:
            for block in self._metadata_blocks:
                self.update_values(block.build(publication))

        # In the case that we only have one singular block, we remove the outmost
        # dictionary this one is only required when we have multiple blocks
        if len(self._data["values"]) == 1:
            first_item = list(self._data["values"].values())[0]
            if (isinstance(first_item, list) and
                isinstance(first_item[0], dict) and
                    "values" in first_item[0]):
                self._data["values"] = first_item[0]["values"]

        # When the contribution has no values, we return None
        if not self._data["values"]:
            return None

        return self._data

    def add_paper_content_template_block(self, block: PaperContentBlock):
        """
        Adds a paper content template block to the builder.
        
        Args:
            block (PaperContentBlock): The paper content block to add.
        """
        self._paper_content_blocks.append(block)

    def add_annotation_block(self, block: AnnotationsBlock):
        """
        Adds an annotation block to the builder.
        
        Args:
            block (AnnotationsBlock): The annotation block to add.
        """
        self._annotation_blocks.append(block)

    def add_metadata_block(self, block: MetadataBlock):
        """
        Adds a metadata block to the builder.
        
        Args:
            block (MetadataBlock): The metadata block to add.
        """
        self._metadata_blocks.append(block)

    def _get_paper_content(self, publication: Publication) -> PaperContent:
        """
        Runs the paper content extractor on the publication and returns the
        extracted paper content. 
        
        Args:
            publication (Publication): The publication for which to extract the
                paper content.
        Returns:
            PaperContent: The extracted paper content.
        """
        if self._paper_content_extractor is None:
            self._paper_content_extractor = PaperContentExtractor(
                llm_config=self.llm_config
            )
        return self._paper_content_extractor.extract_paper_content(
            publication=publication,
            update_cache=False,
            context_size=self.context_size
        )
