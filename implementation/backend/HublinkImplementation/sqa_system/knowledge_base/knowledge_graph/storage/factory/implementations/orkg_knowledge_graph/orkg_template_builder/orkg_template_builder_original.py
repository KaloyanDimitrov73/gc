from typing import List
from pylatexenc.latex2text import LatexNodes2Text
from sqa_system.core.config.models import LLMConfig
from sqa_system.core.data.models.publication import Publication

from .contribution import Contribution


class ORKGTemplateBuilder:
    """
    This is the main class that builds all the ORKG templates for the publication.
    It is responsible for adding the contributions to the ORKG graph.
    It uses the blocks added to the builder to build the template.
    It is also responsible for adding the metadata to the ORKG graph.

    Args:
        llm_config (LLMConfig): The configuration for the language model to use for
            the paper content extraction.
        context_size (int, optional): The context size to use for the paper content 
            extraction.
    """

    def __init__(self, llm_config: LLMConfig, context_size: int = 4000):
        self.context_size = context_size
        self.llm_config = llm_config
        self.contributions: List[Contribution] = []

    def add_contribution(self, contribution: Contribution):
        """
        Adds a contribution to the ORKG template builder used for adding data
        to the ORKG graph.

        Args:
            contribution (Contribution): The contribution to add.
        """
        self.contributions.append(contribution)

    def create_template_for_publication(self,
                                        publication: Publication,
                                        research_field_id: str = "R659055") -> dict:
        """
        Generates an ORKG-compliant template for a given publication.
        It uses the blocks added to the builder to build the template.

        Args:
            publication (Publication): The publication object containing metadata 
                (e.g., title, authors, publication date, venue) and optional annotations.
            research_field_id (str, optional): The ORKG research field identifier. 
                Defaults to "R659055".
        Returns:
            dict: A dictionary representing the publication in the ORKG template format.
        """
        contributions = []
        for contri in self.contributions:
            contri_data = contri.get_contribution_data(publication)
            if not contri_data:
                continue
            contributions.append(contri_data)

        authors = []
        for author in publication.authors:
            authors.append(
                {
                    "label": LatexNodes2Text().latex_to_text(author)
                }
            )

        return {
            "predicates": [],
            "paper": {
                "title": LatexNodes2Text().latex_to_text(publication.title).replace("\\", ""),
                "doi": LatexNodes2Text().latex_to_text(publication.doi),
                "researchField": research_field_id,
                "authors": authors,
                "publicationMonth": publication.month,
                "publicationYear": publication.year,
                "publishedIn": publication.venue,
                "contributions": contributions
            }
        }
