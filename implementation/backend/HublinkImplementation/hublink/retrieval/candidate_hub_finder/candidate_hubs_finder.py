from abc import ABC, abstractmethod
from typing import List

from ..models.processed_question import ProcessedQuestion
from ...core.models.hub_path import HubPath


class CandidateHubsFinder(ABC):
    """
    Component interface for finding candidate hubs for a processed question.
    """

    @abstractmethod
    def find_candidate_hubs(self, processed_question: ProcessedQuestion) -> dict[str, List[HubPath]]:
        """
        Finds the candidate hubs for the given processed question.

        Args:
            processed_question (ProcessedQuestion): The processed question
                containing the embeddings and other information.

        Returns:
            dict[str, List[HubPath]]: A dictionary mapping hub IDs to lists
                of HubPaths. Each list contains unique logical paths by
                path_hash, with dense_score and the current final score
                populated for every scored path.
        """
        ...
