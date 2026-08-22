from typing import List
from typing_extensions import override

from core.logging.logging import get_logger

from ..models.processed_question import ProcessedQuestion
from ...core.hub_storage_manager import HubStorageManager
from ...core.models.hub_path import HubPath
from .candidate_hubs_finder import CandidateHubsFinder

logger = get_logger(__name__)


class ANNHubFinder(CandidateHubsFinder):
    """
    Finds candidate hubs using dense approximate nearest neighbor (ANN)
    search over the question embeddings via the vector store.
    """

    def __init__(self,
                 hub_storage_manager: HubStorageManager,
                 number_of_hubs: int,
                 top_paths_to_keep: int):
        self.hub_storage_manager = hub_storage_manager
        self.number_of_hubs = number_of_hubs
        self.top_paths_to_keep = top_paths_to_keep

    @override
    def find_candidate_hubs(self, processed_question: ProcessedQuestion) -> dict[str, List[HubPath]]:
        """
        The main retrieval algorithm for gathering HubPaths from the vector store
        based on dense ANN search. It uses the embeddings from the question
        to find the candidate hubs and their paths directly from the vector store.

        Args:
            processed_question (ProcessedQuestion): The processed question
                containing the embeddings and other information.

        Returns:
            dict[str, List[HubPath]]: A dictionary mapping hub IDs to lists
                of HubPaths.
        """
        candidate_hubs: dict[str, List[HubPath]] = {}
        while len(candidate_hubs) < self.number_of_hubs:
            retrieval_amount = self.top_paths_to_keep

            try:
                hubs_to_exclude = list(candidate_hubs.keys())

                logger.info("Use the embeddings from the question to find the candidate hubs")
                logger.info("List of candidate hubs: %s", len(hubs_to_exclude))
                logger.info("Question Components: %s", processed_question.components)
                results = self.hub_storage_manager.similarity_search_hubs(
                    query_embeddings=processed_question.embeddings,
                    excluded_hub_ids=hubs_to_exclude,
                    n_results=retrieval_amount
                )
            except Exception as e:
                logger.error(f"Error during similarity_search_hubs: {e}")
                break

            if not results or len(results) == 0:
                logger.debug(
                    "No more results returned from similarity_search_hubs.")
                break

            for hub_id, hub_paths in results.items():
                if hub_id in candidate_hubs:
                    logger.debug(
                        f"Hub {hub_id} already in candidate_hubs; skipping.")
                    continue

                candidate_hubs[hub_id] = hub_paths

        return candidate_hubs
