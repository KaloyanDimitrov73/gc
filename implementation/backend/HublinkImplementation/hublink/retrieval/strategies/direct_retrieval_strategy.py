from typing import List, Optional
from typing_extensions import override

from core.data.models import RetrievalAnswer
from core.logging.logging import get_logger

from .base_retrieval_strategy import BaseRetrievalStrategy
from ..models.processed_question import ProcessedQuestion
from ...core.models.entity_with_direction import EntityWithDirection
from ...core.models.hub import Hub
from ...core.models.hub_path import HubPath

logger = get_logger(__name__)


class DirectRetrievalStrategy(BaseRetrievalStrategy):
    """
    This retrieval strategy directly retrieves the HubPaths based on the embeddings
    from the question from the vector store without considering the hubs beeing 
    reachable from a topic entity.
    """

    @override
    def _run_retrieval(self, processed_question: ProcessedQuestion) -> Optional[RetrievalAnswer]:
        """Runs the main loop of the retrieval strategy."""

        logger.info("Searching candidate hubs")
        self.progress_handler.add_task(string_id="candidate_hub_search", description="Searching candidate hubs", total=1, reset=True)
        candidate_hubs = self._find_candidate_hubs(processed_question)
        self.progress_handler.finish_by_string_id("candidate_hub_search")

        logger.info("Filling paths")
        self.progress_handler.add_task(string_id="path_filling", description="Filling paths", total=len(candidate_hubs), reset=True)
        candidate_hubs = self._fill_or_remove_paths(
            processed_question=processed_question,
            candidate_hubs=candidate_hubs,
            path_threshold=self.settings.top_paths_to_keep
        )
        self.progress_handler.finish_by_string_id("path_filling")

        hubs = self._convert_to_hubs(
            candidate_hubs=candidate_hubs
        )

        logger.info("Pruning hubs sorted by their weighted hub score")
        self.progress_handler.add_task(string_id="hub_pruning", description="Pruning hubs", total=1, reset=True)
        filtered_candidates = self._prune_hubs(
            hubs=hubs,
            alpha=self.settings.path_weight_alpha,
        )
        self.progress_handler.finish_by_string_id("hub_pruning")

        logger.info("Generate partial answers")

        partial_answers = (
            self._get_hub_answers_directly(filtered_candidates)
            if self.settings.use_direct_final_answer
            else self._get_partial_answers(
                processed_question=processed_question,
                hub_scoring=filtered_candidates
            )
        )

        # If we have no partial answers, we return an empty answer
        # else we try to generate a final answer based on the partial
        # answers
        if len(partial_answers) > 0:
            logger.debug("Found answers in hubs: %s", [
                         answer.hub_answer for answer in partial_answers])
            final_answer = self.answer_generator.get_final_answer(
                question=processed_question.question,
                hub_answers=partial_answers,
                settings=self.settings
            )
            if final_answer:
                logger.debug("Final answer found: %s",
                             final_answer.retriever_answer)
                return final_answer
            logger.debug("Insufficient information in hubs")

        return RetrievalAnswer(contexts=[], retriever_answer=None)

    def _convert_to_hubs(
            self,
            candidate_hubs: dict[str, List[HubPath]]) -> List[Hub]:
        """
        Converts the hubpaths into Hub objects.

        Args:
            candidate_hubs (dict[str, List[HubPath]]): The candidate hub paths
                clustered by their root id.

        Returns:
            List[Huby]: A list of Hub objects
                representing the candidate hubs and their paths.
        """
        converted_hubs: List[Hub] = []
        for hub_id, hub_paths in candidate_hubs.items():
            converted_hubs.append(Hub(
                root_entity=EntityWithDirection(
                    entity=self.graph.get_entity_by_id(hub_id),
                    left=False,
                    path_from_topic=[]
                ),
                paths=hub_paths
            ))
        return converted_hubs

    def _find_candidate_hubs(self, processed_question: ProcessedQuestion) -> dict[str, List[HubPath]]:
        """
        The main retrieval algorithm for gathering HubPaths from the vector store
        based on the direct strategy. It uses the embeddings from the question
        to find the candidate hubs and their paths directly from the vector store.

        Args:
            processed_question (ProcessedQuestion): The processed question
                containing the embeddings and other information.

        Returns:
            dict[str, List[HubPath]]: A dictionary mapping hub IDs to lists
                of HubPaths.
        """
        candidate_hubs: dict[str, List[HubPath]] = {}
        unique_path_hashes = set()
        while len(candidate_hubs) < self.settings.number_of_hubs:
            retrieval_amount = self.settings.top_paths_to_keep

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

                if hub_id not in candidate_hubs:
                    candidate_hubs[hub_id] = []

                for hub_path in hub_paths:
                    if hub_path.path_hash in unique_path_hashes:
                        continue
                    unique_path_hashes.add(hub_path.path_hash)
                    candidate_hubs[hub_id].append(hub_path)

        return candidate_hubs

    def _fill_or_remove_paths(self,
                              processed_question: ProcessedQuestion,
                              candidate_hubs: dict[str, List[HubPath]],
                              path_threshold: int) -> dict[str, List[HubPath]]:
        """
        For the given candidate hubs, this function ensures that each hub has
        the amount of paths specified in the threshold. If the hub has more
        paths than the threshold, it will be truncated. If it has less, it will
        try to fill the paths by searching for more paths in the vector store.         

        Args:
            processed_question (ProcessedQuestion): The processed question
                containing the embeddings and other information.
            candidate_hubs (dict[str, List[HubPath]]): The candidate hubs
                with their paths.
            path_threshold (int): The desired number of paths for each hub.

        Returns:
            dict[str, List[HubPath]]: A dictionary mapping hub IDs to lists
                of HubPaths, ensuring that each hub has the desired number
                of paths.
        """
        # Now we make sure that for each candidate hub, we have the desired amount
        # of paths
        prepared_candidate_hubs = {}
        for hub_id, current_hub_paths in list(candidate_hubs.items()):

            if len(current_hub_paths) > path_threshold:
                prepared_candidate_hubs[hub_id] = current_hub_paths[:path_threshold]
                continue
            if len(current_hub_paths) == path_threshold:
                prepared_candidate_hubs[hub_id] = current_hub_paths
                continue
            # If we have less paths than desired, we need to get more
            try:
                paths = self._get_hub_paths_for_hub(
                    processed_question=processed_question,
                    hub_id=hub_id
                )
                if len(paths) > path_threshold:
                    prepared_candidate_hubs[hub_id] = paths[:path_threshold]
                else:
                    prepared_candidate_hubs[hub_id] = paths

            except Exception as e:
                logger.error(
                    f"Error during similarity_search_by_hub_entity: {e}")
                continue
        return prepared_candidate_hubs
