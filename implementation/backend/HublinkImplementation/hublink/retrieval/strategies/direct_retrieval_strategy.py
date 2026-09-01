import threading
from typing import List, Optional
from typing_extensions import override

from core.data.models import RetrievalAnswer
from core.logging.logging import get_logger
from .base_retrieval_strategy import BaseRetrievalStrategy, RetrievalStrategyData
from ..candidate_hub_finder.ann_hub_finder import ANNHubFinder
from ..candidate_hub_finder.candidate_hubs_finder import CandidateHubsFinder
from ..candidate_hub_finder.hybrid_search.bm25_fusion_decorator import (
    Bm25FusionDecorator,
)
from ..candidate_hub_finder.hybrid_search.splade_fusion_decorator import (
    SpladeFusionDecorator,
)
from ..candidate_hub_finder.hybrid_search.sparse_evidence_merger import (
    SparseEvidenceMerger,
)
from ..candidate_hub_finder.hybrid_search.rrf import (
    normalize_rrf_score,
    rrf_score_from_ranks,
)
from ...core.sparse_index.sparse_storage_manager import SparseStorageManager
from ..models.processed_question import ProcessedQuestion
from ...core.models.entity_with_direction import EntityWithDirection
from ...core.models.hub import Hub
from ...core.models.hub_path import HubPath

logger = get_logger(__name__)


class DirectRetrievalStrategy(BaseRetrievalStrategy):
    """
    This retrieval strategy directly retrieves the HubPaths based on the embeddings
    from the question from the vector store without considering the hubs beeing
    reachable from a topic entity. Candidate hub finding is delegated to a
    pluggable CandidateHubsFinder component, defaulting to ANNHubFinder
    (dense ANN search over the question embeddings).

    Args:
        retrieval_data (RetrievalStrategyData): The data required for the
            retrieval strategy.
        sparse_storage_manager: Loaded sparse-index storage manager.
    """

    def __init__(
            self,
            retrieval_data: RetrievalStrategyData,
            sparse_storage_manager: Optional[SparseStorageManager] = None
    ) -> None:
        super().__init__(retrieval_data)
        self.sparse_storage_manager = sparse_storage_manager
        self.candidate_hub_finder: CandidateHubsFinder = ANNHubFinder(
            hub_storage_manager=self.hub_storage_manager,
            number_of_hubs=self.settings.number_of_hubs,
            top_paths_to_keep=self.settings.top_paths_to_keep,
        )
        evidence_merger = SparseEvidenceMerger(
            hub_storage_manager=self.hub_storage_manager,
        )
        if (
            self.settings.use_bm25_hybrid_search
            and self.sparse_storage_manager is not None
            and self.sparse_storage_manager.has_bm25
        ):
            self.candidate_hub_finder = Bm25FusionDecorator(
                candidate_hub_finder=self.candidate_hub_finder,
                hub_storage_manager=self.hub_storage_manager,
                sparse_storage_manager=self.sparse_storage_manager,
                top_paths_to_keep=self.settings.top_paths_to_keep,
                number_of_hubs=self.settings.number_of_hubs,
                evidence_merger=evidence_merger,
            )
        if (
            self.settings.use_splade_hybrid_search
            and self.sparse_storage_manager is not None
            and self.sparse_storage_manager.has_splade
        ):
            self.candidate_hub_finder = SpladeFusionDecorator(
                candidate_hub_finder=self.candidate_hub_finder,
                hub_storage_manager=self.hub_storage_manager,
                sparse_storage_manager=self.sparse_storage_manager,
                top_paths_to_keep=self.settings.top_paths_to_keep,
                number_of_hubs=self.settings.number_of_hubs,
                evidence_merger=evidence_merger,
            )

    @override
    def _run_retrieval(self, processed_question: ProcessedQuestion, conversation_history: Optional[List[str]] = None, cancel_event: Optional[threading.Event] = None) -> Optional[RetrievalAnswer]:
        """Runs the main loop of the retrieval strategy."""

        logger.info("Searching candidate hubs")
        self.progress_handler.add_task(string_id="candidate_hub_search", description="Searching candidate hubs", total=1, reset=True)
        candidate_hubs = self._find_candidate_hubs(processed_question)
        self.progress_handler.finish_by_string_id("candidate_hub_search")

        if cancel_event is not None and cancel_event.is_set():
            logger.info("Canceling after: Searching candidate hubs")
            return None

        logger.info("Filling paths")
        self.progress_handler.add_task(string_id="path_filling", description="Filling paths", total=len(candidate_hubs), reset=True)
        logger.info("Scoring and limiting candidate paths")
        self.progress_handler.add_task(string_id="path_scoring", description="Scoring candidate paths", total=len(candidate_hubs), reset=True)
        candidate_hubs = self._fill_or_remove_paths(
            processed_question=processed_question,
            candidate_hubs=candidate_hubs,
            path_threshold=self.settings.top_paths_to_keep
        )
        self.progress_handler.finish_by_string_id("path_scoring")

        hubs = self._convert_to_hubs(
            candidate_hubs=candidate_hubs
        )

        if cancel_event is not None and cancel_event.is_set():
            logger.info("Canceling after: Filling paths")
            return None

        logger.info("Pruning hubs sorted by their weighted hub score")
        self.progress_handler.add_task(string_id="hub_pruning", description="Pruning hubs", total=1, reset=True)
        filtered_candidates = self._prune_hubs(
            hubs=hubs,
            alpha=self.settings.path_weight_alpha,
        )
        self.progress_handler.finish_by_string_id("hub_pruning")

        if cancel_event is not None and cancel_event.is_set():
            logger.info("Canceling after: Pruning hubs")
            return None

        logger.info("Generate partial answers")

        partial_answers = (
            self._get_hub_answers_directly(filtered_candidates)
            if self.settings.use_direct_final_answer
            else self._get_partial_answers(
                processed_question=processed_question,
                hub_scoring=filtered_candidates
            )
        )

        if cancel_event is not None and cancel_event.is_set():
            logger.info("Canceling after: Partial answers")
            return None

        # If we have no partial answers, we return an empty answer
        # else we try to generate a final answer based on the partial
        # answers
        if len(partial_answers) > 0:
            logger.debug("Found answers in hubs: %s", [
                         answer.hub_answer for answer in partial_answers])
            final_answer = self.answer_generator.get_final_answer(
                question=processed_question.question,
                hub_answers=partial_answers,
                settings=self.settings,
                conversation_history=conversation_history
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
            List[Hub]: A list of Hub objects
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
        Finds candidate hubs by delegating to the configured CandidateHubsFinder
        component.

        Args:
            processed_question (ProcessedQuestion): The processed question
                containing the embeddings and other information.

        Returns:
            dict[str, List[HubPath]]: A dictionary mapping hub IDs to lists
                of HubPaths.
        """
        return self.candidate_hub_finder.find_candidate_hubs(processed_question)

    def _fill_or_remove_paths(self,
                              processed_question: ProcessedQuestion,
                              candidate_hubs: dict[str, List[HubPath]],
                              path_threshold: int) -> dict[str, List[HubPath]]:
        """Scores candidate paths and applies the per-hub path limit.

        Dense candidate hubs are already filled and assigned immutable dense
        ranks by ``ANNHubFinder`` before any sparse channel runs. No retrieval
        is performed here because doing so would add dense evidence after the
        independent channel rankings have been frozen.

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
        return self._score_and_limit_paths(
            candidate_hubs=candidate_hubs,
            path_threshold=path_threshold,
        )

    def _score_and_limit_paths(
            self,
            candidate_hubs: dict[str, List[HubPath]],
            path_threshold: int) -> dict[str, List[HubPath]]:
        """Assigns globally comparable path scores and applies the per-hub cap."""
        identified_paths = [
            (hub_id, path)
            for hub_id, paths in candidate_hubs.items()
            for path in paths
        ]
        if not identified_paths:
            return candidate_hubs

        sparse_channels = sorted({
            channel
            for _, path in identified_paths
            for channel in path.sparse_ranks
        })
        if not sparse_channels:
            for _, path in identified_paths:
                path.score = path.dense_score
        else:
            has_dense_ranking = any(
                path.dense_rank is not None
                for _, path in identified_paths
            )
            channel_count = len(sparse_channels) + int(has_dense_ranking)

            for _, path in identified_paths:
                ranks = []
                if path.dense_rank is not None:
                    ranks.append(path.dense_rank)
                ranks.extend(
                    path.sparse_ranks[channel]
                    for channel in sparse_channels
                    if channel in path.sparse_ranks
                )
                raw_score = rrf_score_from_ranks(
                    ranks=ranks,
                    k=self.settings.rrf_k,
                )
                path.score = normalize_rrf_score(
                    score=raw_score,
                    channel_count=channel_count,
                    k=self.settings.rrf_k,
                )

        scored_candidate_hubs: dict[str, List[HubPath]] = {}
        for hub_id, paths in candidate_hubs.items():
            paths.sort(
                key=lambda path: (
                    -path.score
                    if path.score is not None
                    else float("inf"),
                    path.path_hash,
                ),
            )
            scored_candidate_hubs[hub_id] = paths[:path_threshold]
        return scored_candidate_hubs
