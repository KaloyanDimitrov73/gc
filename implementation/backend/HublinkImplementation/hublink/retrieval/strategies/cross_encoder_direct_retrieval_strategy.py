from typing import List, Optional
from typing_extensions import override

from core.logging.logging import get_logger

from ..cross_encoder.cross_encoder_scorer import CrossEncoderScorer
from ...core.sparse_index.sparse_storage_manager import SparseStorageManager
from ..models.processed_question import ProcessedQuestion
from ...core.models.hub_path import HubPath
from .base_retrieval_strategy import RetrievalStrategyData
from .direct_retrieval_strategy import DirectRetrievalStrategy

logger = get_logger(__name__)


class CrossEncoderDirectRetrievalStrategy(DirectRetrievalStrategy):
    """
    Extends DirectRetrievalStrategy by rescoring each candidate hub's
    paths with a cross-encoder after they are filled/truncated, so that
    downstream pruning uses cross-encoder relevance scores instead of the
    raw candidate-hub-finder scores.
    """

    def __init__(
            self,
            retrieval_data: RetrievalStrategyData,
            sparse_storage_manager: Optional[SparseStorageManager] = None
    ) -> None:
        super().__init__(
            retrieval_data,
            sparse_storage_manager=sparse_storage_manager,
        )
        self.cross_encoder = CrossEncoderScorer()

    @override
    def _fill_or_remove_paths(self,
                              processed_question: ProcessedQuestion,
                              candidate_hubs: dict[str, List[HubPath]],
                              path_threshold: int) -> dict[str, List[HubPath]]:
        hubs = super()._fill_or_remove_paths(
            processed_question=processed_question,
            candidate_hubs=candidate_hubs,
            path_threshold=path_threshold
        )

        for paths in hubs.values():
            if not paths:
                continue
            scores = self.cross_encoder.score_batch(
                processed_question.question,
                [path.path_text for path in paths]
            )
            for path, score in zip(paths, scores):
                path.score = score
            paths.sort(key=lambda path: path.score if path.score is not None else float("-inf"), reverse=True)

        return hubs
