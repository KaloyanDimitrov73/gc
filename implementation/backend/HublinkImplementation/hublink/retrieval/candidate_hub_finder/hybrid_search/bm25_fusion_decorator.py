from typing import List, Optional

from typing_extensions import override

from core.logging.logging import get_logger
from hublink.core.hub_storage_manager import HubStorageManager
from hublink.core.models.hub_path import HubPath
from hublink.core.sparse_index.sparse_storage_manager import SparseStorageManager
from hublink.retrieval.candidate_hub_finder.candidate_hubs_finder import (
    CandidateHubsFinder,
)
from hublink.retrieval.candidate_hub_finder.candidate_hubs_finder_decorator import (
    CandidateHubsFinderDecorator,
)
from hublink.retrieval.models.processed_question import ProcessedQuestion

from .sparse_evidence_merger import SparseEvidenceMerger

logger = get_logger(__name__)


class Bm25FusionDecorator(CandidateHubsFinderDecorator):
    """Adds BM25 candidates and path evidence to wrapped candidates."""

    def __init__(self,
                 candidate_hub_finder: CandidateHubsFinder,
                 hub_storage_manager: HubStorageManager,
                 sparse_storage_manager: SparseStorageManager,
                 top_paths_to_keep: int,
                 number_of_hubs: int,
                 rrf_k: int = 60,
                 evidence_merger: Optional[SparseEvidenceMerger] = None):
        super().__init__(candidate_hub_finder)
        self.hub_storage_manager = hub_storage_manager
        self.sparse_storage_manager = sparse_storage_manager
        self.top_paths_to_keep = top_paths_to_keep
        self.number_of_hubs = number_of_hubs
        self.rrf_k = rrf_k
        self.evidence_merger = evidence_merger or SparseEvidenceMerger(
            hub_storage_manager=hub_storage_manager,
            rrf_k=rrf_k
        )

    @override
    def find_candidate_hubs(
            self,
            processed_question: ProcessedQuestion) -> dict[str, List[HubPath]]:
        dense_hubs = self.candidate_hub_finder.find_candidate_hubs(
            processed_question)

        logger.info(
            "Running BM25 sparse search (number_of_hubs=%d)",
            self.number_of_hubs,
        )
        bm25_result = self.sparse_storage_manager.search_bm25(
            query_text=processed_question.question,
            top_hubs=self.number_of_hubs,
            paths_per_hub=self.top_paths_to_keep
        )
        if not bm25_result.ranked_hub_ids:
            return dense_hubs

        candidate_hub_ids = list(dense_hubs)
        candidate_hub_ids.extend(
            hub_id
            for hub_id in bm25_result.ranked_hub_ids
            if hub_id not in dense_hubs
        )

        candidate_hubs: dict[str, List[HubPath]] = {}
        for hub_id in candidate_hub_ids:
            existing_paths = dense_hubs.get(hub_id, [])
            if not existing_paths:
                try:
                    existing_paths = (
                        self.hub_storage_manager
                        .similarity_search_by_hub_entity(
                            query_embeddings=processed_question.embeddings,
                            hub_entity_id=hub_id,
                            n_results=self.top_paths_to_keep
                        )
                    )
                except Exception as error:
                    logger.error(
                        "Error fetching dense paths for BM25-only hub %s: %s",
                        hub_id,
                        error
                    )

            merged_paths = self.evidence_merger.merge(
                processed_question=processed_question,
                existing_paths=existing_paths,
                sparse_hits=bm25_result.hits_by_hub.get(hub_id, []),
                sparse_channel="bm25",
            )
            if merged_paths:
                candidate_hubs[hub_id] = merged_paths

        return candidate_hubs
