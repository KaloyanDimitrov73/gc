from hublink.core.hub_storage_manager import HubStorageManager
from hublink.core.models.hub_path import HubPath
from hublink.core.sparse_index.sparse_search_result import SparsePathHit
from hublink.retrieval.models.processed_question import ProcessedQuestion

from .rrf import two_way_rrf


class SparseEvidenceMerger:
    """Merges existing evidence paths with one sparse channel's path hits."""

    def __init__(
            self,
            hub_storage_manager: HubStorageManager,
            rrf_k: int = 60):
        self.hub_storage_manager = hub_storage_manager
        self.rrf_k = rrf_k

    def merge(
            self,
            processed_question: ProcessedQuestion,
            existing_paths: list[HubPath],
            sparse_hits: list[SparsePathHit],
            sparse_channel: str = "sparse") -> list[HubPath]:
        if not sparse_hits:
            return existing_paths

        sparse_hashes = [hit.path_hash for hit in sparse_hits]
        sparse_paths_by_hash = (
            self.hub_storage_manager.retrieve_scored_hub_paths_by_keys(
                path_hashes=sparse_hashes,
                query_embeddings=processed_question.embeddings
            )
        )

        paths_by_hash = {
            path.path_hash: path
            for path in existing_paths
        }

        for hit in sparse_hits:
            path = paths_by_hash.get(hit.path_hash)
            if path is None:
                path = sparse_paths_by_hash.get(hit.path_hash)
            if path is None:
                continue
            paths_by_hash[hit.path_hash] = path
            path.sparse_scores[sparse_channel] = hit.score
            global_rank = (
                hit.global_rank
                if hit.global_rank is not None
                else hit.rank
            )
            path.sparse_ranks[sparse_channel] = global_rank

        existing_ranking = [
            path.path_hash for path in existing_paths
        ]
        sparse_ranking = [
            hit.path_hash
            for hit in sparse_hits
            if hit.path_hash in paths_by_hash
        ]
        fused_path_hashes = two_way_rrf(
            wrapped_ranking=existing_ranking,
            sparse_ranking=sparse_ranking,
            k=self.rrf_k
        )

        return [
            paths_by_hash[path_hash]
            for path_hash in fused_path_hashes
            if path_hash in paths_by_hash
        ]
