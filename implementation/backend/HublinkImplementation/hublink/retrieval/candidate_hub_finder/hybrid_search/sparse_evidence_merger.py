from hublink.core.hub_storage_manager import HubStorageManager
from hublink.core.models.hub_path import HubPath
from hublink.core.sparse_index.sparse_search_result import SparsePathHit


class SparseEvidenceMerger:
    """Adds one sparse channel's independently retrieved path evidence."""

    def __init__(
            self,
            hub_storage_manager: HubStorageManager):
        self.hub_storage_manager = hub_storage_manager

    def merge(
            self,
            existing_paths: list[HubPath],
            sparse_hits: list[SparsePathHit],
            sparse_channel: str = "sparse") -> list[HubPath]:
        if not sparse_hits:
            return existing_paths

        paths_by_hash = {
            path.path_hash: path
            for path in existing_paths
        }
        sparse_hashes = [
            hit.path_hash
            for hit in sparse_hits
            if hit.path_hash not in paths_by_hash
        ]
        sparse_paths_by_hash = (
            self.hub_storage_manager.retrieve_hub_paths_by_keys(
                hash_keys=sparse_hashes,
            )
            if sparse_hashes
            else {}
        )
        merged_paths = list(existing_paths)

        for hit in sparse_hits:
            path = paths_by_hash.get(hit.path_hash)
            if path is None:
                path = sparse_paths_by_hash.get(hit.path_hash)
            if path is None:
                continue
            if hit.path_hash not in paths_by_hash:
                merged_paths.append(path)
            paths_by_hash[hit.path_hash] = path
            path.sparse_scores[sparse_channel] = hit.score
            global_rank = (
                hit.global_rank
                if hit.global_rank is not None
                else hit.rank
            )
            path.sparse_ranks[sparse_channel] = global_rank

        return merged_paths
