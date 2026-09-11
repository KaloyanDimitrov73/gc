from typing import Dict, List

from hublink.core.models.hub_path import HubPath
from knowledge_base.sparse_index_store.sparse_index_store import (
    SparseIndexDocument,
)


def collect_sparse_index_documents_from_paths(
        paths_by_hub: Dict[str, List[HubPath]]) -> List[SparseIndexDocument]:
    """Collects sparse documents from paths already cached by dense indexing."""
    documents: List[SparseIndexDocument] = []

    for hub_id, paths in paths_by_hub.items():
        for path in paths:
            if not path.path_text:
                continue
            documents.append(_to_sparse_document(hub_id, path))

    return documents


def _to_sparse_document(
        hub_id: str,
        path: HubPath) -> SparseIndexDocument:
    return SparseIndexDocument(
        record_id=f"{hub_id}:{path.path_hash}",
        text=path.path_text,
        hub_id=hub_id,
        path_hash=path.path_hash,
    )
